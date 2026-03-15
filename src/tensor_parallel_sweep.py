#!/usr/bin/env python3
"""
Tensor Parallel GEMM Sweep — two experiments:

Experiment 1: Fix TP_DEGREE=4, sweep matrix sizes (2048, 4096, 8192, 16384)
Experiment 2: Fix SIZE=4096, sweep TP_DEGREE (2, 4, 8, 16, 32)

Each TP degree runs in a subprocess to avoid CCOM communicator size conflicts.

Usage:
    python tensor_parallel_sweep.py
    python tensor_parallel_sweep.py --point tp 4096 4    # single point
    python tensor_parallel_sweep.py --point dp 4096      # single DP point
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

import ml_dtypes
import numpy as np

import nkipy.distributed.collectives as cc
import spike
from nkipy.core.compile import compile_to_neff, nkipy_compiler_args
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel
from spike import SpikeModel, SystemTraceSession
from spike.spike_tensor import SpikeTensor


DTYPE = ml_dtypes.float8_e5m2
WARMUP_ITER = 5
BENCH_ITER = 10
BUILD_DIR = "/tmp/build/tp_sweep"


# ─── Compile helper ─────────────────────────────────────────────────────

def compile_kernel(kernel_fn, *proto_args, name="kernel"):
    traced = NKIPyKernel.trace(kernel_fn)
    traced.specialize(*proto_args)
    hlo = traced._code
    h = hashlib.sha256()
    h.update(hlo.to_proto().SerializeToString())
    h.update(nkipy_compiler_args.encode("utf-8"))
    content_hash = h.hexdigest()[:12]
    output_dir = f"{BUILD_DIR}/{name}_{content_hash}"
    neff_path = f"{output_dir}/{name}.neff"
    if os.path.exists(neff_path):
        pass  # cached
    else:
        compile_to_neff(traced, output_dir=output_dir, neff_name=f"{name}.neff",
                        additional_compiler_args=nkipy_compiler_args, save_artifacts=True)
    return neff_path


# ─── Kernels ─────────────────────────────────────────────────────────────

def make_row_parallel_kernel(tp_degree):
    def kernel(A_shard, B_shard):
        C_partial = A_shard @ B_shard
        C = cc.all_reduce(C_partial, replica_groups=[list(range(tp_degree))],
                          reduce_op=np.add)
        return C
    kernel.__name__ = f"row_par_tp{tp_degree}"
    return kernel


# ─── Trace parsing ──────────────────────────────────────────────────────

def parse_trace(events_json):
    if not events_json:
        return []
    root = json.loads(events_json)
    starts = {}
    durations_ms = []
    for ev in root.get("events", []):
        if ev.get("event_type") != "nc_exec_running":
            continue
        phase, tid = ev.get("phase"), ev.get("tracking_id")
        nc_ts = ev.get("data", {}).get("nc_timestamp_ns")
        if nc_ts is None or tid is None:
            continue
        if phase == "start":
            starts[tid] = nc_ts
        elif phase == "stop":
            s = starts.pop(tid, None)
            if s is not None:
                durations_ms.append((nc_ts - s) / 1_000_000.0)
    return durations_ms


# ─── Worker ──────────────────────────────────────────────────────────────

def run_worker(model, inputs, outputs, barrier_warmup, barrier_start, barrier_done,
               core_idx, round_times):
    for _ in range(WARMUP_ITER):
        model(inputs=inputs, outputs=outputs)
    barrier_warmup.wait()
    barrier_start.wait()
    per_round = []
    for _ in range(BENCH_ITER):
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        per_round.append((time.perf_counter() - t0) * 1000.0)
    round_times[core_idx] = per_round
    barrier_done.wait()


# ─── Single point runner ────────────────────────────────────────────────

def run_dp_point(size):
    """Run DP (1 core) benchmark, print JSON result to stdout."""
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"
    spike.configure(visible_cores=[0])

    def kernel(A, B):
        return A @ B

    np.random.seed(42)
    A_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)
    B_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)

    dk = DeviceKernel.compile_and_load(kernel, A_proto, B_proto,
                                       name=f"dp_{size}", use_cached_if_exists=True)

    A = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)
    B = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)
    out = np.zeros((size, size), dtype=DTYPE)
    inputs = {"A": SpikeTensor.from_numpy(A, "A", core_id=0),
              "B": SpikeTensor.from_numpy(B, "B", core_id=0)}
    outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=0)}

    barrier_warmup = threading.Barrier(2)
    barrier_start = threading.Barrier(2)
    barrier_done = threading.Barrier(2)
    round_times = {}

    t = threading.Thread(target=run_worker,
                         args=(dk, inputs, outputs, barrier_warmup, barrier_start,
                               barrier_done, 0, round_times))

    with SystemTraceSession() as trace:
        t.start()
        barrier_warmup.wait()
        trace.drain_events()
        barrier_start.wait()
        barrier_done.wait()
        events_json = trace.fetch_events_json()
    t.join()

    dev_durations = parse_trace(events_json)
    all_ms = round_times.get(0, [])
    host_mean = sum(all_ms) / len(all_ms) if all_ms else 0
    dev_mean = sum(dev_durations) / len(dev_durations) if dev_durations else 0

    result = {"mode": "dp", "size": size, "tp": 1,
              "host_mean_ms": host_mean, "dev_mean_ms": dev_mean}
    print(f"RESULT:{json.dumps(result)}")


def run_tp_point(size, tp_degree):
    """Run TP benchmark, print JSON result to stdout."""
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    spike.configure(visible_cores=range(tp_degree))

    k_shard = size // tp_degree
    kernel_fn = make_row_parallel_kernel(tp_degree)
    A_proto = ((np.random.rand(size, k_shard) - 0.5) * 2).astype(DTYPE)
    B_proto = ((np.random.rand(k_shard, size) - 0.5) * 2).astype(DTYPE)

    neff_path = compile_kernel(kernel_fn, A_proto, B_proto,
                               name=f"rp_{size}_tp{tp_degree}")

    models = []
    for cid in range(tp_degree):
        models.append(SpikeModel.load_from_neff(
            neff_path, core_id=cid,
            cc_enabled=True, rank_id=cid, world_size=tp_degree))

    A_full = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)
    B_full = ((np.random.rand(size, size) - 0.5) * 2).astype(DTYPE)
    per_core_io = []
    for cid in range(tp_degree):
        A_shard = A_full[:, cid * k_shard:(cid + 1) * k_shard].copy()
        B_shard = B_full[cid * k_shard:(cid + 1) * k_shard, :].copy()
        out = np.zeros((size, size), dtype=DTYPE)
        inputs = {"A_shard": SpikeTensor.from_numpy(A_shard, "A_shard", core_id=cid),
                  "B_shard": SpikeTensor.from_numpy(B_shard, "B_shard", core_id=cid)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=cid)}
        per_core_io.append((inputs, outputs))

    barrier_warmup = threading.Barrier(tp_degree + 1)
    barrier_start = threading.Barrier(tp_degree + 1)
    barrier_done = threading.Barrier(tp_degree + 1)
    round_times = {}

    threads = [
        threading.Thread(target=run_worker,
                         args=(models[i], per_core_io[i][0], per_core_io[i][1],
                               barrier_warmup, barrier_start, barrier_done,
                               i, round_times))
        for i in range(tp_degree)
    ]

    with SystemTraceSession() as trace:
        for t in threads:
            t.start()
        barrier_warmup.wait()
        trace.drain_events()
        barrier_start.wait()
        barrier_done.wait()
        events_json = trace.fetch_events_json()

    for t in threads:
        t.join()

    dev_durations = parse_trace(events_json)
    all_ms = []
    for cid in sorted(round_times):
        all_ms.extend(round_times[cid])
    host_mean = sum(all_ms) / len(all_ms) if all_ms else 0
    dev_mean = sum(dev_durations) / len(dev_durations) if dev_durations else 0

    result = {"mode": "tp", "size": size, "tp": tp_degree,
              "host_mean_ms": host_mean, "dev_mean_ms": dev_mean}
    print(f"RESULT:{json.dumps(result)}")


# ─── Subprocess launcher ────────────────────────────────────────────────

def run_subprocess(mode, size, tp=1):
    """Launch a subprocess for one benchmark point, return parsed result."""
    cmd = [sys.executable, __file__, "--point", mode, str(size)]
    if mode == "tp":
        cmd.append(str(tp))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Parse result from stdout
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[7:])

    # Failed
    print(f"  WARNING: subprocess failed for {mode} size={size} tp={tp}")
    if proc.stderr:
        # Print last few lines of stderr
        stderr_lines = proc.stderr.strip().splitlines()
        for line in stderr_lines[-5:]:
            print(f"    {line}")
    return {"mode": mode, "size": size, "tp": tp,
            "host_mean_ms": 0, "dev_mean_ms": 0}


# ─── Main (orchestrator) ────────────────────────────────────────────────

def main_orchestrator():
    print("=" * 80)
    print("  Tensor Parallel GEMM Sweep")
    print("  Row-parallel (all_reduce), ASYNC_INFLIGHT=1")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════
    # Experiment 1: Fix TP=4, sweep matrix size
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: Matrix Size Sweep (TP_DEGREE=4)")
    print("=" * 80)

    tp_fixed = 4
    sizes = [2048, 4096, 8192, 16384]
    exp1_results = []

    for sz in sizes:
        print(f"\n  --- Size {sz}x{sz} ---")

        print(f"  [DP] 1 core:", end=" ", flush=True)
        dp = run_subprocess("dp", sz)
        print(f"host={dp['host_mean_ms']:.3f} ms, dev={dp['dev_mean_ms']:.3f} ms")

        print(f"  [TP={tp_fixed}] {tp_fixed} cores:", end=" ", flush=True)
        tp = run_subprocess("tp", sz, tp_fixed)
        print(f"host={tp['host_mean_ms']:.3f} ms, dev={tp['dev_mean_ms']:.3f} ms")

        if tp['host_mean_ms'] > 0:
            speedup = dp['host_mean_ms'] / tp['host_mean_ms']
            efficiency = speedup / tp_fixed * 100
        else:
            speedup = 0
            efficiency = 0

        exp1_results.append({
            "size": sz, "dp_ms": dp['host_mean_ms'], "tp_ms": tp['host_mean_ms'],
            "speedup": speedup, "efficiency": efficiency,
        })

    print(f"\n  {'─' * 70}")
    print(f"  Experiment 1 Results (TP_DEGREE={tp_fixed}):")
    print(f"  {'─' * 70}")
    print(f"  {'Size':>7} {'DP 1-core (ms)':>14} {'TP 4-core (ms)':>14} {'Speedup':>8} {'Efficiency':>10}")
    print(f"  {'─' * 70}")
    for r in exp1_results:
        print(f"  {r['size']:>7} {r['dp_ms']:>14.3f} {r['tp_ms']:>14.3f} {r['speedup']:>8.2f}x {r['efficiency']:>9.1f}%")
    print(f"  {'─' * 70}")

    # ═══════════════════════════════════════════════════════════════════
    # Experiment 2: Fix size=4096, sweep TP degree
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: TP Degree Sweep (SIZE=4096)")
    print("=" * 80)

    size_fixed = 4096
    tp_degrees = [2, 4, 8, 16, 32]

    print(f"\n  [DP baseline] 1 core:", end=" ", flush=True)
    dp_baseline = run_subprocess("dp", size_fixed)
    dp_ms = dp_baseline['host_mean_ms']
    print(f"host={dp_ms:.3f} ms")

    exp2_results = []
    for tp in tp_degrees:
        print(f"\n  [TP={tp}] {tp} cores:", end=" ", flush=True)
        r = run_subprocess("tp", size_fixed, tp)
        print(f"host={r['host_mean_ms']:.3f} ms, dev={r['dev_mean_ms']:.3f} ms")

        if r['host_mean_ms'] > 0:
            speedup = dp_ms / r['host_mean_ms']
            efficiency = speedup / tp * 100
            flops = 2 * size_fixed ** 3
            tflops = flops / (r['host_mean_ms'] * 1e-3) / 1e12
        else:
            speedup = efficiency = tflops = 0

        exp2_results.append({
            "tp": tp, "latency_ms": r['host_mean_ms'], "dev_ms": r['dev_mean_ms'],
            "speedup": speedup, "efficiency": efficiency, "tflops": tflops,
        })

    print(f"\n  {'─' * 70}")
    print(f"  Experiment 2 Results (SIZE={size_fixed}, DP baseline: {dp_ms:.3f} ms):")
    print(f"  {'─' * 70}")
    print(f"  {'TP':>4} {'Latency (ms)':>12} {'Dev (ms)':>10} {'Speedup':>8} {'Efficiency':>10} {'TFLOPS':>8}")
    print(f"  {'─' * 70}")
    for r in exp2_results:
        print(f"  {r['tp']:>4} {r['latency_ms']:>12.3f} {r['dev_ms']:>10.3f} {r['speedup']:>8.2f}x {r['efficiency']:>9.1f}% {r['tflops']:>8.2f}")
    print(f"  {'─' * 70}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    print("\n  Exp 1 — Larger matrices → better TP efficiency (TP=4):")
    for r in exp1_results:
        bar = "█" * max(1, int(r['efficiency'] / 2))
        print(f"    {r['size']:>5}x{r['size']}: {r['efficiency']:5.1f}% {bar}")

    print(f"\n  Exp 2 — TP degree scaling (SIZE=4096):")
    for r in exp2_results:
        bar = "█" * max(1, int(r['efficiency'] / 2))
        print(f"    TP={r['tp']:>2}: {r['speedup']:5.2f}x speedup, {r['efficiency']:5.1f}% eff {bar}")

    print(f"\n{'=' * 80}\n")


# ─── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--point" in sys.argv:
        # Subprocess mode: run single point
        idx = sys.argv.index("--point")
        mode = sys.argv[idx + 1]
        size = int(sys.argv[idx + 2])
        if mode == "tp":
            tp = int(sys.argv[idx + 3])
            run_tp_point(size, tp)
        else:
            run_dp_point(size)
    else:
        # Orchestrator mode
        main_orchestrator()
