#!/usr/bin/env python3
"""
Tensor Parallel GEMM — split a single large GEMM across multiple NeuronCores.

Two strategies:
  1. Column-parallel: split B along columns, each rank computes A @ B_shard,
     then all_gather to reconstruct full C.
  2. Row-parallel: split A along columns & B along rows, each rank computes
     partial sum A_shard @ B_shard, then all_reduce(add) for full C.

Compare with data-parallel baseline (each core does independent full GEMM).

Usage:
    python tensor_parallel_gemm.py
"""

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import ml_dtypes
import numpy as np

import nkipy.distributed.collectives as cc
import spike
from nkipy.core.compile import compile_to_neff, CompilationTarget
from nkipy.core.compile import nkipy_compiler_args
from nkipy.core.trace import NKIPyKernel
from nkipy.core.backend.hlo import HLOModule
from nkipy.runtime import DeviceKernel
from spike import SpikeModel, SystemTraceSession
from spike.spike_tensor import SpikeTensor


# ─── Configuration ──────────────────────────────────────────────────────
M = 4096
K = 4096
N = 4096
DTYPE = ml_dtypes.float8_e5m2
WARMUP_ITER = 5
BENCH_ITER = 10

# Tensor parallelism degree — must evenly divide N and K
TP_DEGREE = 4

BUILD_DIR = "/tmp/build/tp_gemm"


# ─── Compile-only helper (no loading) ──────────────────────────────────

def compile_kernel(kernel_fn, *proto_args, name="kernel"):
    """Trace, specialize, compile to NEFF. Returns neff_path (no device load)."""
    traced = NKIPyKernel.trace(kernel_fn)
    numpy_args = [a for a in proto_args]
    traced.specialize(*numpy_args)

    hlo = traced._code
    h = hashlib.sha256()
    h.update(hlo.to_proto().SerializeToString())
    h.update(nkipy_compiler_args.encode("utf-8"))
    content_hash = h.hexdigest()[:12]

    output_dir = f"{BUILD_DIR}/{name}_{content_hash}"
    neff_path = f"{output_dir}/{name}.neff"

    if os.path.exists(neff_path):
        print(f"  Using cached: {neff_path}")
    else:
        print(f"  Compiling {name}...")
        t0 = time.time()
        compile_to_neff(
            traced, output_dir=output_dir, neff_name=f"{name}.neff",
            additional_compiler_args=nkipy_compiler_args,
            save_artifacts=True,
        )
        print(f"  Compiled in {time.time() - t0:.1f}s")

    return neff_path


# ─── Kernel Definitions ────────────────────────────────────────────────

def kernel_data_parallel(A, B):
    """Baseline: full GEMM on each core independently."""
    return A @ B


def kernel_column_parallel(A, B_shard):
    """
    Column-parallel TP:
      Each rank holds B_shard of shape (K, N/tp).
      C_shard = A @ B_shard              → (M, N/tp)
      C = all_gather(C_shard, dim=1)     → (M, N)
    """
    C_shard = A @ B_shard
    C = cc.all_gather(
        C_shard,
        all_gather_dim=1,
        replica_groups=[list(range(TP_DEGREE))],
    )
    return C


def kernel_row_parallel(A_shard, B_shard):
    """
    Row-parallel TP:
      Each rank holds A_shard (M, K/tp) and B_shard (K/tp, N).
      C_partial = A_shard @ B_shard      → (M, N) partial sum
      C = all_reduce(C_partial, add)     → (M, N) full result
    """
    C_partial = A_shard @ B_shard
    C = cc.all_reduce(
        C_partial,
        replica_groups=[list(range(TP_DEGREE))],
        reduce_op=np.add,
    )
    return C


# ─── Trace Parsing ──────────────────────────────────────────────────────

def parse_trace(events_json: str):
    if not events_json:
        return [], []
    root = json.loads(events_json)
    events = root.get("events", [])
    starts = {}
    durations_ms = []
    timestamps = []

    for ev in events:
        if ev.get("event_type") != "nc_exec_running":
            continue
        phase = ev.get("phase")
        tid = ev.get("tracking_id")
        core_id = ev.get("data", {}).get("nc_id", -1)
        nc_ts = ev.get("data", {}).get("nc_timestamp_ns")
        if nc_ts is None or tid is None:
            continue
        if phase == "start":
            starts[tid] = (core_id, nc_ts)
        elif phase == "stop":
            info = starts.pop(tid, None)
            if info is not None:
                cid, start_ns = info
                durations_ms.append((nc_ts - start_ns) / 1_000_000.0)
                timestamps.append((cid, start_ns, nc_ts))
    return durations_ms, timestamps


# ─── Worker ─────────────────────────────────────────────────────────────

def run_worker(model, inputs, outputs,
               barrier_warmup, barrier_bench_start, barrier_bench_done,
               core_idx, warmup_iter, bench_iter, round_times):
    for _ in range(warmup_iter):
        model(inputs=inputs, outputs=outputs)
    barrier_warmup.wait()

    barrier_bench_start.wait()

    per_round = []
    for _ in range(bench_iter):
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_round.append(elapsed_ms)

    round_times[core_idx] = per_round
    barrier_bench_done.wait()


# ─── Common benchmark runner ────────────────────────────────────────────

@dataclass
class BenchResult:
    label: str
    num_cores: int
    per_core_round_ms: dict = field(default_factory=dict)
    device_durations_ms: list = field(default_factory=list)
    device_timestamps: list = field(default_factory=list)


def _run_benchmark(label, num_cores, models, per_core_io) -> BenchResult:
    barrier_warmup = threading.Barrier(num_cores + 1)
    barrier_bench_start = threading.Barrier(num_cores + 1)
    barrier_bench_done = threading.Barrier(num_cores + 1)
    round_times = {}

    threads = [
        threading.Thread(
            target=run_worker,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier_warmup, barrier_bench_start, barrier_bench_done,
                  i, WARMUP_ITER, BENCH_ITER, round_times),
        )
        for i in range(num_cores)
    ]

    with SystemTraceSession() as trace:
        for t in threads:
            t.start()

        barrier_warmup.wait()
        trace.drain_events()

        barrier_bench_start.wait()
        barrier_bench_done.wait()

        events_json = trace.fetch_events_json()

    for t in threads:
        t.join()

    durations_ms, timestamps = parse_trace(events_json)

    return BenchResult(
        label=label,
        num_cores=num_cores,
        per_core_round_ms=round_times,
        device_durations_ms=durations_ms,
        device_timestamps=timestamps,
    )


# ─── Benchmark Runners ──────────────────────────────────────────────────

def run_data_parallel(num_cores: int) -> BenchResult:
    """Each core does full M×K @ K×N GEMM independently."""
    print(f"\n{'─' * 60}")
    print(f"  Data Parallel: {num_cores} cores, each does {M}x{K} @ {K}x{N}")
    print(f"{'─' * 60}")

    np.random.seed(42)
    A_proto = ((np.random.rand(M, K) - 0.5) * 2).astype(DTYPE)
    B_proto = ((np.random.rand(K, N) - 0.5) * 2).astype(DTYPE)

    # Use DeviceKernel for DP (no collectives, normal load)
    kernel = DeviceKernel.compile_and_load(
        kernel_data_parallel, A_proto, B_proto,
        name="dp_gemm", use_cached_if_exists=True,
    )

    models = [kernel]
    for cid in range(1, num_cores):
        models.append(SpikeModel.load_from_neff(kernel.neff_path, core_id=cid))

    per_core_io = []
    for cid in range(num_cores):
        A = ((np.random.rand(M, K) - 0.5) * 2).astype(DTYPE)
        B = ((np.random.rand(K, N) - 0.5) * 2).astype(DTYPE)
        out = np.zeros((M, N), dtype=DTYPE)
        inputs = {"A": SpikeTensor.from_numpy(A, "A", core_id=cid),
                  "B": SpikeTensor.from_numpy(B, "B", core_id=cid)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=cid)}
        per_core_io.append((inputs, outputs))

    return _run_benchmark("Data Parallel", num_cores, models, per_core_io)


def run_column_parallel() -> BenchResult:
    """Column-parallel TP: A @ B_shard → all_gather."""
    n_shard = N // TP_DEGREE
    print(f"\n{'─' * 60}")
    print(f"  Column-Parallel TP: {TP_DEGREE} cores")
    print(f"  Each rank: A ({M},{K}) @ B_shard ({K},{n_shard}) → all_gather → C ({M},{N})")
    print(f"{'─' * 60}")

    np.random.seed(42)
    A_proto = ((np.random.rand(M, K) - 0.5) * 2).astype(DTYPE)
    B_shard_proto = ((np.random.rand(K, n_shard) - 0.5) * 2).astype(DTYPE)

    # Compile only (don't load — NEFF has collectives)
    neff_path = compile_kernel(kernel_column_parallel, A_proto, B_shard_proto,
                               name="col_par_gemm")

    # Load on each core with cc_enabled
    models = []
    for cid in range(TP_DEGREE):
        models.append(SpikeModel.load_from_neff(
            neff_path, core_id=cid,
            cc_enabled=True, rank_id=cid, world_size=TP_DEGREE,
        ))

    B_full = ((np.random.rand(K, N) - 0.5) * 2).astype(DTYPE)
    per_core_io = []
    for cid in range(TP_DEGREE):
        A = ((np.random.rand(M, K) - 0.5) * 2).astype(DTYPE)
        B_shard = B_full[:, cid * n_shard : (cid + 1) * n_shard].copy()
        out = np.zeros((M, N), dtype=DTYPE)
        inputs = {"A": SpikeTensor.from_numpy(A, "A", core_id=cid),
                  "B_shard": SpikeTensor.from_numpy(B_shard, "B_shard", core_id=cid)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=cid)}
        per_core_io.append((inputs, outputs))

    return _run_benchmark("Column-Parallel TP", TP_DEGREE, models, per_core_io)


def run_row_parallel() -> BenchResult:
    """Row-parallel TP: A_shard @ B_shard → all_reduce."""
    k_shard = K // TP_DEGREE
    print(f"\n{'─' * 60}")
    print(f"  Row-Parallel TP: {TP_DEGREE} cores")
    print(f"  Each rank: A_shard ({M},{k_shard}) @ B_shard ({k_shard},{N}) → all_reduce → C ({M},{N})")
    print(f"{'─' * 60}")

    np.random.seed(42)
    A_shard_proto = ((np.random.rand(M, k_shard) - 0.5) * 2).astype(DTYPE)
    B_shard_proto = ((np.random.rand(k_shard, N) - 0.5) * 2).astype(DTYPE)

    # Compile only
    neff_path = compile_kernel(kernel_row_parallel, A_shard_proto, B_shard_proto,
                               name="row_par_gemm")

    # Load with cc_enabled
    models = []
    for cid in range(TP_DEGREE):
        models.append(SpikeModel.load_from_neff(
            neff_path, core_id=cid,
            cc_enabled=True, rank_id=cid, world_size=TP_DEGREE,
        ))

    A_full = ((np.random.rand(M, K) - 0.5) * 2).astype(DTYPE)
    B_full = ((np.random.rand(K, N) - 0.5) * 2).astype(DTYPE)
    per_core_io = []
    for cid in range(TP_DEGREE):
        A_shard = A_full[:, cid * k_shard : (cid + 1) * k_shard].copy()
        B_shard = B_full[cid * k_shard : (cid + 1) * k_shard, :].copy()
        out = np.zeros((M, N), dtype=DTYPE)
        inputs = {"A_shard": SpikeTensor.from_numpy(A_shard, "A_shard", core_id=cid),
                  "B_shard": SpikeTensor.from_numpy(B_shard, "B_shard", core_id=cid)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=cid)}
        per_core_io.append((inputs, outputs))

    return _run_benchmark("Row-Parallel TP", TP_DEGREE, models, per_core_io)


# ─── Reporting ──────────────────────────────────────────────────────────

def stats(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    return {"mean": mean, "min": s[0], "max": s[-1], "std": std,
            "p50": s[n // 2], "p95": s[int(n * 0.95)]}


def report(*results: BenchResult):
    flops_per_gemm = 2 * M * K * N

    print("\n")
    print("=" * 80)
    print(f"  TENSOR PARALLEL GEMM COMPARISON")
    print(f"  GEMM: ({M},{K}) x ({K},{N}) fp8, TP degree = {TP_DEGREE}")
    print("=" * 80)

    summary = []
    for r in results:
        all_round_ms = []
        for core_id in sorted(r.per_core_round_ms):
            all_round_ms.extend(r.per_core_round_ms[core_id])

        if not all_round_ms:
            print(f"\n  {r.label}: no data")
            continue

        s = stats(all_round_ms)
        dev_s = stats(r.device_durations_ms) if r.device_durations_ms else {}
        dev_mean = dev_s.get("mean", 0)

        print(f"\n  {r.label} ({r.num_cores} cores):")
        print(f"    Host time / dispatch:     {s['mean']:.3f} ms  (mean)")
        print(f"                              {s['min']:.3f} ms  (min)")
        print(f"                              {s['max']:.3f} ms  (max)")
        print(f"                              {s['std']:.3f} ms  (std)")
        if dev_mean > 0:
            print(f"    Device time / dispatch:   {dev_mean:.3f} ms  (mean)")

        if r.label.startswith("Data"):
            tflops_per_core = flops_per_gemm / (s['mean'] * 1e-3) / 1e12
            agg_tflops = tflops_per_core * r.num_cores
            print(f"    Per-core throughput:      {tflops_per_core:.2f} TFLOPS")
            print(f"    Aggregate throughput:     {agg_tflops:.2f} TFLOPS ({r.num_cores} cores)")
            summary.append({"label": r.label, "latency_ms": s['mean'],
                           "agg_tflops": agg_tflops, "cores": r.num_cores})
        else:
            tp_latency_ms = s['mean']
            tp_tflops = flops_per_gemm / (tp_latency_ms * 1e-3) / 1e12
            print(f"    GEMM latency:            {tp_latency_ms:.3f} ms")
            print(f"    Throughput:              {tp_tflops:.2f} TFLOPS ({r.num_cores} cores)")
            summary.append({"label": r.label, "latency_ms": tp_latency_ms,
                           "agg_tflops": tp_tflops, "cores": r.num_cores})

    if len(summary) >= 2:
        print(f"\n  {'─' * 60}")
        print(f"  Summary:")
        print(f"  {'─' * 60}")
        print(f"  {'Method':<25} {'Cores':>5} {'Latency (ms)':>14} {'TFLOPS':>10}")
        print(f"  {'─' * 60}")
        for s in summary:
            print(f"  {s['label']:<25} {s['cores']:>5} {s['latency_ms']:>14.3f} {s['agg_tflops']:>10.2f}")
        print(f"  {'─' * 60}")

        dp = summary[0]
        dp_single_core_latency = dp['latency_ms']
        for s in summary[1:]:
            speedup_vs_single = dp_single_core_latency / s['latency_ms']
            print(f"\n  {s['label']}:")
            print(f"    Latency speedup vs 1-core DP: {speedup_vs_single:.2f}x")
            print(f"    Ideal speedup ({s['cores']}-way TP):     {s['cores']:.2f}x")
            efficiency = speedup_vs_single / s['cores'] * 100
            print(f"    TP efficiency:                {efficiency:.1f}%")

    print(f"\n{'=' * 80}\n")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print(f"  Tensor Parallel GEMM Benchmark")
    print(f"  GEMM: ({M},{K}) x ({K},{N}) fp8")
    print(f"  TP degree: {TP_DEGREE}")
    print(f"  Sync: ASYNC_INFLIGHT=1")
    print("=" * 80)

    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    spike.configure(visible_cores=range(TP_DEGREE))

    # 1. Data parallel baseline
    dp_result = run_data_parallel(num_cores=TP_DEGREE)

    # 2. Column-parallel TP (compile-only + manual cc_enabled load)
    col_result = run_column_parallel()

    # 3. Row-parallel TP
    row_result = run_row_parallel()

    report(dp_result, col_result, row_result)


if __name__ == "__main__":
    main()
