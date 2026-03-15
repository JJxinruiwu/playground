#!/usr/bin/env python3
"""
End-to-End Time Breakdown — where does the overhead come from?

Measures each phase of a GEMM execution independently:

  Phase 1: Compilation        (compile NKIPy kernel → NEFF)
  Phase 2: NEFF Load          (load NEFF onto NeuronCore(s))
  Phase 3: H2D Transfer       (numpy → SpikeTensor on device)
  Phase 4: Dispatch + Execute (model() call, device trace splits dispatch vs compute)
  Phase 5: D2H Transfer       (SpikeTensor.numpy() back to host)

Runs two scenarios:
  (A) Single-core:  1 core, 1 GEMM/dispatch
  (B) Multi-core:   64 cores, 1 GEMM/dispatch
  (C) Multi-core batched: 64 cores, 20 GEMMs/dispatch (kernel loop)

For each scenario, prints a stacked bar-style breakdown showing where time goes.
"""

import json
import os
import threading
import time

import ml_dtypes
import numpy as np

import spike
from nkipy.runtime import DeviceKernel
from spike import SpikeModel, SystemTraceSession
from spike.spike_tensor import SpikeTensor


# ─── Configuration ──────────────────────────────────────────────────────
SIZE = 4096
KERNEL_LOOP = 20
WARMUP_ITER = 5
BENCH_ITER = 20


# ─── Kernels ────────────────────────────────────────────────────────────

def kernel_single(A, B):
    return A @ B


def kernel_batched(A, B):
    return A @ B


# ─── Trace Parsing ──────────────────────────────────────────────────────

def parse_trace_durations(events_json: str) -> list[float]:
    """Parse nc_exec_running durations (ms) from NRT trace JSON."""
    if not events_json:
        return []
    root = json.loads(events_json)
    events = root.get("events", [])
    starts = {}
    durations_ms = []
    for ev in events:
        if ev.get("event_type") != "nc_exec_running":
            continue
        phase = ev.get("phase")
        tid = ev.get("tracking_id")
        nc_ts = ev.get("data", {}).get("nc_timestamp_ns")
        if nc_ts is None or tid is None:
            continue
        if phase == "start":
            starts[tid] = nc_ts
        elif phase == "stop":
            start_ts = starts.pop(tid, None)
            if start_ts is not None:
                durations_ms.append((nc_ts - start_ts) / 1_000_000.0)
    return durations_ms


# ─── Phase Measurement Helpers ──────────────────────────────────────────

def measure_compile(kernel_fn, input_shape, name):
    """Phase 1: Compile kernel → NEFF. Returns (kernel, elapsed_ms)."""
    np.random.seed(42)
    A_proto = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    # Force recompile by using a unique name
    t0 = time.perf_counter()
    kernel = DeviceKernel.compile_and_load(
        kernel_fn, A_proto, B_proto,
        name=name, use_cached_if_exists=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return kernel, elapsed_ms


def measure_load(kernel, num_cores):
    """Phase 2: Load NEFF on cores 1..num_cores-1. Returns (models, elapsed_ms)."""
    models = [kernel]
    if num_cores == 1:
        return models, 0.0

    t0 = time.perf_counter()
    for cid in range(1, num_cores):
        models.append(SpikeModel.load_from_neff(kernel.neff_path, core_id=cid))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return models, elapsed_ms


def measure_h2d(input_shape, num_cores):
    """Phase 3: Allocate SpikeTensors (H2D). Returns (per_core_io, elapsed_ms)."""
    per_core_io = []
    t0 = time.perf_counter()
    for cid in range(num_cores):
        A = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros(input_shape, dtype=ml_dtypes.float8_e5m2)
        inputs = {
            "A": SpikeTensor.from_numpy(A, "A", core_id=cid),
            "B": SpikeTensor.from_numpy(B, "B", core_id=cid),
        }
        outputs = {
            "output0": SpikeTensor.from_numpy(out, "output0", core_id=cid),
        }
        per_core_io.append((inputs, outputs))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return per_core_io, elapsed_ms


def measure_d2h(per_core_io, num_cores):
    """Phase 5: Copy output tensors back to host. Returns elapsed_ms."""
    t0 = time.perf_counter()
    for cid in range(num_cores):
        _ = per_core_io[cid][1]["output0"].numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms


# ─── Single-Core Execution + Trace ──────────────────────────────────────

def measure_exec_single_core(model, per_core_io, warmup_iter, bench_iter):
    """
    Phase 4 (single core): dispatch + execute with ASYNC_INFLIGHT=1.
    Returns (host_times_ms, device_times_ms).
    """
    inputs, outputs = per_core_io[0]

    # Warmup (no tracing)
    for _ in range(warmup_iter):
        model(inputs=inputs, outputs=outputs)

    # Benchmark with tracing
    host_times_ms = []
    with SystemTraceSession() as trace:
        trace.drain_events()
        for _ in range(bench_iter):
            t0 = time.perf_counter()
            model(inputs=inputs, outputs=outputs)
            host_times_ms.append((time.perf_counter() - t0) * 1000)
        events_json = trace.fetch_events_json()

    device_times_ms = parse_trace_durations(events_json)
    return host_times_ms, device_times_ms


# ─── Multi-Core Execution + Trace ──────────────────────────────────────

def _worker(model, inputs, outputs, barrier_warmup, barrier_bench,
            warmup_iter, bench_iter, host_times, idx):
    for _ in range(warmup_iter):
        model(inputs=inputs, outputs=outputs)
    barrier_warmup.wait()

    per_round = []
    for _ in range(bench_iter):
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        per_round.append((time.perf_counter() - t0) * 1000)
    host_times[idx] = per_round
    barrier_bench.wait()


def measure_exec_multi_core(models, per_core_io, num_cores,
                            warmup_iter, bench_iter):
    """
    Phase 4 (multi-core): dispatch + execute with ASYNC_INFLIGHT=1.
    Returns (host_times_per_core, device_times_ms).
    """
    barrier_warmup = threading.Barrier(num_cores + 1)
    barrier_bench = threading.Barrier(num_cores + 1)
    host_times = {}

    threads = [
        threading.Thread(
            target=_worker,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier_warmup, barrier_bench,
                  warmup_iter, bench_iter, host_times, i),
        )
        for i in range(num_cores)
    ]

    with SystemTraceSession() as trace:
        for t in threads:
            t.start()

        barrier_warmup.wait()
        trace.drain_events()

        barrier_bench.wait()
        events_json = trace.fetch_events_json()

    for t in threads:
        t.join()

    device_times_ms = parse_trace_durations(events_json)
    return host_times, device_times_ms


# ─── Scenario Runner ───────────────────────────────────────────────────

def run_scenario(label, kernel_fn, input_shape, kernel_loop, num_cores, kernel_name):
    """Run a full breakdown for one scenario. Returns dict of phase timings."""
    print(f"\n{'━' * 70}")
    print(f"  Scenario: {label}")
    print(f"  {num_cores} core(s), {kernel_loop} GEMM(s)/dispatch, {SIZE}x{SIZE} FP8")
    print(f"{'━' * 70}")

    # Phase 1: Compile
    print("\n  [Phase 1] Compilation...")
    kernel, compile_ms = measure_compile(kernel_fn, input_shape, kernel_name)
    print(f"    → {compile_ms:.1f} ms")

    # Phase 2: NEFF Load
    print(f"  [Phase 2] NEFF Load ({num_cores} core(s))...")
    models, load_ms = measure_load(kernel, num_cores)
    print(f"    → {load_ms:.1f} ms")

    # Phase 3: H2D Transfer
    print(f"  [Phase 3] H2D Transfer ({num_cores} core(s))...")
    per_core_io, h2d_ms = measure_h2d(input_shape, num_cores)
    h2d_per_core = h2d_ms / num_cores
    bytes_per_core = 2 * np.prod(input_shape) * 1  # 2 inputs, fp8 = 1 byte
    h2d_bw = (bytes_per_core * num_cores) / (h2d_ms * 1e-3) / 1e9
    print(f"    → {h2d_ms:.1f} ms total ({h2d_per_core:.2f} ms/core)")
    print(f"      {bytes_per_core / 1e6:.1f} MB/core, effective BW: {h2d_bw:.1f} GB/s")

    # Phase 4: Dispatch + Execute
    print(f"  [Phase 4] Dispatch + Execute ({BENCH_ITER} iterations)...")
    if num_cores == 1:
        host_times_ms, device_times_ms = measure_exec_single_core(
            models[0], per_core_io, WARMUP_ITER, BENCH_ITER)
        # Flatten for consistent handling
        all_host_ms = host_times_ms
    else:
        host_times_per_core, device_times_ms = measure_exec_multi_core(
            models, per_core_io, num_cores, WARMUP_ITER, BENCH_ITER)
        all_host_ms = []
        for cid in sorted(host_times_per_core):
            all_host_ms.extend(host_times_per_core[cid])

    host_mean = sum(all_host_ms) / len(all_host_ms) if all_host_ms else 0
    dev_mean = sum(device_times_ms) / len(device_times_ms) if device_times_ms else 0
    dispatch_overhead = host_mean - dev_mean

    host_per_gemm = host_mean / kernel_loop
    dev_per_gemm = dev_mean / kernel_loop
    dispatch_per_gemm = dispatch_overhead / kernel_loop

    print(f"    Host time / dispatch:     {host_mean:.3f} ms")
    print(f"    Device time / dispatch:   {dev_mean:.3f} ms")
    print(f"    Dispatch overhead:        {dispatch_overhead:.3f} ms "
          f"({dispatch_overhead / host_mean * 100:.1f}% of host time)"
          if host_mean > 0 else "")
    if kernel_loop > 1:
        print(f"    ── Per GEMM ──")
        print(f"    Host time / GEMM:         {host_per_gemm:.4f} ms")
        print(f"    Device time / GEMM:       {dev_per_gemm:.4f} ms")
        print(f"    Dispatch overhead / GEMM: {dispatch_per_gemm:.4f} ms")

    # Phase 5: D2H Transfer
    print(f"  [Phase 5] D2H Transfer ({num_cores} core(s))...")
    d2h_ms = measure_d2h(per_core_io, num_cores)
    d2h_per_core = d2h_ms / num_cores
    out_bytes_per_core = np.prod(input_shape) * 1  # 1 output, fp8
    d2h_bw = (out_bytes_per_core * num_cores) / (d2h_ms * 1e-3) / 1e9
    print(f"    → {d2h_ms:.1f} ms total ({d2h_per_core:.2f} ms/core)")
    print(f"      {out_bytes_per_core / 1e6:.1f} MB/core, effective BW: {d2h_bw:.1f} GB/s")

    return {
        "label": label,
        "num_cores": num_cores,
        "kernel_loop": kernel_loop,
        "compile_ms": compile_ms,
        "load_ms": load_ms,
        "h2d_ms": h2d_ms,
        "host_exec_ms": host_mean,
        "device_exec_ms": dev_mean,
        "dispatch_overhead_ms": dispatch_overhead,
        "d2h_ms": d2h_ms,
        "host_per_gemm_ms": host_per_gemm,
        "device_per_gemm_ms": dev_per_gemm,
        "dispatch_per_gemm_ms": dispatch_per_gemm,
    }


# ─── Summary Report ────────────────────────────────────────────────────

def print_breakdown_table(results):
    """Print a side-by-side breakdown table."""
    print(f"\n{'=' * 80}")
    print("  END-TO-END BREAKDOWN SUMMARY")
    print(f"{'=' * 80}")

    # Header
    labels = [r["label"] for r in results]
    col_w = 18
    header = f"  {'Phase':<28}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{header}")
    print("  " + "─" * (28 + col_w * len(results)))

    # One-time costs
    print("  [One-time costs]")
    for key, name in [("compile_ms", "Compilation"),
                       ("load_ms", "NEFF Load")]:
        row = f"  {name:<28}"
        for r in results:
            v = r[key]
            row += f"{v:>{col_w - 3}.1f} ms"
        print(row)

    # Per-dispatch costs
    print("  [Per-dispatch costs]")
    for key, name in [("h2d_ms", "H2D Transfer (total)"),
                       ("host_exec_ms", "Host Dispatch+Exec"),
                       ("device_exec_ms", "  └ Device Execution"),
                       ("dispatch_overhead_ms", "  └ Dispatch Overhead"),
                       ("d2h_ms", "D2H Transfer (total)")]:
        row = f"  {name:<28}"
        for r in results:
            v = r[key]
            row += f"{v:>{col_w - 3}.3f} ms"
        print(row)

    # Per-GEMM costs
    print("  [Per-GEMM costs]")
    for key, name in [("host_per_gemm_ms", "Host time / GEMM"),
                       ("device_per_gemm_ms", "Device time / GEMM"),
                       ("dispatch_per_gemm_ms", "Dispatch OH / GEMM")]:
        row = f"  {name:<28}"
        for r in results:
            v = r[key]
            row += f"{v:>{col_w - 3}.4f} ms"
        print(row)

    # Derived metrics
    print("  [Derived metrics]")
    flops_per_gemm = 2 * SIZE * SIZE * SIZE
    for r in results:
        dev_tflops = flops_per_gemm / (r["device_per_gemm_ms"] * 1e-3) / 1e12 \
            if r["device_per_gemm_ms"] > 0 else 0
        host_tflops = flops_per_gemm / (r["host_per_gemm_ms"] * 1e-3) / 1e12 \
            if r["host_per_gemm_ms"] > 0 else 0
        r["dev_tflops_per_core"] = dev_tflops
        r["host_tflops_per_core"] = host_tflops
        r["overhead_pct"] = (r["dispatch_overhead_ms"] / r["host_exec_ms"] * 100
                             if r["host_exec_ms"] > 0 else 0)

    row = f"  {'Device TFLOPS/core':<28}"
    for r in results:
        row += f"{r['dev_tflops_per_core']:>{col_w - 3}.2f}   "
    print(row)

    row = f"  {'Host TFLOPS/core':<28}"
    for r in results:
        row += f"{r['host_tflops_per_core']:>{col_w - 3}.2f}   "
    print(row)

    row = f"  {'Agg Device TFLOPS':<28}"
    for r in results:
        v = r['dev_tflops_per_core'] * r['num_cores']
        row += f"{v:>{col_w - 3}.1f}   "
    print(row)

    row = f"  {'Agg Host TFLOPS':<28}"
    for r in results:
        v = r['host_tflops_per_core'] * r['num_cores']
        row += f"{v:>{col_w - 3}.1f}   "
    print(row)

    row = f"  {'Dispatch OH %':<28}"
    for r in results:
        row += f"{r['overhead_pct']:>{col_w - 3}.1f} %  "
    print(row)

    # Bottleneck identification
    print(f"\n  {'─' * 60}")
    print("  BOTTLENECK ANALYSIS")
    print(f"  {'─' * 60}")
    for r in results:
        total_per_dispatch = r["h2d_ms"] + r["host_exec_ms"] + r["d2h_ms"]
        phases = [
            ("H2D", r["h2d_ms"]),
            ("Dispatch OH", r["dispatch_overhead_ms"]),
            ("Device Exec", r["device_exec_ms"]),
            ("D2H", r["d2h_ms"]),
        ]
        phases_sorted = sorted(phases, key=lambda x: x[1], reverse=True)
        top_phase, top_ms = phases_sorted[0]

        print(f"\n  {r['label']}:")
        print(f"    Total per-dispatch wall time: {total_per_dispatch:.3f} ms")
        print(f"    Breakdown:")
        for pname, pms in phases:
            pct = pms / total_per_dispatch * 100 if total_per_dispatch > 0 else 0
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            print(f"      {pname:<16} {pms:>8.3f} ms  ({pct:>5.1f}%)  {bar}")
        print(f"    → Bottleneck: {top_phase} ({top_ms:.3f} ms)")

    print(f"\n{'=' * 80}\n")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  End-to-End Time Breakdown")
    print(f"  {SIZE}x{SIZE} FP8 GEMM — identifying where overhead lives")
    print("=" * 80)

    # Use ASYNC_INFLIGHT=1 so model() blocks → host time = dispatch + exec
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"

    num_cores = 64
    spike.configure(visible_cores=range(num_cores))

    results = []

    # Scenario A: Single-core, 1 GEMM/dispatch
    r = run_scenario(
        label="1core×1GEMM",
        kernel_fn=kernel_single,
        input_shape=(SIZE, SIZE),
        kernel_loop=1,
        num_cores=1,
        kernel_name="breakdown_single",
    )
    results.append(r)

    # Scenario B: 64-core, 1 GEMM/dispatch
    r = run_scenario(
        label="64core×1GEMM",
        kernel_fn=kernel_single,
        input_shape=(SIZE, SIZE),
        kernel_loop=1,
        num_cores=num_cores,
        kernel_name="breakdown_single",
    )
    results.append(r)

    # Scenario C: 64-core, 20 GEMMs/dispatch (kernel loop)
    r = run_scenario(
        label="64core×20GEMM",
        kernel_fn=kernel_batched,
        input_shape=(KERNEL_LOOP, SIZE, SIZE),
        kernel_loop=KERNEL_LOOP,
        num_cores=num_cores,
        kernel_name="breakdown_batched20",
    )
    results.append(r)

    print_breakdown_table(results)


if __name__ == "__main__":
    main()
