#!/usr/bin/env python3
"""
Compare dispatch overhead: v1 (1 GEMM/dispatch) vs v4 (20 GEMMs/dispatch)

Method 1: NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
  → model() blocks until device completes, no D2H needed for sync.
  → Host wall time = dispatch overhead + device execution (accurate).

Method 2: Device trace inter-execution gaps
  → Gaps between consecutive nc_exec_running on same core = device idle time.
  → v1: large gaps (host round-trip per GEMM), v4: gaps only between batches.

Usage:
    python compare_dispatch_overhead.py
"""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import ml_dtypes
import numpy as np

import spike
from nkipy.runtime import DeviceKernel
from spike import SpikeModel, SystemTraceSession
from spike.spike_tensor import SpikeTensor


# ─── Configuration ──────────────────────────────────────────────────────
SIZE = 4096
NUM_CORES = 64
WARMUP_ITER = 5
BENCH_ITER = 10
KERNEL_LOOP = 20  # v4 batch size


@dataclass
class BenchResult:
    label: str
    kernel_loop: int
    dispatches_per_core: int
    total_gemms: int
    # Per-core per-round host wall times (synchronized via ASYNC_INFLIGHT=1)
    per_core_round_ms: dict = field(default_factory=dict)  # core_id → [ms, ...]
    # Device trace data
    device_durations_ms: list = field(default_factory=list)
    device_timestamps: list = field(default_factory=list)  # (core_id, start_ns, stop_ns)


# ─── Trace Parsing ──────────────────────────────────────────────────────

def parse_trace(events_json: str):
    """Parse nc_exec_running events → durations and per-core timestamps."""
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


def compute_inter_exec_gaps(timestamps):
    """
    Per-core gaps between consecutive executions (device idle time).
    Only returns non-negative gaps (filter clock artifacts).
    """
    by_core = defaultdict(list)
    for cid, start_ns, stop_ns in timestamps:
        by_core[cid].append((start_ns, stop_ns))

    gaps_ms = []
    for cid in sorted(by_core):
        execs = sorted(by_core[cid])
        for i in range(1, len(execs)):
            gap_ns = execs[i][0] - execs[i - 1][1]
            if gap_ns >= 0:
                gaps_ms.append(gap_ns / 1_000_000.0)
    return gaps_ms


def compute_device_utilization(timestamps, durations_ms):
    """
    Per-core utilization = busy_time / (last_stop - first_start).
    Returns (mean_utilization%, per_core_utilization dict).
    """
    by_core = defaultdict(list)
    for cid, start_ns, stop_ns in timestamps:
        by_core[cid].append((start_ns, stop_ns))

    per_core_util = {}
    for cid in sorted(by_core):
        execs = sorted(by_core[cid])
        if not execs:
            continue
        first_start = execs[0][0]
        last_stop = execs[-1][1]
        total_span = last_stop - first_start
        busy_time = sum(stop - start for start, stop in execs)
        if total_span > 0:
            per_core_util[cid] = busy_time / total_span * 100
        else:
            per_core_util[cid] = 0.0

    mean_util = sum(per_core_util.values()) / len(per_core_util) if per_core_util else 0
    return mean_util, per_core_util


# ─── Kernel Definitions ────────────────────────────────────────────────

def kernel_single(A, B):
    return A @ B


def kernel_batched(A, B):
    return A @ B


# ─── Worker (Sync via ASYNC_INFLIGHT=1) ────────────────────────────────

def run_worker(model, inputs, outputs,
               barrier_warmup, barrier_bench_start, barrier_bench_done,
               core_idx, warmup_iter, bench_iter, round_times):
    """
    Worker thread. With ASYNC_INFLIGHT=1, model() blocks until device done.
    No .numpy() needed for synchronization.
    """
    for _ in range(warmup_iter):
        model(inputs=inputs, outputs=outputs)
    barrier_warmup.wait()

    barrier_bench_start.wait()

    per_round = []
    for _ in range(bench_iter):
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        # model() returns only after device execution completes (ASYNC_INFLIGHT=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_round.append(elapsed_ms)

    round_times[core_idx] = per_round
    barrier_bench_done.wait()


# ─── Benchmark Runner ───────────────────────────────────────────────────

def run_benchmark(label, kernel_fn, input_shape, kernel_loop,
                  dispatches_per_core) -> BenchResult:
    print(f"\n{'─' * 60}")
    print(f"  Running: {label}")
    print(f"  Input shape: {input_shape}, dispatches/core: {dispatches_per_core}")
    print(f"  Sync method: ASYNC_INFLIGHT=1 (no D2H overhead)")
    print(f"{'─' * 60}")

    # Compile
    np.random.seed(42)
    A_proto = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    safe_name = ''.join(c if c.isalnum() or c == '_' else '' for c in label)
    kernel = DeviceKernel.compile_and_load(
        kernel_fn, A_proto, B_proto,
        name=f"cmp_{safe_name}",
        use_cached_if_exists=True,
    )

    # Load on all cores
    models = [kernel]
    for cid in range(1, NUM_CORES):
        models.append(SpikeModel.load_from_neff(kernel.neff_path, core_id=cid))

    # Allocate tensors
    per_core_io = []
    for cid in range(NUM_CORES):
        A = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(*input_shape) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros(input_shape, dtype=ml_dtypes.float8_e5m2)
        inputs = {"A": SpikeTensor.from_numpy(A, "A", core_id=cid),
                  "B": SpikeTensor.from_numpy(B, "B", core_id=cid)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=cid)}
        per_core_io.append((inputs, outputs))

    # Run with tracing
    barrier_warmup = threading.Barrier(NUM_CORES + 1)
    barrier_bench_start = threading.Barrier(NUM_CORES + 1)
    barrier_bench_done = threading.Barrier(NUM_CORES + 1)
    round_times = {}

    threads = [
        threading.Thread(
            target=run_worker,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier_warmup, barrier_bench_start, barrier_bench_done,
                  i, WARMUP_ITER, dispatches_per_core, round_times),
        )
        for i in range(NUM_CORES)
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
    total_gemms = NUM_CORES * dispatches_per_core * kernel_loop

    return BenchResult(
        label=label,
        kernel_loop=kernel_loop,
        dispatches_per_core=dispatches_per_core,
        total_gemms=total_gemms,
        per_core_round_ms=round_times,
        device_durations_ms=durations_ms,
        device_timestamps=timestamps,
    )


# ─── Analysis & Reporting ──────────────────────────────────────────────

def stats(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    return {
        "mean": mean, "min": s[0], "max": s[-1], "std": std,
        "p50": s[n // 2], "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)],
    }


def report(v1: BenchResult, v4: BenchResult):
    flops_per_gemm = 2 * SIZE * SIZE * SIZE

    print("\n")
    print("=" * 80)
    print("  DISPATCH OVERHEAD COMPARISON: v1 (1/dispatch) vs v4 (20/dispatch)")
    print("  Sync method: ASYNC_INFLIGHT=1 (model() blocks, no D2H)")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 1: Host Wall Time (accurate via ASYNC_INFLIGHT=1)
    # ═══════════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  METHOD 1: Host Wall Time (ASYNC_INFLIGHT=1, no D2H)           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── 1a: Per-dispatch host time ──
    print("\n  [1a] Per-Dispatch Host Wall Time")
    print("  " + "─" * 50)

    host_per_gemm = {}
    for r in [v1, v4]:
        all_round_ms = []
        for core_id in sorted(r.per_core_round_ms):
            all_round_ms.extend(r.per_core_round_ms[core_id])

        if not all_round_ms:
            print(f"\n  {r.label}: no host timing data")
            continue

        s = stats(all_round_ms)
        per_gemm_mean = s["mean"] / r.kernel_loop
        host_per_gemm[r.label] = per_gemm_mean

        print(f"\n  {r.label}:")
        print(f"    GEMMs per dispatch:       {r.kernel_loop}")
        print(f"    Host time / dispatch:")
        print(f"      mean:  {s['mean']:.3f} ms")
        print(f"      min:   {s['min']:.3f} ms")
        print(f"      max:   {s['max']:.3f} ms")
        print(f"      std:   {s['std']:.3f} ms")
        print(f"      p50:   {s['p50']:.3f} ms")
        print(f"      p95:   {s['p95']:.3f} ms")
        print(f"    Host time / GEMM:         {per_gemm_mean:.4f} ms")

    # ── 1b: Device execution time (should be equal) ──
    print(f"\n  [1b] Device Execution Time (pure compute, should be ~equal)")
    print("  " + "─" * 50)

    for r in [v1, v4]:
        if not r.device_durations_ms:
            print(f"\n  {r.label}: no device trace data")
            continue
        s = stats(r.device_durations_ms)
        per_gemm = s["mean"] / r.kernel_loop
        print(f"\n  {r.label}:")
        print(f"    Captured executions:      {len(r.device_durations_ms)}")
        print(f"    Device time / dispatch:   {s['mean']:.3f} ms  (mean)")
        print(f"    Device time / GEMM:       {per_gemm:.4f} ms")

    # ── 1c: Overhead decomposition ──
    print(f"\n  [1c] Dispatch Overhead = Host Time - Device Time")
    print("  " + "─" * 50)

    overhead_data = {}
    for r in [v1, v4]:
        all_round_ms = []
        for core_id in sorted(r.per_core_round_ms):
            all_round_ms.extend(r.per_core_round_ms[core_id])
        if not all_round_ms or not r.device_durations_ms:
            continue

        host_mean = sum(all_round_ms) / len(all_round_ms)
        dev_mean = sum(r.device_durations_ms) / len(r.device_durations_ms)
        overhead_per_dispatch = host_mean - dev_mean
        overhead_per_gemm = overhead_per_dispatch / r.kernel_loop
        overhead_pct = (overhead_per_dispatch / host_mean) * 100 if host_mean > 0 else 0

        overhead_data[r.label] = {
            "host_mean": host_mean, "dev_mean": dev_mean,
            "overhead_per_dispatch": overhead_per_dispatch,
            "overhead_per_gemm": overhead_per_gemm,
            "overhead_pct": overhead_pct,
        }

        print(f"\n  {r.label}:")
        print(f"    Host time / dispatch:        {host_mean:.3f} ms")
        print(f"    Device time / dispatch:      {dev_mean:.3f} ms")
        print(f"    Overhead / dispatch:          {overhead_per_dispatch:.3f} ms  ({overhead_pct:.1f}% of host time)")
        print(f"    Overhead / GEMM:              {overhead_per_gemm:.4f} ms")

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 2: Device Trace Gap Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  METHOD 2: Device Trace — Inter-Execution Gaps & Utilization    ║")
    print("║  (purely device-side, independent validation of Method 1)       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── 2a: Inter-execution gaps ──
    print("\n  [2a] Inter-Execution Gaps (device idle between dispatches)")
    print("  " + "─" * 50)
    print("  v1: gap after every GEMM (host round-trip)")
    print("  v4: gap only after every 20 GEMMs (batch boundary)")

    for r in [v1, v4]:
        gaps = compute_inter_exec_gaps(r.device_timestamps)
        if not gaps:
            print(f"\n  {r.label}: insufficient data")
            continue
        s = stats(gaps)
        total_gap_ms = sum(gaps)
        num_gaps = len(gaps)
        gaps_per_core = num_gaps / NUM_CORES

        print(f"\n  {r.label}:")
        print(f"    Total gaps:            {num_gaps}  ({gaps_per_core:.0f} per core)")
        print(f"    Mean gap:              {s['mean']:.3f} ms")
        print(f"    Median gap:            {s['p50']:.3f} ms")
        print(f"    Min gap:               {s['min']:.3f} ms")
        print(f"    Max gap:               {s['max']:.3f} ms")
        print(f"    Total gap time:        {total_gap_ms:.1f} ms (sum across all cores)")

    # ── 2b: Device utilization ──
    print(f"\n  [2b] Device Utilization (busy / total span, per core)")
    print("  " + "─" * 50)

    for r in [v1, v4]:
        mean_util, per_core_util = compute_device_utilization(
            r.device_timestamps, r.device_durations_ms)
        if not per_core_util:
            print(f"\n  {r.label}: insufficient data")
            continue
        utils = list(per_core_util.values())
        s = stats(utils)
        print(f"\n  {r.label}:")
        print(f"    Mean utilization:      {mean_util:.1f}%")
        print(f"    Min utilization:       {s['min']:.1f}%")
        print(f"    Max utilization:       {s['max']:.1f}%")
        print(f"    Std:                   {s['std']:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    results_summary = []
    for r in [v1, v4]:
        all_round_ms = []
        for core_id in sorted(r.per_core_round_ms):
            all_round_ms.extend(r.per_core_round_ms[core_id])
        if not all_round_ms or not r.device_durations_ms:
            continue

        host_mean = sum(all_round_ms) / len(all_round_ms)
        dev_mean = sum(r.device_durations_ms) / len(r.device_durations_ms)
        overhead = host_mean - dev_mean

        mean_util, _ = compute_device_utilization(
            r.device_timestamps, r.device_durations_ms)
        gaps = compute_inter_exec_gaps(r.device_timestamps)
        mean_gap = sum(gaps) / len(gaps) if gaps else 0

        results_summary.append({
            "label": r.label, "kl": r.kernel_loop,
            "host_per_gemm": host_mean / r.kernel_loop,
            "dev_per_gemm": dev_mean / r.kernel_loop,
            "overhead_per_gemm": overhead / r.kernel_loop,
            "overhead_per_dispatch": overhead,
            "device_util": mean_util,
            "mean_gap": mean_gap,
        })

    if len(results_summary) == 2:
        r1, r4 = results_summary
        print(f"""
  ┌──────────────────────────┬──────────────────┬──────────────────┐
  │                          │ v1 (1 GEMM/disp) │ v4 (20 GEMM/disp)│
  ├──────────────────────────┼──────────────────┼──────────────────┤
  │ Host time / GEMM         │ {r1['host_per_gemm']:>12.4f} ms   │ {r4['host_per_gemm']:>12.4f} ms   │
  │ Device time / GEMM       │ {r1['dev_per_gemm']:>12.4f} ms   │ {r4['dev_per_gemm']:>12.4f} ms   │
  │ Overhead / GEMM          │ {r1['overhead_per_gemm']:>12.4f} ms   │ {r4['overhead_per_gemm']:>12.4f} ms   │
  │ Overhead / dispatch      │ {r1['overhead_per_dispatch']:>12.4f} ms   │ {r4['overhead_per_dispatch']:>12.4f} ms   │
  │ Device utilization       │ {r1['device_util']:>12.1f} %    │ {r4['device_util']:>12.1f} %    │
  │ Mean inter-exec gap      │ {r1['mean_gap']:>12.3f} ms   │ {r4['mean_gap']:>12.3f} ms   │
  └──────────────────────────┴──────────────────┴──────────────────┘""")

        # Overhead reduction
        if r1['overhead_per_gemm'] > 0 and r4['overhead_per_gemm'] > 0:
            ratio = r1['overhead_per_gemm'] / r4['overhead_per_gemm']
            print(f"\n  Dispatch overhead per GEMM reduced by {ratio:.1f}x")
            print(f"  (expected ~{KERNEL_LOOP}x from batching {KERNEL_LOOP} GEMMs/dispatch)")
        elif r1['overhead_per_gemm'] > 0:
            print(f"\n  v4 dispatch overhead per GEMM is negligible — fully amortized!")

        # Host speedup
        if r4['host_per_gemm'] > 0:
            speedup = r1['host_per_gemm'] / r4['host_per_gemm']
            print(f"\n  Host-side per-GEMM speedup: {speedup:.2f}x")
            print(f"    v1: {r1['host_per_gemm']:.4f} ms/GEMM → v4: {r4['host_per_gemm']:.4f} ms/GEMM")

        # Utilization improvement
        if r1['device_util'] > 0:
            util_improvement = r4['device_util'] / r1['device_util']
            print(f"\n  Device utilization improvement: {util_improvement:.2f}x")
            print(f"    v1: {r1['device_util']:.1f}% → v4: {r4['device_util']:.1f}%")

        # Throughput
        print(f"\n  Effective Throughput (host-measured, {NUM_CORES} cores):")
        for rs in [r1, r4]:
            gemms_per_core = BENCH_ITER * rs['kl']
            total_wall_est = rs['host_per_gemm'] * gemms_per_core
            flops = gemms_per_core * flops_per_gemm
            tflops = flops / (total_wall_est * 1e-3) / 1e12
            agg_tflops = tflops * NUM_CORES
            print(f"    {rs['label']}: {agg_tflops:.2f} TFLOPS")

    print(f"\n{'=' * 80}\n")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  Dispatch Overhead Comparison: v1 vs v4")
    print(f"  {SIZE}x{SIZE} fp8 GEMM on {NUM_CORES} cores")
    print(f"  Sync: NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1")
    print("=" * 80)

    # Method 1: force sync dispatch — model() blocks until device done
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"
    spike.configure(visible_cores=range(NUM_CORES))

    # v1: 10 dispatches × 1 GEMM = 10 GEMMs/core
    v1_result = run_benchmark(
        label="v1_1GEMMperDispatch",
        kernel_fn=kernel_single,
        input_shape=(SIZE, SIZE),
        kernel_loop=1,
        dispatches_per_core=BENCH_ITER,
    )

    # v4: 10 dispatches × 20 GEMMs = 200 GEMMs/core
    v4_result = run_benchmark(
        label="v4_20GEMMperDispatch",
        kernel_fn=kernel_batched,
        input_shape=(KERNEL_LOOP, SIZE, SIZE),
        kernel_loop=KERNEL_LOOP,
        dispatches_per_core=BENCH_ITER,
    )

    report(v1_result, v4_result)


if __name__ == "__main__":
    main()
