#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication v4 — v1 + kernel-internal loop ×20 + device tracing

Key change: the kernel itself loops 20 matmuls on-device per host dispatch.
One host submission → 20 device-side GEMMs, zero host involvement between them.

Uses a single SystemTraceSession (all cores) for accurate device-side timing
via NeuronCore hardware clocks.
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

KERNEL_LOOP = 20  # on-device iterations per host dispatch


def nkipy_kernel(A, B):
    # A: (KERNEL_LOOP, size, size), B: (KERNEL_LOOP, size, size)
    # Batched matmul: C[i] = A[i] @ B[i] for each i independently.
    return A @ B


def _parse_trace_durations(events_json: str) -> list[float]:
    """Parse nc_exec_running durations (ms) from NRT sys trace JSON."""
    if not events_json:
        return []
    root = json.loads(events_json)
    events = root.get("events", [])
    starts: dict[int, int] = {}
    durations_ms: list[float] = []
    for event in events:
        if event.get("event_type") != "nc_exec_running":
            continue
        phase = event.get("phase")
        tracking_id = event.get("tracking_id")
        nc_ts = event.get("data", {}).get("nc_timestamp_ns")
        if nc_ts is None or tracking_id is None:
            continue
        if phase == "start":
            starts[tracking_id] = nc_ts
        elif phase == "stop":
            start_ts = starts.pop(tracking_id, None)
            if start_ts is not None:
                durations_ms.append((nc_ts - start_ts) / 1_000_000.0)
    return durations_ms


def run_on_core(model, inputs, outputs, barrier_warmup_done, barrier_bench_done,
                core_idx, warmup_iterations, benchmark_iterations):
    """Worker: run warmup, signal, run benchmark, signal."""
    for _ in range(warmup_iterations):
        model(inputs=inputs, outputs=outputs)
    barrier_warmup_done.wait()

    for _ in range(benchmark_iterations):
        model(inputs=inputs, outputs=outputs)
    barrier_bench_done.wait()


def main():
    print("=" * 80)
    print(f"Multi-Core Matrix Multiplication v4 (kernel loop ×{KERNEL_LOOP} + device tracing)")
    print("=" * 80)

    # Configuration
    size = 4096
    num_cores = 64
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "63"
    spike.configure(visible_cores=range(num_cores))
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size:      {KERNEL_LOOP}x{size}x{size} (batched)")
    print(f"  Num cores:        {num_cores}")
    print("  Data type:        float8_e5m2")
    print(f"  Kernel loop:      {KERNEL_LOOP} independent GEMMs per dispatch (batched)")
    print(f"  Warmup iter:      {warmup_iterations}")
    print(f"  Bench iter:       {benchmark_iterations}")
    print(f"  Total GEMMs/core: {benchmark_iterations * KERNEL_LOOP}")
    print("  Timing:           device-side (SystemTraceSession, all cores)")

    # [1] Compile once
    print("\n[1/4] Compiling kernel (once)...")
    np.random.seed(42)
    A_proto = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    t0 = time.time()
    kernel_core0 = DeviceKernel.compile_and_load(
        nkipy_kernel, A_proto, B_proto, name="simple_kernel_loop20",
        use_cached_if_exists=True,
    )
    print(f"  ✓ Compiled in {time.time() - t0:.2f}s  →  {kernel_core0.neff_path}")

    # [2] Load same NEFF on all cores
    print(f"\n[2/4] Loading NEFF on {num_cores} cores...")
    models = [kernel_core0]
    for core_id in range(1, num_cores):
        models.append(SpikeModel.load_from_neff(kernel_core0.neff_path, core_id=core_id))
    print(f"  ✓ Loaded on all {num_cores} cores")

    # [3] Allocate independent input/output tensors per core
    print("\n[3/4] Allocating tensors per core...")
    per_core_io = []
    for core_id in range(num_cores):
        A = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros((KERNEL_LOOP, size, size), dtype=ml_dtypes.float8_e5m2)
        inputs  = {"A": SpikeTensor.from_numpy(A, "A", core_id=core_id),
                   "B": SpikeTensor.from_numpy(B, "B", core_id=core_id)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id)}
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # [4] Benchmark with single SystemTraceSession tracing all cores
    print("\n[4/4] Benchmarking (all cores in parallel, device tracing)...")

    barrier_warmup_done = threading.Barrier(num_cores + 1)
    barrier_bench_done = threading.Barrier(num_cores + 1)

    threads = [
        threading.Thread(
            target=run_on_core,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier_warmup_done, barrier_bench_done,
                  i, warmup_iterations, benchmark_iterations),
        )
        for i in range(num_cores)
    ]

    # Single trace session for all cores
    with SystemTraceSession() as trace:
        for t in threads:
            t.start()

        barrier_warmup_done.wait()
        trace.drain_events()

        barrier_bench_done.wait()
        events_json = trace.fetch_events_json()

    for t in threads:
        t.join()

    # Parse device-side durations (each duration = one dispatch = KERNEL_LOOP GEMMs)
    durations_ms = _parse_trace_durations(events_json)

    # Stats
    flops_per_gemm = 2 * size * size * size
    flops_per_dispatch = flops_per_gemm * KERNEL_LOOP
    bytes_per_dispatch = 3 * KERNEL_LOOP * size * size * 1  # fp8 = 1 byte
    expected_execs = num_cores * benchmark_iterations

    print(f"\n  Device Trace Results:")
    print("  ─────────────────────────────────────")
    print(f"  Captured executions:   {len(durations_ms)} (expected {expected_execs})")

    if durations_ms:
        mean_ms = sum(durations_ms) / len(durations_ms)
        min_ms = min(durations_ms)
        max_ms = max(durations_ms)
        std_ms = (sum((d - mean_ms) ** 2 for d in durations_ms) / len(durations_ms)) ** 0.5
        mean_ms_per_gemm = mean_ms / KERNEL_LOOP

        per_core_tflops = flops_per_dispatch / (mean_ms * 1e-3) / 1e12
        aggregate_tflops = per_core_tflops * num_cores
        aggregate_bw_gbs = bytes_per_dispatch * num_cores / (mean_ms * 1e-3) / 1e9

        print(f"  Kernel loop:           {KERNEL_LOOP} GEMMs/dispatch")
        print(f"  Device time / dispatch:{mean_ms:.3f} ms (mean)")
        print(f"                         {min_ms:.3f} ms (min)")
        print(f"                         {max_ms:.3f} ms (max)")
        print(f"                         {std_ms:.3f} ms (std)")
        print(f"  Device time / GEMM:    {mean_ms_per_gemm:.3f} ms (mean)")
        print("  ─────────────────────────────────────")
        print(f"  Per-core throughput:   {per_core_tflops:.2f} TFLOPS")
        print(f"  Aggregate throughput:  {aggregate_tflops:.2f} TFLOPS")
        print(f"  Aggregate memory BW:   {aggregate_bw_gbs:.2f} GB/s")
        print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
