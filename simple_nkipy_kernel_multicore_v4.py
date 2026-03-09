#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication v4 — v1 + kernel-internal loop ×20

Key change: the kernel itself loops 20 matmuls on-device per host dispatch.
One host submission → 20 device-side GEMMs, zero host involvement between them.

This reduces NRT queue pressure by 20× compared to v1:
  v1: 64 cores × 10 iters = 640 host dispatches fighting the 63-slot queue
  v4: 64 cores × 10 iters = 640 dispatches, but each dispatch does 20 GEMMs
      → effective throughput measured over 20× more compute per round-trip
"""

import os
import threading
import time

import ml_dtypes
import numpy as np

import spike
from nkipy.runtime import DeviceKernel
from spike import SpikeModel
from spike.spike_tensor import SpikeTensor

KERNEL_LOOP = 20  # on-device iterations per host dispatch


def nkipy_kernel(A, B):
    # A: (KERNEL_LOOP, size, size), B: (KERNEL_LOOP, size, size)
    # Batched matmul: C[i] = A[i] @ B[i] for each i independently.
    # Each slice has different data → no saturation, no dead-code elimination.
    return A @ B


def run_on_core(model, inputs, outputs, barrier, results, core_idx, iterations):
    """Worker: wait at barrier, then execute kernel N times, record wall time."""
    barrier.wait()
    start = time.perf_counter()
    for _ in range(iterations):
        model(inputs=inputs, outputs=outputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[core_idx] = elapsed_ms


def main():
    print("=" * 80)
    print(f"Multi-Core Matrix Multiplication v4 (v1 + kernel loop ×{KERNEL_LOOP})")
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
        inputs  = {"A": SpikeTensor.from_numpy(A,   "A",       core_id=core_id),
                   "B": SpikeTensor.from_numpy(B,   "B",       core_id=core_id)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id)}
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # [4] Benchmark
    print("\n[4/4] Benchmarking (all cores in parallel)...")

    def make_threads(iters, result_buf):
        return [
            threading.Thread(
                target=run_on_core,
                args=(models[i], per_core_io[i][0], per_core_io[i][1],
                      barrier, result_buf, i, iters),
            )
            for i in range(num_cores)
        ]

    # Warmup
    barrier = threading.Barrier(num_cores)
    dummy = [None] * num_cores
    for t in (threads := make_threads(warmup_iterations, dummy)): t.start()
    for t in threads: t.join()

    # Benchmark
    barrier = threading.Barrier(num_cores)
    results_ms = [None] * num_cores
    for t in (threads := make_threads(benchmark_iterations, results_ms)): t.start()
    for t in threads: t.join()

    # Stats — each host iter did KERNEL_LOOP GEMMs on device
    total_wall_ms = max(results_ms)
    mean_ms_per_dispatch = total_wall_ms / benchmark_iterations
    mean_ms_per_gemm = mean_ms_per_dispatch / KERNEL_LOOP

    flops_per_gemm = 2 * size * size * size
    flops_per_dispatch = flops_per_gemm * KERNEL_LOOP
    aggregate_tflops = flops_per_dispatch * num_cores / (mean_ms_per_dispatch * 1e-3) / 1e12

    # Memory BW: all KERNEL_LOOP slices of A, B read + output written (fp8 = 1 byte each)
    bytes_per_dispatch = 3 * KERNEL_LOOP * size * size * 1
    aggregate_bw_gbs = bytes_per_dispatch * num_cores / (mean_ms_per_dispatch * 1e-3) / 1e9

    print("\n  Performance Results:")
    print("  ─────────────────────────────────────")
    print(f"  Cores:                 {num_cores}")
    print(f"  Kernel loop:           {KERNEL_LOOP} GEMMs/dispatch")
    print(f"  Wall time / dispatch:  {mean_ms_per_dispatch:.3f} ms")
    print(f"  Wall time / GEMM:      {mean_ms_per_gemm:.3f} ms")
    for i, ms in enumerate(results_ms):
        print(f"  Core {i:2d} total ({benchmark_iterations} dispatches): {ms:.2f} ms")
    print("  ─────────────────────────────────────")
    print(f"  Aggregate throughput:  {aggregate_tflops:.2f} TFLOPS")
    print(f"  Aggregate memory BW:   {aggregate_bw_gbs:.2f} GB/s")
    print(f"  Per-core throughput:   {aggregate_tflops / num_cores:.2f} TFLOPS")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
