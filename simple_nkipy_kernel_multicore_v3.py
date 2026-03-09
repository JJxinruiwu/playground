#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication v3 — v1 + NUMA-aware thread pinning

Each Python thread is pinned to CPUs on the same NUMA node as its NeuronCore.
This reduces host↔device scheduling latency for dispatch and result polling.

trn2.48xlarge NUMA layout (from AWS Neuron docs):
  core_ids  0-15  → Device  0-3  → NUMA node 1 → CPUs 48-95, 144-191
  core_ids 16-47  → Device  4-11 → NUMA node 0 → CPUs  0-47,  96-143
  core_ids 48-63  → Device 12-15 → NUMA node 1 → CPUs 48-95, 144-191
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

# CPU affinity per core_id based on trn2.48xlarge NUMA table
_NUMA0_CPUS = list(range(0, 48)) + list(range(96, 144))   # NUMA node 0
_NUMA1_CPUS = list(range(48, 96)) + list(range(144, 192)) # NUMA node 1

def _cpus_for_core(core_id: int) -> list[int]:
    """Return the preferred CPU affinity list for a given NeuronCore ID."""
    if 16 <= core_id <= 47:
        return _NUMA0_CPUS
    return _NUMA1_CPUS


def nkipy_kernel(A, B):
    return A @ B


def run_on_core(model, inputs, outputs, barrier, results, core_idx, iterations):
    """Worker: pin to NUMA-local CPUs, wait at barrier, execute kernel N times."""
    try:
        os.sched_setaffinity(0, _cpus_for_core(core_idx))
    except OSError:
        pass  # not supported on all platforms, continue without pinning

    barrier.wait()  # synchronize launch across all threads
    start = time.perf_counter()
    for _ in range(iterations):
        model(inputs=inputs, outputs=outputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[core_idx] = elapsed_ms


def main():
    print("=" * 80)
    print("Multi-Core Matrix Multiplication v3 (v1 + NUMA pinning)")
    print("=" * 80)

    # Configuration
    size = 4096
    num_cores = 64
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "63"
    spike.configure(visible_cores=range(num_cores))
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size:  {size}x{size}")
    print(f"  Num cores:    {num_cores}")
    print("  Data type:    float8_e5m2")
    print(f"  Warmup iter:  {warmup_iterations}")
    print(f"  Bench iter:   {benchmark_iterations}")
    print("  NUMA pinning: enabled")

    # [1] Compile once
    print("\n[1/4] Compiling kernel (once)...")
    np.random.seed(42)
    A_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    t0 = time.time()
    kernel_core0 = DeviceKernel.compile_and_load(
        nkipy_kernel, A_proto, B_proto, name="simple_kernel", use_cached_if_exists=True,
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
        A = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros((size, size), dtype=ml_dtypes.float8_e5m2)
        inputs  = {"A": SpikeTensor.from_numpy(A,   "A",       core_id=core_id),
                   "B": SpikeTensor.from_numpy(B,   "B",       core_id=core_id)}
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id)}
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # [4] Benchmark
    print("\n[4/4] Benchmarking (all cores in parallel, NUMA-pinned)...")

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
    warmup_threads = make_threads(warmup_iterations, dummy)
    for t in warmup_threads: t.start()
    for t in warmup_threads: t.join()

    # Benchmark
    barrier = threading.Barrier(num_cores)
    results_ms = [None] * num_cores
    bench_threads = make_threads(benchmark_iterations, results_ms)
    for t in bench_threads: t.start()
    for t in bench_threads: t.join()

    # Stats
    total_wall_ms = max(results_ms)
    mean_ms_per_iter = total_wall_ms / benchmark_iterations
    flops_per_gemm = 2 * size * size * size
    aggregate_tflops = flops_per_gemm * num_cores / (mean_ms_per_iter * 1e-3) / 1e12
    bytes_total = 3 * size * size * 1 * num_cores
    aggregate_bw_gbs = bytes_total / (mean_ms_per_iter * 1e-3) / 1e9

    print("\n  Performance Results:")
    print("  ─────────────────────────────────────")
    print(f"  Cores:                 {num_cores}")
    print(f"  Wall time / iter:      {mean_ms_per_iter:.3f} ms")
    for i, ms in enumerate(results_ms):
        numa = 0 if 16 <= i <= 47 else 1
        print(f"  Core {i:2d} [NUMA {numa}] total ({benchmark_iterations} iters): {ms:.2f} ms")
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
