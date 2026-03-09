#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication v2 — Synchronized Round-Based Execution

v1 problem: 64 threads each loop 10 times independently → up to 640 submissions
fighting the NRT queue (cap=63), causing backpressure and high per-core variance.

v2 fix: 10 synchronized rounds. Each round:
  1. Main thread releases all 64 workers simultaneously (barrier_start)
  2. All 64 cores submit their kernel (max 64 inflight — 1 per core)
  3. Main thread waits for all 64 to complete (barrier_end)
  4. Main records per-round wall time, repeats

Result: clean per-round timing, no inter-round queue pressure.
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


def nkipy_kernel(A, B):
    return A @ B


def run_on_core(model, inputs, outputs, barrier_start, barrier_end,
                total_iterations, warmup_iterations, core_results, core_idx):
    """Worker: each iteration waits for synchronized start, executes once, signals done.
    Records total wall time for benchmark iterations (excluding warmup) in core_results."""
    bench_elapsed_ms = 0.0
    for i in range(total_iterations):
        barrier_start.wait()
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        barrier_end.wait()
        if i >= warmup_iterations:
            bench_elapsed_ms += elapsed
    core_results[core_idx] = bench_elapsed_ms


def main():
    print("=" * 80)
    print("Multi-Core Matrix Multiplication v2 (Synchronized Rounds)")
    print("=" * 80)

    # Configuration
    size = 4096
    num_cores = 64
    warmup_iterations = 5
    benchmark_iterations = 20

    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "63"
    spike.configure(visible_cores=range(num_cores))

    print("\nConfiguration:")
    print(f"  Matrix size:  {size}x{size}")
    print(f"  Num cores:    {num_cores}")
    print("  Data type:    float8_e5m2")
    print(f"  Warmup iter:  {warmup_iterations}")
    print(f"  Bench iter:   {benchmark_iterations}")
    print("  Mode:         synchronized rounds (all cores per round)")

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

    # [3] Allocate tensors per core
    print("\n[3/4] Allocating tensors per core...")
    per_core_io = []
    for core_id in range(num_cores):
        A = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros((size, size), dtype=ml_dtypes.float8_e5m2)
        inputs = {
            "A": SpikeTensor.from_numpy(A, "A", core_id=core_id),
            "B": SpikeTensor.from_numpy(B, "B", core_id=core_id),
        }
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id)}
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # [4] Synchronized round-based benchmark
    print("\n[4/4] Benchmarking (synchronized rounds)...")

    total_iterations = warmup_iterations + benchmark_iterations
    # +1 so the main thread participates in both barriers
    barrier_start = threading.Barrier(num_cores + 1)
    barrier_end = threading.Barrier(num_cores + 1)
    core_results = [None] * num_cores

    threads = [
        threading.Thread(
            target=run_on_core,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier_start, barrier_end,
                  total_iterations, warmup_iterations, core_results, i),
        )
        for i in range(num_cores)
    ]
    for t in threads:
        t.start()

    durations_ms = []
    for i in range(total_iterations):
        barrier_start.wait()     # release all 64 workers simultaneously
        t0 = time.perf_counter()
        barrier_end.wait()       # wait for all 64 cores to complete
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i >= warmup_iterations:
            durations_ms.append(elapsed_ms)

    for t in threads:
        t.join()

    # Stats
    mean_ms = sum(durations_ms) / len(durations_ms)
    min_ms = min(durations_ms)
    max_ms = max(durations_ms)

    flops_per_round = 2 * size * size * size * num_cores
    flops_single = 2 * size * size * size
    aggregate_tflops_mean = flops_per_round / (mean_ms * 1e-3) / 1e12
    aggregate_tflops_peak = flops_per_round / (min_ms * 1e-3) / 1e12
    bytes_per_round = 3 * size * size * 1 * num_cores
    aggregate_bw_gbs = bytes_per_round / (mean_ms * 1e-3) / 1e9

    # Per-iteration throughput
    iter_tflops = [flops_per_round / (d * 1e-3) / 1e12 for d in durations_ms]

    # Per-core wall time (total across all benchmark iterations)
    core_total_ms = core_results  # filled by workers

    print("\n  Per-Iteration Throughput:")
    print("  ─────────────────────────────────────")
    for idx, (d, tflops) in enumerate(zip(durations_ms, iter_tflops)):
        print(f"  Iter {idx+1:2d}: {d:.3f} ms  →  {tflops:.2f} TFLOPS")

    print("\n  Per-Core Wall Time (benchmark iters total):")
    print("  ─────────────────────────────────────")
    for i, ms in enumerate(core_total_ms):
        per_core_tflops = (flops_single * benchmark_iterations) / (ms * 1e-3) / 1e12
        print(f"  Core {i:2d}: {ms:.2f} ms  ({per_core_tflops:.2f} TFLOPS)")

    print("\n  Summary:")
    print("  ─────────────────────────────────────")
    print(f"  Cores:                       {num_cores}")
    print(f"  Mean round time:             {mean_ms:.3f} ms")
    print(f"  Min  round time:             {min_ms:.3f} ms")
    print(f"  Max  round time:             {max_ms:.3f} ms")
    print("  ─────────────────────────────────────")
    print(f"  Aggregate throughput (mean): {aggregate_tflops_mean:.2f} TFLOPS")
    print(f"  Aggregate throughput (peak): {aggregate_tflops_peak:.2f} TFLOPS")
    print(f"  Per-core throughput (mean):  {aggregate_tflops_mean / num_cores:.2f} TFLOPS")
    print(f"  Aggregate memory BW:         {aggregate_bw_gbs:.2f} GB/s")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
