#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication v5 — v4 + v2 + v3 (all three optimizations combined)

Combines:
  v4: Batched kernel (KERNEL_LOOP=20 on-device GEMMs per host dispatch)
  v2: Double-barrier synchronized rounds — main thread participates as barrier party,
      giving clean per-round wall times from a single reference clock
  v3: NUMA-aware CPU thread pinning via os.sched_setaffinity

Result:
  - 20× fewer host dispatches (from v4)
  - Zero inter-round queue pressure, clean per-round timing (from v2)
  - Reduced host↔device scheduling latency via NUMA locality (from v3)
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


# ── NUMA pinning table (trn2.48xlarge) ──────────────────────────────────────
# core_ids  0-15  → Devices  0-3  → NUMA node 1 → CPUs 48-95, 144-191
# core_ids 16-47  → Devices  4-11 → NUMA node 0 → CPUs  0-47,  96-143
# core_ids 48-63  → Devices 12-15 → NUMA node 1 → CPUs 48-95, 144-191
_NUMA0_CPUS = list(range(0, 48)) + list(range(96, 144))
_NUMA1_CPUS = list(range(48, 96)) + list(range(144, 192))


def _cpus_for_core(core_id: int) -> list:
    """Return the preferred CPU affinity list for a given NeuronCore ID."""
    if 16 <= core_id <= 47:
        return _NUMA0_CPUS
    return _NUMA1_CPUS


def nkipy_kernel(A, B):
    # A: (KERNEL_LOOP, size, size), B: (KERNEL_LOOP, size, size)
    # Batched matmul: C[i] = A[i] @ B[i] for each i independently.
    return A @ B


def run_on_core(model, inputs, outputs, barrier_start, barrier_end,
                total_iterations, warmup_iterations, core_results, core_idx):
    """
    Worker thread: pin to NUMA-local CPUs, then run synchronized rounds.

    Each round:
      1. Wait at barrier_start (main releases all workers simultaneously)
      2. Execute kernel (one host dispatch = KERNEL_LOOP on-device GEMMs)
      3. Wait at barrier_end (main records round wall time after last worker finishes)

    Records per-round times for benchmark rounds (excluding warmup) in core_results.
    """
    # NUMA pinning (v3)
    try:
        os.sched_setaffinity(0, _cpus_for_core(core_idx))
    except OSError:
        pass  # not supported on all platforms; continue without pinning

    per_round_ms = []
    for i in range(total_iterations):
        barrier_start.wait()          # synchronized start (v2)
        t0 = time.perf_counter()
        model(inputs=inputs, outputs=outputs)
        # Force device sync: read output back to host so we wait for all
        # queued async work to complete before stopping the timer.
        outputs["output0"].numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        barrier_end.wait()            # synchronized end (v2)
        if i >= warmup_iterations:
            per_round_ms.append(elapsed_ms)

    core_results[core_idx] = per_round_ms


def main():
    print("=" * 80)
    print(f"Multi-Core Matrix Multiplication v5 (v4 batched + v2 sync + v3 NUMA)")
    print("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────────
    size = 4096
    num_cores = 64
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "63"
    spike.configure(visible_cores=range(num_cores))
    warmup_iterations = 5
    benchmark_iterations = 10
    total_iterations = warmup_iterations + benchmark_iterations

    print("\nConfiguration:")
    print(f"  Matrix size:      {KERNEL_LOOP}×{size}×{size} (batched)")
    print(f"  Num cores:        {num_cores}")
    print("  Data type:        float8_e5m2")
    print(f"  Kernel loop:      {KERNEL_LOOP} independent GEMMs per dispatch")
    print(f"  Warmup rounds:    {warmup_iterations}")
    print(f"  Bench rounds:     {benchmark_iterations}")
    print(f"  Total GEMMs/core: {benchmark_iterations * KERNEL_LOOP}")
    print("  Barrier:          double (start + end), main thread participates")
    print("  NUMA pinning:     enabled")

    # ── [1] Compile once ─────────────────────────────────────────────────────
    print("\n[1/4] Compiling kernel (once)...")
    np.random.seed(42)
    A_proto = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    t0 = time.time()
    kernel_core0 = DeviceKernel.compile_and_load(
        nkipy_kernel, A_proto, B_proto,
        name="simple_kernel_loop20",
        use_cached_if_exists=True,
    )
    print(f"  ✓ Compiled in {time.time() - t0:.2f}s  →  {kernel_core0.neff_path}")

    # ── [2] Load same NEFF on all cores ──────────────────────────────────────
    print(f"\n[2/4] Loading NEFF on {num_cores} cores...")
    models = [kernel_core0]
    for core_id in range(1, num_cores):
        models.append(SpikeModel.load_from_neff(kernel_core0.neff_path, core_id=core_id))
    print(f"  ✓ Loaded on all {num_cores} cores")

    # ── [3] Allocate independent input/output tensors per core ───────────────
    print("\n[3/4] Allocating tensors per core...")
    per_core_io = []
    for core_id in range(num_cores):
        A = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        B = ((np.random.rand(KERNEL_LOOP, size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
        out = np.zeros((KERNEL_LOOP, size, size), dtype=ml_dtypes.float8_e5m2)
        inputs = {
            "A": SpikeTensor.from_numpy(A, "A", core_id=core_id),
            "B": SpikeTensor.from_numpy(B, "B", core_id=core_id),
        }
        outputs = {"output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id)}
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # ── [4] Double-barrier synchronized benchmark ─────────────────────────────
    print("\n[4/4] Benchmarking (synchronized rounds, NUMA-pinned)...")

    # Both barriers sized num_cores+1 so main thread participates (v2 pattern)
    barrier_start = threading.Barrier(num_cores + 1)
    barrier_end = threading.Barrier(num_cores + 1)
    core_results = [None] * num_cores  # list of per-round ms per core

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

    # Main thread drives both barriers; records per-round wall time
    round_wall_ms = []
    for i in range(total_iterations):
        barrier_start.wait()          # release all workers
        t_round = time.perf_counter()
        barrier_end.wait()            # wait for all workers (including slowest)
        elapsed_ms = (time.perf_counter() - t_round) * 1000.0
        if i >= warmup_iterations:
            round_wall_ms.append(elapsed_ms)
        phase = "warmup" if i < warmup_iterations else "bench "
        print(f"    [{phase} round {i+1:2d}] {elapsed_ms:.3f} ms", flush=True)

    for t in threads:
        t.join()

    # ── Stats ────────────────────────────────────────────────────────────────
    flops_per_gemm = 2 * size * size * size
    flops_per_dispatch = flops_per_gemm * KERNEL_LOOP  # one dispatch = KERNEL_LOOP GEMMs
    flops_per_round_all_cores = flops_per_dispatch * num_cores

    mean_ms = sum(round_wall_ms) / len(round_wall_ms)
    min_ms  = min(round_wall_ms)
    max_ms  = max(round_wall_ms)

    # Throughput: all cores × KERNEL_LOOP GEMMs per round
    mean_tflops = flops_per_round_all_cores / (mean_ms * 1e-3) / 1e12
    peak_tflops = flops_per_round_all_cores / (min_ms  * 1e-3) / 1e12

    # Per-GEMM wall time (from main thread's round time)
    mean_ms_per_gemm = mean_ms / KERNEL_LOOP

    # Memory BW (fp8 = 1 byte): A + B read + C written per GEMM per core
    bytes_per_round = 3 * KERNEL_LOOP * size * size * 1 * num_cores
    aggregate_bw_gbs = bytes_per_round / (mean_ms * 1e-3) / 1e9

    print("\n  Per-Round Throughput:")
    print("  ─────────────────────────────────────")
    for idx, d in enumerate(round_wall_ms):
        tflops = flops_per_round_all_cores / (d * 1e-3) / 1e12
        ms_per_g = d / KERNEL_LOOP
        print(f"  Round {idx+1:2d}: {d:.3f} ms  ({ms_per_g:.4f} ms/GEMM)  →  {tflops:.2f} TFLOPS")

    print("\n  Per-Core Cumulative Time (straggler analysis):")
    print("  ─────────────────────────────────────")
    for i, per_round in enumerate(core_results):
        total_ms = sum(per_round)
        per_core_tflops = (flops_per_dispatch * benchmark_iterations) / (total_ms * 1e-3) / 1e12
        numa_node = 0 if 16 <= i <= 47 else 1
        print(f"  Core {i:2d} [NUMA {numa_node}]: {total_ms:.2f} ms total  "
              f"({per_core_tflops:.2f} TFLOPS/core)")

    print("\n  Summary:")
    print("  ─────────────────────────────────────")
    print(f"  Cores:                       {num_cores}")
    print(f"  Kernel loop:                 {KERNEL_LOOP} GEMMs/dispatch")
    print(f"  Mean round time:             {mean_ms:.3f} ms")
    print(f"  Min  round time:             {min_ms:.3f} ms")
    print(f"  Max  round time:             {max_ms:.3f} ms")
    print(f"  Mean time per GEMM:          {mean_ms_per_gemm:.4f} ms")
    print("  ─────────────────────────────────────")
    print(f"  Aggregate throughput (mean): {mean_tflops:.2f} TFLOPS")
    print(f"  Aggregate throughput (peak): {peak_tflops:.2f} TFLOPS")
    print(f"  Per-core throughput (mean):  {mean_tflops / num_cores:.2f} TFLOPS")
    print(f"  Aggregate memory BW:         {aggregate_bw_gbs:.2f} GB/s")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
