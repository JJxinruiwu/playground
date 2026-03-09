#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Multi-Core Matrix Multiplication Example

Runs the same GEMM kernel on multiple NeuronCores in parallel using Python threading.
Each core operates on independent data (data parallelism — NOT tensor parallelism).

NeuronCores are independent compute units (like individual GPUs). To use them:
  1. Compile once → get a .neff file
  2. Load the same NEFF on each core with a different core_id
  3. Allocate SpikeTensors on each core with the matching core_id
  4. Launch all cores simultaneously via threading
  5. Aggregate throughput = num_cores × single-core throughput
"""

import threading
import time

import ml_dtypes
import numpy as np

import os

import spike
from nkipy.runtime import DeviceKernel
from spike import SpikeModel
from spike.spike_tensor import SpikeTensor


def nkipy_kernel(A, B):
    return A @ B


def run_on_core(model, inputs, outputs, barrier, results, core_idx, iterations):
    """Worker: wait at barrier, then execute kernel N times, record wall time."""
    barrier.wait()  # synchronize launch across all threads
    start = time.perf_counter()
    for _ in range(iterations):
        model(inputs=inputs, outputs=outputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    results[core_idx] = elapsed_ms


def main():
    print("=" * 80)
    print("Multi-Core Matrix Multiplication (Data Parallel)")
    print("=" * 80)

    # Configuration
    size = 4096
    num_cores = 64          # number of NeuronCores to use (4 per device × 2 devices)
    # Allow NRT to accept all submissions without blocking on each completion.
    # Without this, each spike.execute() serializes: submit→wait→return, causing
    # thundering-herd contention when 64 threads hammer the NRT submission lock.
    # os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = str(num_cores)
    spike.configure(visible_cores=range(num_cores))  # must be called before any spike op
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size:  {size}x{size}")
    print(f"  Num cores:    {num_cores}")
    print("  Data type:    float8_e5m2")
    print(f"  Warmup iter:  {warmup_iterations}")
    print(f"  Bench iter:   {benchmark_iterations}")

    # [1] Compile once — DeviceKernel loads on core_id=0 by default
    print("\n[1/4] Compiling kernel (once)...")
    np.random.seed(42)
    A_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_proto = ((np.random.rand(size, size) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)

    t0 = time.time()
    kernel_core0 = DeviceKernel.compile_and_load(
        nkipy_kernel,
        A_proto,
        B_proto,
        name="simple_kernel",
        use_cached_if_exists=True,
    )
    print(f"  ✓ Compiled in {time.time() - t0:.2f}s  →  {kernel_core0.neff_path}")

    # [2] Load same NEFF on all cores
    print(f"\n[2/4] Loading NEFF on {num_cores} cores...")
    models = [kernel_core0]  # core 0 already loaded
    for core_id in range(1, num_cores):
        m = SpikeModel.load_from_neff(kernel_core0.neff_path, core_id=core_id)
        models.append(m)
        print(f"  ✓ Loaded on core {core_id}")

    # [3] Allocate independent input/output tensors per core
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
        outputs = {
            "output0": SpikeTensor.from_numpy(out, "output0", core_id=core_id),
        }
        per_core_io.append((inputs, outputs))
    print(f"  ✓ Allocated tensors on {num_cores} cores")

    # [4] Benchmark: all cores launch simultaneously via threading
    print("\n[4/4] Benchmarking (all cores in parallel)...")

    # Warmup
    barrier = threading.Barrier(num_cores)
    dummy_results = [None] * num_cores
    warmup_threads = [
        threading.Thread(
            target=run_on_core,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier, dummy_results, i, warmup_iterations),
        )
        for i in range(num_cores)
    ]
    for t in warmup_threads:
        t.start()
    for t in warmup_threads:
        t.join()

    # Benchmark
    barrier = threading.Barrier(num_cores)
    results_ms = [None] * num_cores
    bench_threads = [
        threading.Thread(
            target=run_on_core,
            args=(models[i], per_core_io[i][0], per_core_io[i][1],
                  barrier, results_ms, i, benchmark_iterations),
        )
        for i in range(num_cores)
    ]
    for t in bench_threads:
        t.start()
    for t in bench_threads:
        t.join()

    # Stats — use max wall time across cores (they run in parallel)
    total_wall_ms = max(results_ms)
    mean_ms_per_iter = total_wall_ms / benchmark_iterations

    flops_per_gemm = 2 * size * size * size
    total_flops = flops_per_gemm * num_cores  # all cores per iteration
    aggregate_tflops = total_flops / (mean_ms_per_iter * 1e-3) / 1e12

    bytes_per_core = 3 * size * size * 1   # fp8 = 1 byte, read A+B write out
    total_bytes = bytes_per_core * num_cores
    aggregate_bw_gbs = total_bytes / (mean_ms_per_iter * 1e-3) / 1e9

    print("\n  Performance Results:")
    print("  ─────────────────────────────────────")
    print(f"  Cores:                 {num_cores}")
    print(f"  Wall time / iter:      {mean_ms_per_iter:.3f} ms")
    for i, ms in enumerate(results_ms):
        print(f"  Core {i} total ({benchmark_iterations} iters): {ms:.2f} ms")
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
