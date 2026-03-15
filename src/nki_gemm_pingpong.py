#!/usr/bin/env python3
"""
NKI Tiled GEMM v3b — C = A.T @ B, PSUM accumulation (no tensor_tensor in inner loop)

Key optimization: accumulate matmul results directly in PSUM using +=,
eliminating 8192 tensor_tensor (PSUM→SBUF) ops from the inner loop.
Only 256 tensor_tensor ops needed at the end to move final results to SBUF for DMA.

Also reuses A tile across N tiles (loaded once per (m, k) pair).

nc_matmul: dst = stationary.T @ moving
  stationary = A tile (M_TILE, K_TILE) from A (M, K)
  moving     = B tile (M_TILE, N_TILE) from B (M, N)
  Contraction over M_TILE (partition dim = 128).
"""

import time

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nkipy.core import nki_op  # noqa: F401 — monkey-patch for NKI jit
from nkipy.runtime import DeviceKernel, DeviceTensor


# ── Tile dimensions ──────────────────────────────────────────────────────────
M_TILE = 128
K_TILE = 128
N_TILE = 512


@nki.jit
def nki_gemm_psum(A, B):
    """
    Tiled GEMM with PSUM accumulation: C = A.T @ B

    Args:
        A: HBM tensor, shape (M, K), dtype fp8_e5m2
        B: HBM tensor, shape (M, N), dtype fp8_e5m2

    Returns:
        C: HBM tensor, shape (K, N), dtype fp8_e5m2
    """
    M, K = A.shape
    M2, N = B.shape

    num_m = M // M_TILE
    num_k = K // K_TILE
    num_n = N // N_TILE

    C = hbm.view(dtype=A.dtype, shape=(K, N))  # noqa: F821

    for k_idx in nl.affine_range(num_k):
        # Contraction over M with PSUM accumulation
        # First iteration: initialize PSUM accumulators from matmul results
        # Subsequent iterations: accumulate via +=
        psum_accs = [None] * num_n

        for mi in nl.affine_range(num_m):
            # Load A tile once per (m, k) — reused across all n-tiles
            a_sbuf = nl.ndarray((M_TILE, K_TILE), dtype=A.dtype, buffer=nl.sbuf)
            nisa.dma_copy(a_sbuf,
                          A[mi * M_TILE:(mi + 1) * M_TILE,
                            k_idx * K_TILE:(k_idx + 1) * K_TILE])

            for ni in nl.affine_range(num_n):
                b_sbuf = nl.ndarray((M_TILE, N_TILE), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(b_sbuf,
                              B[mi * M_TILE:(mi + 1) * M_TILE,
                                ni * N_TILE:(ni + 1) * N_TILE])

                # Accumulate directly in PSUM — no tensor_tensor needed!
                partial = nisa.nc_matmul(a_sbuf, b_sbuf)
                if mi == 0:
                    psum_accs[ni] = partial
                else:
                    psum_accs[ni] += partial

        # Move PSUM → SBUF → HBM (only num_n = 8 tensor_tensor ops total)
        for ni in range(num_n):
            tmp_sbuf = nl.ndarray((K_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(tmp_sbuf, 0)
            nisa.tensor_tensor(tmp_sbuf, tmp_sbuf, psum_accs[ni], nl.add)
            nisa.dma_copy(
                C[k_idx * K_TILE:(k_idx + 1) * K_TILE,
                  ni * N_TILE:(ni + 1) * N_TILE],
                tmp_sbuf)

    return C


def nkipy_wrapper(A, B):
    """NKIPy wrapper required by DeviceKernel.compile_and_load."""
    return nki_gemm_psum(A, B)


def main():
    print("=" * 80)
    print("NKI Tiled GEMM v3b (PSUM accumulation, C = A.T @ B, fp8)")
    print("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────────
    M = K = N = 4096
    warmup_iterations = 5
    benchmark_iterations = 10

    num_m = M // M_TILE
    num_k = K // K_TILE
    num_n = N // N_TILE

    print("\nConfiguration:")
    print(f"  Matrix size:  A({M},{K}) @ B({M},{N}) → C({K},{N})")
    print("  Operation:    C = A.T @ B")
    print("  Data type:    float8_e5m2 (inputs & output)")
    print(f"  Tile dims:    M_TILE={M_TILE}, K_TILE={K_TILE}, N_TILE={N_TILE}")
    print(f"  Inner loop:   {num_m} m-tiles × {num_n} n-tiles = {num_m * num_n} matmuls/k-tile")
    print("  Accumulate:   PSUM += (no tensor_tensor in inner loop)")
    print(f"  A reuse:      each A tile used {num_n}x (across n-tiles)")
    print(f"  DMA loads:    A={num_k * num_m} (was {num_k * num_n * num_m}),"
          f" B={num_k * num_m * num_n} (was {num_k * num_n * num_m})")
    print(f"  Warmup iter:  {warmup_iterations}")
    print(f"  Bench iter:   {benchmark_iterations}")

    # ── [1] Create test data ─────────────────────────────────────────────────
    print("\n[1/6] Creating test data...")
    np.random.seed(42)
    A_np = ((np.random.rand(M, K) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_np = ((np.random.rand(M, N) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    out_np = np.zeros((K, N), dtype=ml_dtypes.float8_e5m2)
    print(f"  A: {A_np.shape} {A_np.dtype}, B: {B_np.shape} {B_np.dtype}")

    # ── [2] Compile kernel ───────────────────────────────────────────────────
    print("\n[2/6] Compiling NKI PSUM-accum GEMM v3b kernel...")
    t_compile = time.time()
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper, A_np, B_np,
        name="nki_gemm_psum_v3b_fp8",
        use_cached_if_exists=True,
    )
    print(f"  Compiled in {time.time() - t_compile:.2f}s  →  {kernel.neff_path}")

    # ── [3] Create device tensors ────────────────────────────────────────────
    print("\n[3/6] Creating device tensors...")
    device_A = DeviceTensor.from_numpy(A_np)
    device_B = DeviceTensor.from_numpy(B_np)
    device_out = DeviceTensor.from_numpy(out_np)
    print("  Device tensors allocated")

    # ── [4] Execute and validate ─────────────────────────────────────────────
    print("\n[4/6] Executing kernel + validating against NumPy reference...")
    kernel(
        inputs={"A": device_A, "B": device_B},
        outputs={"output0": device_out},
    )
    result = device_out.numpy()

    A_f32 = A_np.astype(np.float32)
    B_f32 = B_np.astype(np.float32)
    ref = A_f32.T @ B_f32
    ref_fp8 = ref.astype(ml_dtypes.float8_e5m2).astype(np.float32)
    result_f32 = result.astype(np.float32)

    max_err = np.max(np.abs(result_f32 - ref_fp8))
    mean_err = np.mean(np.abs(result_f32 - ref_fp8))
    print(f"  Max abs error: {max_err:.4f},  Mean abs error: {mean_err:.4f}")

    # ── [5] Profile (NTFF trace) ─────────────────────────────────────────────
    print("\n[5/6] Generating NTFF profile trace...")
    kernel(
        inputs={"A": device_A, "B": device_B},
        outputs={"output0": device_out},
        save_trace=True,
    )
    print(f"  Profile saved alongside {kernel.neff_path}")

    # ── [6] Benchmark ────────────────────────────────────────────────────────
    print("\n[6/6] Benchmarking...")
    stats = kernel.benchmark(
        inputs={"A": device_A, "B": device_B},
        outputs={"output0": device_out},
        warmup_iter=warmup_iterations,
        benchmark_iter=benchmark_iterations,
    )

    flops = 2 * M * K * N
    mean_tflops = flops / (stats.mean_ms * 1e-3) / 1e12
    peak_tflops = flops / (stats.min_ms  * 1e-3) / 1e12
    bytes_total = 3 * M * K * 1
    mean_bw_gbs = bytes_total / (stats.mean_ms * 1e-3) / 1e9

    print("\n  Performance Results:")
    print("  ─────────────────────────────────────")
    print(f"  Mean time:         {stats.mean_ms:.3f} ms")
    print(f"  Min  time:         {stats.min_ms:.3f} ms")
    print(f"  Max  time:         {stats.max_ms:.3f} ms")
    print(f"  Std dev:           {stats.std_dev_ms:.3f} ms")
    print("  ─────────────────────────────────────")
    print(f"  Throughput (mean): {mean_tflops:.2f} TFLOPS")
    print(f"  Throughput (peak): {peak_tflops:.2f} TFLOPS")
    print(f"  Memory BW (mean):  {mean_bw_gbs:.2f} GB/s")
    print("  ─────────────────────────────────────")
    print(f"  vs nkipy compiler: {mean_tflops / 141 * 100:.1f}% of 141 TFLOPS")
    print(f"  vs v1 (baseline):  {6.559 / stats.mean_ms:.2f}x speedup")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
