#!/usr/bin/env python3
"""
NKI SBUF-Tiled GEMM — Manual tiling for maximum SBUF reuse

Computes C = A @ B in FP8 using explicit SBUF tiling via low-level NKI ISA.

nc_matmul computes: dst = stationary.T @ moving, contracting over partition dim.
To compute A @ B, we pass A^T as stationary (K on partition dim) and B as moving.
The kernel accepts A_T (pre-transposed) to avoid on-device transpose overhead.

Tile dimensions:
  M_TILE = 128   (stationary free dim, ≤ 128 for systolic array)
  K_TILE = 128   (partition/contraction dim, ≤ 128 for SBUF partitions)
  N_TILE = 512   (moving free dim, can be > 128)

Algorithm:
  For each (m, n) output tile:
    Initialize accumulator acc[M_TILE, N_TILE] = 0  (float32 precision)
    For each k tile:
      DMA A_T[k*K_TILE:(k+1)*K_TILE, m*M_TILE:(m+1)*M_TILE] → a_T_sbuf
      DMA B[k*K_TILE:(k+1)*K_TILE, n*N_TILE:(n+1)*N_TILE]   → b_sbuf
      partial = nc_matmul(stationary=a_T_sbuf, moving=b_sbuf)  # a_T.T @ b = A_tile @ B_tile
      acc += partial
    DMA acc → C[m*M_TILE:(m+1)*M_TILE, n*N_TILE:(n+1)*N_TILE]

References:
  simple_nki_kernel.py  — sbuf.view, hbm.view, nisa.dma_copy, @nki.jit patterns
  simple_nkipy_kernel_fp8.py — FP8 benchmark harness
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
M_TILE = 128   # stationary free dim (must be ≤ 128)
K_TILE = 128   # contraction/partition dim (must be ≤ 128)
N_TILE = 512   # moving free dim (can exceed 128)


@nki.jit(platform_target="trn2")
def nki_gemm_tiled(A_T, B):
    """
    SBUF-tiled GEMM: C = A @ B, with A_T = A.T pre-transposed.

    Args:
        A_T: HBM tensor, shape (K, M), dtype fp8_e5m2 — transposed A
        B:   HBM tensor, shape (K, N), dtype fp8_e5m2

    Returns:
        C: HBM tensor, shape (M, N), dtype float32

    nc_matmul semantics: dst = stationary.T @ moving
      stationary = A_T tile (K_TILE, M_TILE) — ≤ 128x128 constraint satisfied
      moving     = B tile   (K_TILE, N_TILE)
      dst        = A_T.T @ B = A_tile @ B_tile → (M_TILE, N_TILE)
    """
    K, M = A_T.shape
    K2, N = B.shape

    # Allocate output in HBM (returned to caller)
    C = hbm.view(dtype=nl.float32, shape=(M, N))  # noqa: F821

    # Outer loops over output tiles (use range() to avoid tensor name conflicts)
    for m in range(M // M_TILE):
        for n in range(N // N_TILE):

            # SBUF accumulator (zero-initialized)
            acc_sbuf = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(acc_sbuf, 0)

            # Inner loop: accumulate K tiles
            for k in range(K // K_TILE):
                # DMA A^T tile from HBM → SBUF: shape (K_TILE, M_TILE)
                # K is on partition dim (axis 0) for nc_matmul contraction
                a_T_sbuf = nl.ndarray((K_TILE, M_TILE), dtype=A_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    a_T_sbuf,
                    A_T[k * K_TILE:(k + 1) * K_TILE,
                        m * M_TILE:(m + 1) * M_TILE],
                )

                # DMA B tile from HBM → SBUF: shape (K_TILE, N_TILE)
                b_sbuf = nl.ndarray((K_TILE, N_TILE), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    b_sbuf,
                    B[k * K_TILE:(k + 1) * K_TILE,
                      n * N_TILE:(n + 1) * N_TILE],
                )

                # Matmul: dst = a_T.T @ b = A_tile @ B_tile → (M_TILE, N_TILE)
                # stationary=a_T_sbuf (128x128 ≤ limit), moving=b_sbuf
                partial_psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(partial_psum, a_T_sbuf, b_sbuf)

                # Accumulate PSUM partial into SBUF accumulator
                nisa.tensor_tensor(acc_sbuf, acc_sbuf, partial_psum, nl.add)

            # Write accumulated output tile back to HBM
            nisa.dma_copy(
                C[m * M_TILE:(m + 1) * M_TILE,
                  n * N_TILE:(n + 1) * N_TILE],
                acc_sbuf,
            )

    return C


def nkipy_wrapper(A_T, B):
    """NKIPy wrapper required by DeviceKernel.compile_and_load."""
    return nki_gemm_tiled(A_T, B)


def main():
    print("=" * 80)
    print("NKI SBUF-Tiled GEMM Benchmark")
    print("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────────
    M = K = N = 4096
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size: {M}×{K} @ {K}×{N}")
    print("  Data type:   float8_e5m2 (inputs) → float32 (output)")
    print(f"  Tile dims:   M_TILE={M_TILE}, K_TILE={K_TILE}, N_TILE={N_TILE}")
    print(f"  K tiles:     {K // K_TILE}")
    print(f"  Warmup iter: {warmup_iterations}")
    print(f"  Bench iter:  {benchmark_iterations}")

    # ── [1] Create test data ─────────────────────────────────────────────────
    print("\n[1/6] Creating test data...")
    np.random.seed(42)
    A_np = ((np.random.rand(M, K) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    B_np = ((np.random.rand(K, N) - 0.5) * 2).astype(ml_dtypes.float8_e5m2)
    # Pre-transpose A for the kernel (nc_matmul needs K on partition dim)
    A_T_np = np.ascontiguousarray(A_np.T)
    out_np = np.zeros((M, N), dtype=np.float32)
    print(f"  A: {A_np.shape} {A_np.dtype}, A_T: {A_T_np.shape}, B: {B_np.shape} {B_np.dtype}")

    # ── [2] Compile kernel ───────────────────────────────────────────────────
    print("\n[2/6] Compiling NKI tiled GEMM kernel...")
    t_compile = time.time()
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper, A_T_np, B_np,
        name="nki_gemm_tiled",
        use_cached_if_exists=True,
    )
    print(f"  Compiled in {time.time() - t_compile:.2f}s  →  {kernel.neff_path}")

    # ── [3] Create device tensors ────────────────────────────────────────────
    print("\n[3/6] Creating device tensors...")
    device_A_T = DeviceTensor.from_numpy(A_T_np)
    device_B = DeviceTensor.from_numpy(B_np)
    device_out = DeviceTensor.from_numpy(out_np)
    print("  Device tensors allocated")

    # ── [4] Execute and validate ─────────────────────────────────────────────
    print("\n[4/6] Executing kernel + validating against NumPy reference...")
    kernel(
        inputs={"A_T": device_A_T, "B": device_B},
        outputs={"output0": device_out},
    )
    result = device_out.numpy()

    # FP8 reference: compute in float32 from fp8 inputs (quantized precision)
    A_f32 = A_np.astype(np.float32)
    B_f32 = B_np.astype(np.float32)
    ref = A_f32 @ B_f32

    try:
        np.testing.assert_allclose(result, ref, rtol=1e-1, atol=1e-1)
        max_err = np.max(np.abs(result - ref))
        mean_err = np.mean(np.abs(result - ref))
        print(f"  Passes tolerance (rtol=1e-1, atol=1e-1)")
        print(f"  Max abs error: {max_err:.4f},  Mean abs error: {mean_err:.4f}")
    except AssertionError as e:
        print(f"  Validation FAILED: {e}")

    # ── [5] Profile (NTFF trace) ─────────────────────────────────────────────
    print("\n[5/6] Generating NTFF profile trace...")
    kernel(
        inputs={"A_T": device_A_T, "B": device_B},
        outputs={"output0": device_out},
        save_trace=True,
    )
    print(f"  Profile saved alongside {kernel.neff_path}")

    # ── [6] Benchmark ────────────────────────────────────────────────────────
    print("\n[6/6] Benchmarking...")
    stats = kernel.benchmark(
        inputs={"A_T": device_A_T, "B": device_B},
        outputs={"output0": device_out},
        warmup_iter=warmup_iterations,
        benchmark_iter=benchmark_iterations,
    )

    flops = 2 * M * K * N
    mean_tflops = flops / (stats.mean_ms * 1e-3) / 1e12
    peak_tflops = flops / (stats.min_ms  * 1e-3) / 1e12
    bytes_fp8 = (M * K + K * N) * 1 + M * N * 4
    mean_bw_gbs = bytes_fp8 / (stats.mean_ms * 1e-3) / 1e9

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
    print(f"  Arithmetic intens: {flops / bytes_fp8:.0f} FLOP/byte")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
