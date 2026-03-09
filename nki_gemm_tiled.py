#!/usr/bin/env python3
"""
NKI SBUF-Tiled GEMM — Manual tiling for maximum SBUF reuse

Computes C = A @ B in FP8 using explicit SBUF tiling via low-level NKI ISA.

Tile dimensions (tunable; must fit two tiles in SBUF simultaneously):
  M_TILE = 128   (partition dimension, matches NeuronCore SBUF partition count)
  K_TILE = 512   (inner reduction tile)
  N_TILE = 512   (output tile width)

Algorithm:
  For each (m, n) output tile:
    Initialize accumulator acc[M_TILE, N_TILE] = 0  (float32 precision)
    For each k tile:
      DMA A[m*M_TILE:(m+1)*M_TILE, k*K_TILE:(k+1)*K_TILE] → a_sbuf
      DMA B[k*K_TILE:(k+1)*K_TILE, n*N_TILE:(n+1)*N_TILE] → b_sbuf
      acc += nc_matmul(a_sbuf, b_sbuf)   # partial sum
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
M_TILE = 128   # must match hardware partition dim
K_TILE = 512   # inner reduction tile (tunable)
N_TILE = 512   # output column tile (tunable)


@nki.jit(platform_target="trn2")
def nki_gemm_tiled(A, B):
    """
    SBUF-tiled GEMM: C = A @ B

    Args:
        A: HBM tensor, shape (M, K), dtype fp8_e5m2
        B: HBM tensor, shape (K, N), dtype fp8_e5m2

    Returns:
        C: HBM tensor, shape (M, N), dtype float32

    SBUF layout per (m, n) tile iteration:
        a_sbuf : (M_TILE, K_TILE) fp8  — current A tile
        b_sbuf : (K_TILE, N_TILE) fp8  — current B tile
        acc    : (M_TILE, N_TILE) f32  — running partial sum (PSUM or SBUF)

    Note: two tiles (a_sbuf + b_sbuf) must fit in SBUF simultaneously.
    At fp8 (1 byte/elem): 128×512 + 512×512 = 65536 + 262144 = 327 KB.
    trn2 SBUF = 24 MB per core — well within limits.
    """
    M, K = A.shape
    K2, N = B.shape

    # Allocate output in HBM (returned to caller)
    C = hbm.view(dtype=nl.float32, shape=(M, N))  # noqa: F821

    # Outer loops over output tiles
    for m in nl.affine_range(M // M_TILE):
        for n in nl.affine_range(N // N_TILE):

            # Accumulator: zero-initialized, float32 for precision
            # Uses PSUM (hardware-cleared partial-sum buffer) for accumulation
            acc = nl.zeros((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)

            # Inner loop: accumulate K tiles
            for k in nl.affine_range(K // K_TILE):
                # DMA A tile from HBM → SBUF
                a_sbuf = sbuf.view(dtype=A.dtype, shape=(M_TILE, K_TILE))  # noqa: F821
                nisa.dma_copy(
                    dst=a_sbuf,
                    src=A[m * M_TILE:(m + 1) * M_TILE,
                           k * K_TILE:(k + 1) * K_TILE],
                )

                # DMA B tile from HBM → SBUF
                b_sbuf = sbuf.view(dtype=B.dtype, shape=(K_TILE, N_TILE))  # noqa: F821
                nisa.dma_copy(
                    dst=b_sbuf,
                    src=B[k * K_TILE:(k + 1) * K_TILE,
                           n * N_TILE:(n + 1) * N_TILE],
                )

                # Partial matmul: partial[M_TILE, N_TILE] = a_sbuf @ b_sbuf
                # nisa.nc_matmul: moving=a_sbuf (M×K), stationary=b_sbuf (K×N)
                # → result shape (M_TILE, N_TILE)
                partial = nisa.nc_matmul(stationary=b_sbuf, moving=a_sbuf)

                # Accumulate partial sum into acc (float32 addition)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=partial, op=nl.add)

            # Write accumulated output tile back to HBM
            nisa.dma_copy(
                dst=C[m * M_TILE:(m + 1) * M_TILE,
                       n * N_TILE:(n + 1) * N_TILE],
                src=acc,
            )

    return C


def nkipy_wrapper(A, B):
    """NKIPy wrapper required by DeviceKernel.compile_and_load."""
    return nki_gemm_tiled(A, B)


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
    out_np = np.zeros((M, N), dtype=np.float32)
    print(f"  ✓ A: {A_np.shape} {A_np.dtype}, B: {B_np.shape} {B_np.dtype}")

    # ── [2] Compile kernel ───────────────────────────────────────────────────
    print("\n[2/6] Compiling NKI tiled GEMM kernel...")
    t_compile = time.time()
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper, A_np, B_np,
        name="nki_gemm_tiled",
        use_cached_if_exists=True,
    )
    print(f"  ✓ Compiled in {time.time() - t_compile:.2f}s  →  {kernel.neff_path}")

    # ── [3] Create device tensors ────────────────────────────────────────────
    print("\n[3/6] Creating device tensors...")
    device_A = DeviceTensor.from_numpy(A_np)
    device_B = DeviceTensor.from_numpy(B_np)
    device_out = DeviceTensor.from_numpy(out_np)
    print("  ✓ Device tensors allocated")

    # ── [4] Execute and validate ─────────────────────────────────────────────
    print("\n[4/6] Executing kernel + validating against NumPy reference...")
    kernel(
        inputs={"A": device_A, "B": device_B},
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
        print(f"  ✓ Passes tolerance (rtol=1e-1, atol=1e-1)")
        print(f"  ✓ Max abs error: {max_err:.4f},  Mean abs error: {mean_err:.4f}")
    except AssertionError as e:
        print(f"  ✗ Validation FAILED: {e}")

    # ── [5] Profile (NTFF trace) ─────────────────────────────────────────────
    print("\n[5/6] Generating NTFF profile trace...")
    kernel(
        inputs={"A": device_A, "B": device_B},
        outputs={"output0": device_out},
        save_trace=True,
    )
    print(f"  ✓ Profile saved alongside {kernel.neff_path}")

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
    bytes_total = (M * K + K * N + M * N) * 1  # fp8 = 1 byte for inputs; f32=4 for output
    # Use fp8 input size as bottleneck measure
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
