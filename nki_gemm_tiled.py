#!/usr/bin/env python3
"""
NKI SBUF-Tiled GEMM v3 — C = A.T @ B, blocked tiling to reduce DMA traffic

Key optimization: BLOCK_K blocking reuses B tiles across multiple K-tiles.

Loop structure:
  for k_block (K // K_TILE // BLOCK_K):
    init BLOCK_K × num_n accumulators in SBUF
    for m (contraction):
      load all num_n B tiles once (reused across BLOCK_K k-tiles)
      for kl (BLOCK_K):
        load 1 A tile
        for n:
          matmul + accumulate
    store output tiles

DMA reduction:
  v1: A=8192 loads (512MB), B=8192 loads (512MB) → 1024MB total
  v3: A=1024 loads (16MB),  B=2048 loads (128MB) → 144MB total (7x reduction)

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
M_TILE = 128   # contraction/partition dim (must be ≤ 128)
K_TILE = 128   # stationary free dim (must be ≤ 128)
N_TILE = 512   # moving free dim (≤ 512 on trn2)
BLOCK_K = 4    # process 4 K-tiles per outer block → 32 accumulators in SBUF (~8MB)


@nki.jit
def nki_gemm_tiled(A, B):
    """
    Blocked tiled GEMM: C = A.T @ B

    Args:
        A: HBM tensor, shape (M, K), dtype fp8_e5m2
        B: HBM tensor, shape (M, N), dtype fp8_e5m2

    Returns:
        C: HBM tensor, shape (K, N), dtype fp8_e5m2
    """
    M, K = A.shape
    M2, N = B.shape

    num_m = M // M_TILE       # 32 (contraction tiles)
    num_k = K // K_TILE       # 32 (output K tiles)
    num_n = N // N_TILE       # 8  (output N tiles)
    num_kb = num_k // BLOCK_K # 8  (K blocks)

    C = hbm.view(dtype=A.dtype, shape=(K, N))  # noqa: F821

    for kb in nl.affine_range(num_kb):
        # ── Initialize BLOCK_K × num_n accumulators in SBUF ─────────────
        accs = []
        for kl in range(BLOCK_K):
            for ni in range(num_n):
                acc = nl.ndarray((K_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(acc, 0)
                accs.append(acc)

        # ── Contraction loop over M tiles ───────────────────────────────
        for mi in nl.affine_range(num_m):
            # Load ALL B tiles for this m-tile (reused across BLOCK_K k-tiles)
            b_bufs = []
            for ni in range(num_n):
                b = nl.ndarray((M_TILE, N_TILE), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(b, B[mi * M_TILE:(mi + 1) * M_TILE,
                                   ni * N_TILE:(ni + 1) * N_TILE])
                b_bufs.append(b)

            # For each k-tile in this block, load A once and do all N matmuls
            for kl in nl.affine_range(BLOCK_K):
                a_sbuf = nl.ndarray((M_TILE, K_TILE), dtype=A.dtype, buffer=nl.sbuf)
                nisa.dma_copy(a_sbuf,
                              A[mi * M_TILE:(mi + 1) * M_TILE,
                                (kb * BLOCK_K + kl) * K_TILE:
                                (kb * BLOCK_K + kl + 1) * K_TILE])

                for ni in nl.affine_range(num_n):
                    partial = nl.ndarray((K_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(partial, a_sbuf, b_bufs[ni])
                    nisa.tensor_tensor(accs[kl * num_n + ni],
                                       accs[kl * num_n + ni], partial, nl.add)

        # ── Store BLOCK_K × num_n output tiles ─────────────────────────
        for kl in range(BLOCK_K):
            for ni in range(num_n):
                k_idx = kb * BLOCK_K + kl
                nisa.dma_copy(
                    C[k_idx * K_TILE:(k_idx + 1) * K_TILE,
                      ni * N_TILE:(ni + 1) * N_TILE],
                    accs[kl * num_n + ni])

    return C


def nkipy_wrapper(A, B):
    """NKIPy wrapper required by DeviceKernel.compile_and_load."""
    return nki_gemm_tiled(A, B)


def main():
    print("=" * 80)
    print("NKI Blocked Tiled GEMM v3 (C = A.T @ B, fp8, DMA-optimized)")
    print("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────────
    M = K = N = 4096
    warmup_iterations = 5
    benchmark_iterations = 10

    num_m = M // M_TILE
    num_k = K // K_TILE
    num_n = N // N_TILE

    a_dma = (num_k // BLOCK_K) * num_m * BLOCK_K
    b_dma = (num_k // BLOCK_K) * num_m * num_n
    a_dma_orig = num_k * num_n * num_m
    b_dma_orig = a_dma_orig

    print("\nConfiguration:")
    print(f"  Matrix size:  A({M},{K}) @ B({M},{N}) → C({K},{N})")
    print("  Operation:    C = A.T @ B")
    print("  Data type:    float8_e5m2 (inputs & output)")
    print(f"  Tile dims:    M_TILE={M_TILE}, K_TILE={K_TILE}, N_TILE={N_TILE}")
    print(f"  BLOCK_K:      {BLOCK_K} (B tiles reused across {BLOCK_K} k-tiles)")
    print(f"  SBUF usage:   ~{BLOCK_K * num_n * K_TILE * N_TILE * 4 / 1e6:.1f}MB accumulators"
          f" + ~{num_n * M_TILE * N_TILE * 1 / 1e6:.1f}MB B tiles")
    print(f"  DMA loads:    A={a_dma} (was {a_dma_orig}), B={b_dma} (was {b_dma_orig})")
    print(f"  DMA volume:   ~{(a_dma * M_TILE * K_TILE + b_dma * M_TILE * N_TILE) / 1e6:.0f}MB"
          f" (was ~{(a_dma_orig * M_TILE * K_TILE + b_dma_orig * M_TILE * N_TILE) / 1e6:.0f}MB)")
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
    print("\n[2/6] Compiling NKI blocked tiled GEMM v3 kernel...")
    t_compile = time.time()
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper, A_np, B_np,
        name="nki_gemm_blocked_v3_fp8",
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
    print(f"  vs v1 (no block):  {6.559 / stats.mean_ms:.2f}x speedup")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
