#!/usr/bin/env python3
"""
NKI Ping-Pong Tiled GEMM — DMA/compute overlap via double-buffering

Extends nki_gemm_tiled.py by overlapping the DMA load of K-tile (k+1) with
the nc_matmul compute on K-tile (k), hiding DMA latency behind compute.

Double-buffer scheme:
  Allocate two SBUF buffer pairs: a_buf[0/1], b_buf[0/1]
  Prologue : DMA K-tile 0  → buf[0]
  Main loop (k = 0 .. num_k_tiles-2):
    1. Issue DMA for tile k+1 → buf[(k+1)%2]   ← async, overlaps with step 2
    2. nc_matmul on buf[k%2]                    ← compute while DMA runs
    3. Accumulate partial → acc
    (NKI compiler schedules DMA engine and MXU independently)
  Epilogue : nc_matmul on last tile

nc_matmul computes: dst = stationary.T @ moving, contracting over partition dim.
To compute A @ B, we pass A^T as stationary (K on partition dim) and B as moving.
The kernel accepts A_T (pre-transposed) to keep K on the partition dimension.

Expected gain: hides ~DMA latency of each K-tile load behind MXU compute.
To confirm overlap, compare per-K-tile execution time in the NTFF trace
against the non-ping-pong version (nki_gemm_tiled.py).

References:
  nki_gemm_tiled.py    — base tiled structure extended here
  simple_nki_kernel.py — DMA and SBUF patterns
"""

import time

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nkipy.core import nki_op  # noqa: F401 — monkey-patch for NKI jit
from nkipy.runtime import DeviceKernel, DeviceTensor


# ── Tile dimensions (must match nki_gemm_tiled.py for fair comparison) ───────
M_TILE = 128
K_TILE = 128
N_TILE = 512


@nki.jit(platform_target="trn2")
def nki_gemm_pingpong(A_T, B):
    """
    Ping-pong double-buffered tiled GEMM: C = A @ B, with A_T = A.T

    Args:
        A_T: HBM tensor, shape (K, M), dtype fp8_e5m2 — transposed A
        B:   HBM tensor, shape (K, N), dtype fp8_e5m2

    Returns:
        C: HBM tensor, shape (M, N), dtype float32

    SBUF layout per (m, n) tile:
        a_buf[0], a_buf[1] : (K_TILE, M_TILE) fp8  — ping-pong A^T buffers
        b_buf[0], b_buf[1] : (K_TILE, N_TILE) fp8  — ping-pong B buffers
        acc                : (M_TILE, N_TILE) f32  — running partial sum
    """
    K, M = A_T.shape
    K2, N = B.shape
    num_k_tiles = K // K_TILE

    C = hbm.view(dtype=nl.float32, shape=(M, N))  # noqa: F821

    for m in range(M // M_TILE):
        for n in range(N // N_TILE):

            # SBUF accumulator (zero-initialized)
            acc_sbuf = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(acc_sbuf, 0)

            # Allocate two SBUF ping-pong buffer pairs
            a_buf_0 = nl.ndarray((K_TILE, M_TILE), dtype=A_T.dtype, buffer=nl.sbuf)
            a_buf_1 = nl.ndarray((K_TILE, M_TILE), dtype=A_T.dtype, buffer=nl.sbuf)
            b_buf_0 = nl.ndarray((K_TILE, N_TILE), dtype=B.dtype, buffer=nl.sbuf)
            b_buf_1 = nl.ndarray((K_TILE, N_TILE), dtype=B.dtype, buffer=nl.sbuf)
            a_bufs = (a_buf_0, a_buf_1)
            b_bufs = (b_buf_0, b_buf_1)

            # ── Prologue: prefetch tile 0 into buf[0] ────────────────────────
            nisa.dma_copy(a_bufs[0], A_T[0:K_TILE, m * M_TILE:(m + 1) * M_TILE])
            nisa.dma_copy(b_bufs[0], B[0:K_TILE, n * N_TILE:(n + 1) * N_TILE])

            # ── Main loop: tiles 0 .. num_k_tiles-2 ──────────────────────────
            for k in range(num_k_tiles - 1):
                cur = k % 2
                nxt = (k + 1) % 2

                # Step 1: issue async DMA for next tile into buf[nxt]
                nisa.dma_copy(
                    a_bufs[nxt],
                    A_T[(k + 1) * K_TILE:(k + 2) * K_TILE,
                        m * M_TILE:(m + 1) * M_TILE],
                )
                nisa.dma_copy(
                    b_bufs[nxt],
                    B[(k + 1) * K_TILE:(k + 2) * K_TILE,
                      n * N_TILE:(n + 1) * N_TILE],
                )

                # Step 2: compute matmul on current tile (overlaps with DMA)
                # dst = a_T.T @ b = A_tile @ B_tile
                partial_psum = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(partial_psum, a_bufs[cur], b_bufs[cur])

                # Step 3: accumulate partial sum
                nisa.tensor_tensor(acc_sbuf, acc_sbuf, partial_psum, nl.add)

            # ── Epilogue: compute last K-tile ──────────────────────────────────
            last = (num_k_tiles - 1) % 2
            partial_last = nl.ndarray((M_TILE, N_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(partial_last, a_bufs[last], b_bufs[last])
            nisa.tensor_tensor(acc_sbuf, acc_sbuf, partial_last, nl.add)

            # Write output tile back to HBM
            nisa.dma_copy(
                C[m * M_TILE:(m + 1) * M_TILE,
                  n * N_TILE:(n + 1) * N_TILE],
                acc_sbuf,
            )

    return C


def nkipy_wrapper(A_T, B):
    """NKIPy wrapper required by DeviceKernel.compile_and_load."""
    return nki_gemm_pingpong(A_T, B)


def main():
    print("=" * 80)
    print("NKI Ping-Pong Tiled GEMM Benchmark (DMA/Compute Overlap)")
    print("=" * 80)

    # ── Configuration ────────────────────────────────────────────────────────
    M = K = N = 4096
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size:  {M}×{K} @ {K}×{N}")
    print("  Data type:    float8_e5m2 (inputs) → float32 (output)")
    print(f"  Tile dims:    M_TILE={M_TILE}, K_TILE={K_TILE}, N_TILE={N_TILE}")
    print(f"  K tiles:      {K // K_TILE}  (double-buffered)")
    print("  Optimization: DMA/matmul overlap via ping-pong SBUF buffers")
    print(f"  Warmup iter:  {warmup_iterations}")
    print(f"  Bench iter:   {benchmark_iterations}")

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
    print("\n[2/6] Compiling NKI ping-pong GEMM kernel...")
    t_compile = time.time()
    kernel = DeviceKernel.compile_and_load(
        nkipy_wrapper, A_T_np, B_np,
        name="nki_gemm_pingpong",
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
    print("  (Compare per-K-tile execution time vs nki_gemm_tiled.py to confirm overlap)")
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
    print("  ─────────────────────────────────────")
    print("  To see DMA/compute overlap: load the NTFF trace in Neuron Profiler")
    print("  and compare DMA vs MXU timeline against nki_gemm_tiled.py trace.")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
