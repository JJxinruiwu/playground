# GEMM Optimization & Profiling — Trn2 NeuronCores

This document tracks the optimization techniques implemented, performance results,
and profiling methodology for NKIPy/NKI GEMM kernels on AWS Trn2.

---

## 1. Optimization Summary

### 1.1 Precision Reduction: FP16 → FP8

**Script:** `src/simple_nkipy_kernel_fp16.py` → `src/simple_nkipy_kernel_fp8.py`
**Logs:** `log/log_fp16`, `log/log_fp8`

Switching from FP16 to FP8 (float8_e5m2) halves memory traffic per element
and unlocks higher MXU throughput on Trn2.

| Size | FP16 TFLOPS | FP8 TFLOPS | Speedup |
|------|-------------|------------|---------|
| 1024×1024 | 43.91 | 56.72 | 1.29× |
| 2048×2048 | 68.00 | 124.79 | 1.84× |
| 4096×4096 | 12.79 | 140.99 | 11.02× |
| 8192×8192 | 4.07 | 137.02 | 33.67× |

FP8 shows dramatic gains at larger sizes where compute dominance increases.

---

### 1.2 Multi-Core Data Parallelism

**Scripts:** `src/simple_nkipy_kernel_multicore_v0.py` through `v5`
**Logs:** `log/log_fp8_multi_core_v0` through `v4`

Load the same compiled NEFF on all 64 NeuronCores with independent data per core.

| Version | Technique | Aggregate TFLOPS (64 cores) | Per-Core TFLOPS |
|---------|-----------|----------------------------|-----------------|
| v0 | Basic threading | 1,403 | 21.92 |
| v1 | Async dispatch + device tracing | 9,026 (device) / 1,806 (host) | 141.03 (device) / 28.22 (host) |
| v2 | Synchronized rounds (barrier) | — | — |
| v3 | NUMA-aware CPU pinning | 9,027 (device) | 141.04 (device) |
| v4 | Kernel loop ×20 (batched) | 9,219 (device) | 144.05 (device) |
| v5 | v2 + v3 + v4 combined | — | — |

**Key finding:** Device-side throughput (measured via NRT SystemTraceSession) is
~141 TFLOPS/core regardless of host-side optimizations. The gap between host-measured
and device-measured throughput is dispatch overhead.

---

### 1.3 Dispatch Overhead Amortization

**Script:** `src/compare_dispatch_overhead.py`
**Log:** `log/compare_dispatch_overhead.log`

Quantified the host dispatch overhead and its reduction via kernel-loop batching.

| Metric | v1 (1 GEMM/dispatch) | v4 (20 GEMMs/dispatch) |
|--------|----------------------|------------------------|
| Host time/GEMM | 3.580 ms | 0.850 ms |
| Device time/GEMM | 0.975 ms | 0.955 ms |
| Dispatch overhead/GEMM | 2.605 ms (72.8%) | ~0 ms (fully amortized) |
| Device utilization | 15.0% | ~100% |
| Effective throughput | 2,457 TFLOPS | 10,349 TFLOPS |

**Host-side speedup:** 4.21× from batching. Dispatch overhead drops from 72.8%
of host time to negligible.

---

### 1.4 NUMA-Aware CPU Pinning

**Script:** `src/simple_nkipy_kernel_multicore_v3.py`
**Log:** `log/log_fp8_multi_core_v3`

Maps CPU threads to NUMA-local CPUs based on NeuronCore location:
- Cores 16–47 → NUMA node 0
- Cores 0–15, 48–63 → NUMA node 1

Reduces host↔device scheduling latency. Device-side throughput: 9,027 TFLOPS
aggregate (141.04 TFLOPS/core), comparable to v1 async — confirms the bottleneck
is dispatch overhead, not CPU scheduling.

---

### 1.5 NKI Tiled GEMM Kernels

**Scripts:** `src/nki_gemm_tiled.py`, `src/nki_gemm_pingpong.py`
**Logs:** `log/log_nki_gemm_tiled*`, `log/log_nki_gemm_pingpong`, `log/log_nki_gemm_psum_v3b`

Hand-written NKI kernels computing C = A^T @ B with explicit tiling and DMA control.

| Variant | Technique | TFLOPS | Time (ms) | Speedup vs v1 |
|---------|-----------|--------|-----------|---------------|
| v1 (tiled) | Basic SBUF tiling (M=128, K=128, N=512) | 20.95 | 6.560 | 1.00× |
| v2 (tiled) | affine_range + PSUM accumulation | 20.95 | 6.559 | 1.00× |
| v3 (tiled, BLOCK_K=4) | Blocked K-reuse (151 MB DMA vs 671 MB) | 27.17 | 5.058 | 1.30× |
| v3 (tiled, BLOCK_K=8) | Blocked K-reuse (84 MB DMA vs 671 MB) | 27.09 | 5.074 | 1.29× |
| Ping-pong | DMA/matmul overlap via PSUM accum | 10.71 | 12.837 | 0.51× |
| PSUM v3b | Failed — compilation error | — | — | — |

**Key findings:**
- BLOCK_K reuse reduces DMA from 671 MB to 84–151 MB (4.4–7× reduction), yielding
  1.30× speedup. BLOCK_K=4 and BLOCK_K=8 perform nearly identically — SBUF capacity
  is the limiting factor, not DMA volume.
- Ping-pong kernel is currently **slower** (2× regression) — likely due to
  suboptimal overlap scheduling or excessive synchronization. Needs NTFF trace
  investigation.
- All NKI tiled kernels (21–27 TFLOPS) significantly underperform the NKIPy
  simple kernel (141 TFLOPS), indicating the compiler's auto-tiling is superior
  to the current hand-written tiling strategy. Further NKI optimization needed.

---

### 1.6 Tensor Parallelism

**Scripts:** `src/tensor_parallel_gemm.py`, `src/tensor_parallel_sweep.py`
**Logs:** `log/tensor_parallel_gemm.log`, `log/tensor_parallel_sweep.log`

Shard computation across cores with collective communication (all_gather, all_reduce).

#### TP Strategies (4096×4096, TP=4)

| Strategy | Host Latency | Device Latency | Throughput | Speedup vs 1-core | Efficiency |
|----------|-------------|----------------|------------|-------------------|------------|
| Data Parallel | 0.841 ms | 0.974 ms | 653.58 TFLOPS | — | — |
| Column-Parallel (all_gather) | 0.508 ms | 0.560 ms | 270.56 TFLOPS | 1.66× | 41.4% |
| Row-Parallel (all_reduce) | 0.421 ms | 0.468 ms | 326.63 TFLOPS | 2.00× | 50.0% |

Row-parallel outperforms column-parallel due to lower communication volume
(all_reduce sends partial sums vs all_gather sending full shards).

#### Matrix Size Scaling (TP=4)

| Size | Speedup vs 1-core | TP Efficiency |
|------|-------------------|---------------|
| 2048×2048 | 0.44× | 10.9% |
| 4096×4096 | 2.29× | 57.3% |
| 8192×8192 | 3.60× | 90.1% |
| 16384×16384 | 5.02× | 125.6% |

TP efficiency improves dramatically with matrix size — collective overhead is
amortized by larger compute. At 16384×16384, super-linear speedup suggests
cache/memory effects favoring the smaller per-core shards.

#### TP Degree Scaling (4096×4096)

| TP Degree | Speedup | Efficiency | TFLOPS |
|-----------|---------|------------|--------|
| 2 | 1.18× | 59.2% | 173.16 |
| 4 | 1.65× | 41.3% | 241.72 |
| 8 | 0.99× | 12.4% | 144.85 |
| 16 | 0.38× | 2.4% | 55.31 |
| 32 | 0.17× | 0.5% | 24.69 |

**Key finding:** For 4096×4096, TP is only beneficial up to TP=4. Beyond that,
collective overhead dominates. Larger matrices are needed to justify higher TP degrees.

---

### 1.7 End-to-End Time Breakdown

**Script:** `src/breakdown_overhead.py`

Dedicated experiment that measures each phase of GEMM execution independently
to identify where overhead lives. Uses `ASYNC_INFLIGHT=1` so `model()` blocks,
combined with NRT `SystemTraceSession` to split host time into dispatch vs compute.

**Phases measured:**
1. **Compilation** — compile NKIPy kernel → NEFF
2. **NEFF Load** — load NEFF onto NeuronCore(s)
3. **H2D Transfer** — `numpy → SpikeTensor` (with effective bandwidth)
4. **Dispatch + Execute** — `model()` call, split via device trace into:
   - Device execution (on-chip compute)
   - Dispatch overhead (host → device round-trip)
5. **D2H Transfer** — `SpikeTensor.numpy()` back to host

**Scenarios compared:**
| Scenario | Cores | GEMMs/dispatch | Purpose |
|----------|-------|----------------|---------|
| A: 1core×1GEMM | 1 | 1 | Baseline single-core |
| B: 64core×1GEMM | 64 | 1 | Multi-core overhead amplification |
| C: 64core×20GEMM | 64 | 20 | Batched dispatch amortization |

Outputs a stacked breakdown with visual bars and bottleneck identification per scenario.

```bash
python src/breakdown_overhead.py
```

---

## 2. Profiling Methodology

### 2.1 Phase Decomposition

End-to-end time of one `model(inputs, outputs)` call:

```
H2D transfer → kernel dispatch overhead → on-device execution → D2H transfer
```

- **H2D:** `perf_counter` around `SpikeTensor.from_numpy()` calls
- **Dispatch overhead:** measured by subtracting device trace time from host
  wall time (requires `ASYNC_INFLIGHT=1` for accurate host timing)
- **On-device execution:** NRT `SystemTraceSession` (`nc_exec_running` events)
  or `save_trace=True` for NTFF timeline
- **D2H:** `perf_counter` around `.numpy()` call
- **Full breakdown:** `src/breakdown_overhead.py` measures all phases in one run

### 2.2 Device-Side Tracing

Used in v1/v3/v4 for accurate per-core timing independent of host clock:

```python
from spike.lib.nrt.system_trace_session import SystemTraceSession

trace = SystemTraceSession()
trace.start()
# ... run kernel ...
trace.stop()
events = trace.get_events()
# Filter for nc_exec_running events per core
```

### 2.3 Roofline Analysis

For `C = A @ B` with A(M,K), B(K,N) in FP8:

```
FLOPs = 2 × M × N × K
Bytes = (M×K + K×N + M×N) × 1 byte
Arithmetic Intensity = FLOPs / Bytes
```

For 4096×4096 FP8: AI = 2730 FLOP/byte, well above the Trn2 ridge point
(~711 FLOP/byte), confirming this GEMM is **compute-bound**.

---

## 3. Measurement Checklist

| Measurement | Script | Method | Key Metric | Status |
|---|---|---|---|---|
| FP16 baseline | `src/simple_nkipy_kernel_fp16.py` | wall clock | TFLOPS per size | Done |
| FP8 baseline | `src/simple_nkipy_kernel_fp8.py` | wall clock | TFLOPS per size | Done |
| Multi-core scaling | `src/simple_nkipy_kernel_multicore_v0.py` | wall clock | aggregate TFLOPS vs core count | Done |
| Device-side timing | `src/simple_nkipy_kernel_multicore_v1.py` | NRT trace | per-core device TFLOPS | Done |
| Synchronized rounds | `src/simple_nkipy_kernel_multicore_v2.py` | barriers | per-round variance | Done |
| NUMA pinning | `src/simple_nkipy_kernel_multicore_v3.py` | NRT trace + affinity | NUMA0 vs NUMA1 | Done |
| Kernel loop batching | `src/simple_nkipy_kernel_multicore_v4.py` | NRT trace | ms/GEMM vs batch size | Done |
| Combined (v2+v3+v4) | `src/simple_nkipy_kernel_multicore_v5.py` | all above | peak aggregate TFLOPS | Done |
| Dispatch overhead | `src/compare_dispatch_overhead.py` | host−device diff | overhead ms, utilization % | Done |
| E2E time breakdown | `src/breakdown_overhead.py` | all phases | per-phase ms, bottleneck ID | Not yet run |
| NKI tiled GEMM | `src/nki_gemm_tiled.py` | wall clock + trace | TFLOPS, DMA volume | Done |
| NKI ping-pong | `src/nki_gemm_pingpong.py` | wall clock + trace | DMA/MXU overlap | Done |
| Tensor parallelism | `src/tensor_parallel_gemm.py` | host + device | TP efficiency % | Done |
| TP sweep | `src/tensor_parallel_sweep.py` | subprocess | efficiency vs size/degree | Done |
| NTFF trace analysis | `src/nki_gemm_tiled.py` | `save_trace=True` | DMA/MXU timeline | Available |
| NRT queue depth sweep | `src/simple_nkipy_kernel_multicore_v1.py` | env var sweep | TFLOPS vs queue depth | Not yet run |

---

## 4. Open Items & Next Steps

### 4.1 NKI Kernel Performance Gap

NKI hand-written kernels achieve 21–27 TFLOPS vs 141 TFLOPS for the NKIPy auto-compiled
kernel. Potential causes to investigate:

- **Suboptimal tile sizes:** Current M=128, K=128, N=512 may not match MXU pipeline
  depth. Try larger N_TILE or different M/K ratios.
- **Missing DMA/compute overlap:** The tiled kernel issues DMA and matmul sequentially.
  The ping-pong attempt regressed — revisit with NTFF trace to identify stalls.
- **PSUM accumulation bugs:** v3b failed to compile. Fix the `partial` reference
  error and retry.
- **Compiler advantages:** The NKIPy compiler may use optimizations (instruction
  scheduling, prefetching) not yet replicated in hand-written NKI.

### 4.2 Tensor Parallelism at Scale

- TP efficiency degrades sharply beyond TP=4 for 4096×4096. Test with larger
  matrices (8192+) at higher TP degrees to find the practical sweet spot.
- Profile collective communication overhead separately to understand the
  all_reduce vs all_gather cost breakdown.

### 4.3 Remaining Profiling

- **NRT queue depth sweep:** Run v1 with `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`
  set to {1, 8, 16, 32, 63} to find optimal queue depth.
- **NTFF trace for ping-pong kernel:** Identify why DMA/compute overlap is not
  yielding expected speedup.
- **Per-core variance analysis:** Use v5 per-round timestamps to identify
  straggler cores and NUMA effects.

---

## 5. Quick-Start Commands

```bash
# FP8 single-core baseline
python src/simple_nkipy_kernel_fp8.py

# 64-core data parallel with device tracing
python src/simple_nkipy_kernel_multicore_v1.py

# Combined optimizations (NUMA + barriers + kernel loop)
python src/simple_nkipy_kernel_multicore_v5.py

# End-to-end time breakdown (H2D / dispatch / device / D2H)
python src/breakdown_overhead.py

# Dispatch overhead comparison
python src/compare_dispatch_overhead.py

# NKI tiled GEMM with NTFF trace
python src/nki_gemm_tiled.py

# Tensor parallel benchmark
python src/tensor_parallel_gemm.py

# Tensor parallel sweep (size + TP degree)
python src/tensor_parallel_sweep.py

# NRT queue depth sweep
for qd in 1 8 16 32 63; do
    NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=$qd \
        python src/simple_nkipy_kernel_multicore_v1.py 2>&1 \
        | grep "Aggregate throughput"
done
```

---

## 6. File Organization

```
src/                              # All Python source files
├── simple_nkipy_kernel_orig.py   # Original FP16 baseline
├── simple_nkipy_kernel_fp16.py   # FP16 with metrics
├── simple_nkipy_kernel_fp8.py    # FP8 precision reduction
├── simple_nkipy_kernel_multicore_v0.py  # Basic 64-core threading
├── simple_nkipy_kernel_multicore_v1.py  # Async dispatch + device tracing
├── simple_nkipy_kernel_multicore_v2.py  # Synchronized rounds
├── simple_nkipy_kernel_multicore_v3.py  # NUMA-aware pinning
├── simple_nkipy_kernel_multicore_v4.py  # Kernel loop batching (×20)
├── simple_nkipy_kernel_multicore_v5.py  # Combined v2+v3+v4
├── simple_nki_kernel.py          # NKI tensor add demo
├── nki_gemm_tiled.py             # NKI tiled GEMM (blocked)
├── nki_gemm_pingpong.py          # NKI ping-pong PSUM accumulation
├── breakdown_overhead.py          # E2E time breakdown (H2D/dispatch/device/D2H)
├── compare_dispatch_overhead.py  # v1 vs v4 overhead analysis
├── tensor_parallel_gemm.py       # TP strategies benchmark
└── tensor_parallel_sweep.py      # TP size/degree sweep

log/                              # All benchmark logs
├── log_fp16                      # FP16 results
├── log_fp8                       # FP8 results
├── log_fp8_no_transpose          # FP8 without pre-transpose
├── log_fp8_multi_core_v0         # Multi-core v0
├── log_fp8_multi_core_v1         # Multi-core v1 (host timing)
├── log_fp8_multi_core_v1_async   # Multi-core v1 (device timing)
├── log_fp8_multi_core_v3         # Multi-core v3 (NUMA)
├── log_fp8_multi_core_v4         # Multi-core v4 (kernel loop)
├── log_nki_gemm_tiled            # NKI tiled v1
├── log_nki_gemm_tiled_v2         # NKI tiled v2
├── log_nki_gemm_tiled_v3         # NKI tiled v3 (BLOCK_K=4)
├── log_nki_gemm_tiled_v3_bk8     # NKI tiled v3 (BLOCK_K=8)
├── log_nki_gemm_pingpong         # NKI ping-pong
├── log_nki_gemm_psum_v3b         # NKI PSUM v3b (failed)
├── compare_dispatch_overhead.log # Dispatch overhead analysis
├── tensor_parallel_gemm.log      # TP strategies
├── tensor_parallel_sweep.log     # TP sweep
└── v4                            # Multi-core v4 (alt log)
```
