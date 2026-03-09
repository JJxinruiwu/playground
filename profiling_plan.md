# GEMM Profiling Plan — Fine-Grained Timing Breakdown

This document describes how to decompose the end-to-end wall time of the
NKIPy/NKI GEMM benchmarks into actionable phases, and how to use the
resulting data to guide further optimization.

---

## 1. Phase Decomposition

The end-to-end time of one `model(inputs, outputs)` call contains four phases:

```
H2D transfer → kernel dispatch overhead → on-device execution → D2H transfer
```

### 1.1 H2D Transfer (`SpikeTensor.from_numpy`)

**What to time:** allocation of each SpikeTensor from a NumPy array.

```python
from time import perf_counter

tensors_np = [A_np, B_np]
t0 = perf_counter()
for arr, name in zip(tensors_np, ["A", "B"]):
    SpikeTensor.from_numpy(arr, name, core_id=0)
h2d_ms = (perf_counter() - t0) * 1000
print(f"H2D transfer: {h2d_ms:.3f} ms")
```

**Expected range:** scales with tensor size (fp8 4096×4096 ≈ 16 MB per tensor).
Use this to compute effective PCIe bandwidth:
```
BW (GB/s) = (num_tensors × M × K × bytes_per_elem) / (h2d_ms × 1e-3) / 1e9
```

---

### 1.2 Kernel Dispatch Overhead

**Method:** run a trivially small kernel (K=1, M=1, N=1) so that on-device compute
is negligible, then the measured wall time ≈ dispatch overhead alone.

```python
# Tiny kernel to isolate dispatch latency
def tiny_kernel(A, B): return A @ B

A_tiny = np.ones((1, 1), dtype=ml_dtypes.float8_e5m2)
B_tiny = np.ones((1, 1), dtype=ml_dtypes.float8_e5m2)
tiny_model = DeviceKernel.compile_and_load(tiny_kernel, A_tiny, B_tiny, ...)

# Measure dispatch round-trip
REPS = 100
t0 = perf_counter()
for _ in range(REPS):
    tiny_model(inputs={...}, outputs={...})
dispatch_ms = (perf_counter() - t0) * 1000 / REPS
print(f"Dispatch overhead: {dispatch_ms:.3f} ms/call")
```

**Expected range:** 0.2–2 ms per call depending on NRT queue depth and CPU load.
This is the irreducible per-call cost that batching (KERNEL_LOOP) amortizes.

---

### 1.3 On-Device Execution (NTFF Trace)

**Method:** use `save_trace=True` to generate a Neuron Trace Format File (NTFF),
then open it in Neuron Profiler for instruction-level timeline analysis.

```python
# Pattern from simple_nki_kernel.py lines 100–106
kernel(
    inputs={"A": device_A, "B": device_B},
    outputs={"output0": device_out},
    save_trace=True,   # saves .ntff alongside the .neff
)
# Profile is written next to: kernel.neff_path
```

**In Neuron Profiler, look for:**
- `DMA` lanes: time spent on HBM→SBUF transfers (A tiles, B tiles)
- `MXU` / `MATMUL` lanes: time spent in nc_matmul instructions
- Gaps between DMA and MXU: synchronization stalls
- For ping-pong (`nki_gemm_pingpong.py`): DMA and MXU should overlap visually

**Key metric:** `on_device_ms = trace_end_time - trace_start_time`

---

### 1.4 D2H Transfer (`.numpy()`)

**What to time:** copying output SpikeTensor back to host NumPy array.

```python
t0 = perf_counter()
result = output_spike_tensor.numpy()
d2h_ms = (perf_counter() - t0) * 1000
print(f"D2H transfer: {d2h_ms:.3f} ms")
```

---

### 1.5 Full Decomposition Example

```python
# Full decomposition for one round
t0 = perf_counter()
inputs = {
    "A": SpikeTensor.from_numpy(A_np, "A", core_id=0),
    "B": SpikeTensor.from_numpy(B_np, "B", core_id=0),
}
h2d_ms = (perf_counter() - t0) * 1000

t1 = perf_counter()
model(inputs=inputs, outputs=outputs)
dispatch_plus_exec_ms = (perf_counter() - t1) * 1000
# subtract dispatch_ms (measured separately above) to isolate exec
on_device_ms = dispatch_plus_exec_ms - dispatch_ms

t2 = perf_counter()
_ = outputs["output0"].numpy()
d2h_ms = (perf_counter() - t2) * 1000

total_ms = h2d_ms + dispatch_plus_exec_ms + d2h_ms
print(f"H2D: {h2d_ms:.2f} ms | Dispatch+Exec: {dispatch_plus_exec_ms:.2f} ms "
      f"| D2H: {d2h_ms:.2f} ms | Total: {total_ms:.2f} ms")
```

---

## 2. KERNEL_LOOP Sweep — Amortizing Dispatch Overhead

**Goal:** find the KERNEL_LOOP value where per-GEMM cost is dominated by compute
rather than dispatch overhead.

**Setup:** use the v4 structure (`simple_nkipy_kernel_multicore_v4.py`) and sweep
`KERNEL_LOOP` over powers-of-2 and intermediate values:

```
KERNEL_LOOP ∈ {1, 2, 5, 10, 20, 50, 100}
```

For each value:
1. Compile a fresh kernel with that batch size
2. Run benchmark_iterations=10 rounds on a single core
3. Record `mean_ms_per_dispatch` (wall time per `model()` call)
4. Compute `ms_per_gemm = mean_ms_per_dispatch / KERNEL_LOOP`

```python
results = {}  # KERNEL_LOOP → ms_per_gemm

for kl in [1, 2, 5, 10, 20, 50, 100]:
    # Compile with this batch size
    A_proto = np.random.rand(kl, 4096, 4096).astype(ml_dtypes.float8_e5m2)
    B_proto = np.random.rand(kl, 4096, 4096).astype(ml_dtypes.float8_e5m2)
    kernel = DeviceKernel.compile_and_load(
        lambda A, B: A @ B, A_proto, B_proto,
        name=f"kl_{kl}", use_cached_if_exists=True
    )
    # ... run benchmark ...
    results[kl] = mean_ms_per_dispatch / kl

# Plot: x = KERNEL_LOOP, y = ms/GEMM
# The knee of the curve is the recommended KERNEL_LOOP
```

**Interpretation:**
- At small KERNEL_LOOP: `ms/GEMM ≈ dispatch_overhead / KERNEL_LOOP + compute_ms`
  → dispatch-dominated; curve falls steeply
- At large KERNEL_LOOP: `ms/GEMM ≈ compute_ms` → curve flattens
- **Knee point** = where adding more batching gives <5% improvement
  → this is the recommended `KERNEL_LOOP` value

**Expected curve shape:**
```
ms/GEMM
  |  \
  |   \
  |    \___________
  |_________________ KERNEL_LOOP
      knee ^
```

---

## 3. Per-Core Variance Analysis

**Goal:** quantify how much cores diverge within a synchronized round, and
whether NUMA node membership explains the variance.

**Setup:** extend `simple_nkipy_kernel_multicore_v2.py` (or v5) to collect
per-core per-round timestamps.

The v2/v5 `run_on_core` already records `per_round_ms` list per core.
Post-process as follows:

```python
import numpy as np

# core_results[i] = list of per-round ms for core i
# shape: (num_cores, benchmark_iterations)
data = np.array([core_results[i] for i in range(num_cores)])

# Overall stats
mean_per_core = data.mean(axis=1)     # shape: (num_cores,)
std_per_core  = data.std(axis=1)
p95_per_core  = np.percentile(data, 95, axis=1)
p99_per_core  = np.percentile(data, 99, axis=1)

print(f"{'Core':>4}  {'NUMA':>4}  {'mean ms':>8}  {'std ms':>7}  {'p95 ms':>7}  {'p99 ms':>7}")
for i in range(num_cores):
    numa = 0 if 16 <= i <= 47 else 1
    print(f"{i:4d}  {numa:4d}  {mean_per_core[i]:8.3f}  "
          f"{std_per_core[i]:7.3f}  {p95_per_core[i]:7.3f}  {p99_per_core[i]:7.3f}")

# NUMA group comparison
numa0_mask = np.array([16 <= i <= 47 for i in range(num_cores)])
numa1_mask = ~numa0_mask

print("\nNUMA node comparison:")
print(f"  NUMA 0 (cores 16-47): mean={data[numa0_mask].mean():.3f} ms, "
      f"std={data[numa0_mask].std():.3f} ms")
print(f"  NUMA 1 (rest):        mean={data[numa1_mask].mean():.3f} ms, "
      f"std={data[numa1_mask].std():.3f} ms")
```

**What to look for:**
- **High std/p99 on specific cores:** indicates scheduling jitter or NRT queue
  contention on those cores → consider reducing concurrent submission rate
- **NUMA 1 slower than NUMA 0:** confirms NUMA locality benefit of v3 pinning
- **Straggler cores:** ones where `mean_per_core[i] >> median(mean_per_core)`
  → investigate OS noise, IRQ affinity, or thermal throttling

---

## 4. Roofline Analysis

**Goal:** determine whether the GEMM is compute-bound or memory-bandwidth-bound.

### Arithmetic Intensity

For `C = A @ B` with `A (M, K)`, `B (K, N)`, `C (M, N)` in fp8 (1 byte/elem):

```
FLOPs = 2 × M × N × K
Bytes = (M×K + K×N + M×N) × 1    (fp8 = 1 byte)

Arithmetic Intensity = FLOPs / Bytes
```

For M = N = K = 4096, fp8:
```
FLOPs  = 2 × 4096³ ≈ 137.4 GFLOPs
Bytes  = 3 × 4096² × 1 = 50.3 MB
AI     = 137.4e9 / 50.3e6 ≈ 2730 FLOP/byte
```

At **2730 FLOP/byte**, the GEMM is heavily **compute-bound** on any hardware
where peak compute / peak BW > 2730:
- trn2 peak fp8 MXU throughput: ~3840 TFLOPS (per chip)
- trn2 peak HBM bandwidth: ~5.4 TB/s (per chip)
- Ridge point: 3840e12 / 5.4e12 ≈ 711 FLOP/byte

Since AI = 2730 >> 711 = ridge point, 4096×4096 fp8 GEMM is compute-bound.

### Roofline Plot Instructions

```python
import matplotlib.pyplot as plt
import numpy as np

# Hardware limits (trn2 per chip, adjust to actual spec)
peak_compute_tflops = 3840       # fp8 MXU
peak_bw_tbs = 5.4                # TB/s = 5.4e12 B/s
peak_bw_tflops_per_flop_per_byte = peak_bw_tbs * 1e12 / 1e12  # in TFLOPS at 1 FLOP/byte

ai_values = np.logspace(-1, 5, 500)
roof_compute = np.full_like(ai_values, peak_compute_tflops)
roof_bw = ai_values * peak_bw_tbs  # TFLOPS = AI (FLOP/byte) × BW (TB/s)
roofline = np.minimum(roof_compute, roof_bw)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(ai_values, roofline, 'b-', linewidth=2, label='Roofline')
ax.loglog(ai_values, roof_bw, 'b--', alpha=0.4, label=f'BW limit ({peak_bw_tbs} TB/s)')
ax.axhline(peak_compute_tflops, color='b', linestyle=':', alpha=0.4,
           label=f'Compute limit ({peak_compute_tflops} TFLOPS)')

# Plot achieved TFLOPS from benchmark
achieved_ai = 2730   # FLOP/byte for 4096³ fp8
achieved_tflops = ...  # fill in from benchmark results
ax.loglog(achieved_ai, achieved_tflops, 'ro', markersize=12,
          label=f'Achieved: {achieved_tflops:.0f} TFLOPS')

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
ax.set_ylabel('Performance (TFLOPS)')
ax.set_title('Roofline Model — trn2 fp8 GEMM 4096×4096')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('roofline.png', dpi=150)
```

**Interpreting results:**
- If `achieved_tflops` < `peak_compute × 0.5`: suspect long SBUF pipeline stalls
  or suboptimal tiling → check NTFF trace for idle MXU time
- If `achieved_tflops` ≈ `AI × peak_BW`: incorrectly memory-bound → re-check dtype
  (fp8 inputs but fp32 output moves 4× more bytes for output)

---

## 5. NRT Queue Depth Sweep

**Goal:** find the queue depth that maximizes aggregate throughput when all 64
cores submit simultaneously.

**Background:** `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` caps how many kernel
submissions NRT queues per core. At 64 cores × 1 submission/core = 64 inflight;
the default queue capacity is 63, causing the 64th submission to stall.

**Setup:** use v1 structure (independent per-core threads, no synchronization)
for maximum submission pressure.

```python
import subprocess, os

queue_depths = [1, 8, 16, 32, 63]
results = {}

for qd in queue_depths:
    env = os.environ.copy()
    env["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = str(qd)
    # Run v1 benchmark as subprocess to isolate env
    proc = subprocess.run(
        ["python", "simple_nkipy_kernel_multicore_v1.py"],
        capture_output=True, text=True, env=env
    )
    # Parse aggregate TFLOPS from stdout
    for line in proc.stdout.splitlines():
        if "Aggregate throughput" in line:
            tflops = float(line.split()[-1])
            results[qd] = tflops
            break

print("Queue depth → Aggregate TFLOPS:")
for qd, tflops in sorted(results.items()):
    print(f"  {qd:3d}: {tflops:.2f} TFLOPS")
```

**Expected behavior:**
- `qd=1`: severe stalling (only 1 kernel inflight per core), low throughput
- `qd=8`: improves as pipeline deepens
- `qd=63`: near-maximum throughput (all 64 cores can submit ~simultaneously)
- Diminishing returns above `qd=num_cores`

**Decision rule:** set `qd` to the lowest value that achieves ≥95% of peak
throughput (reduces memory pressure in NRT queue).

---

## 6. Summary: Measurement Checklist

| Measurement | Script | Flag/Env var | Key metric |
|---|---|---|---|
| H2D transfer time | any v* | `perf_counter` around `from_numpy` | ms/tensor, GB/s |
| Dispatch overhead | tiny kernel (K=1) | none | ms/call |
| On-device timeline | `nki_gemm_tiled.py` | `save_trace=True` | NTFF trace |
| D2H transfer time | any v* | `perf_counter` around `.numpy()` | ms/tensor |
| KERNEL_LOOP amortization | v4 modified | sweep KERNEL_LOOP | ms/GEMM vs KERNEL_LOOP |
| Per-core variance | v2 / v5 | per-round timestamps | std, p95, p99 per core |
| NUMA comparison | v5 | NUMA pinning on/off | NUMA0 vs NUMA1 latency |
| Roofline position | any v* | achieved vs peak | % of roofline |
| NRT queue depth | v1 | `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS` | TFLOPS vs qd |
| Ping-pong overlap | `nki_gemm_pingpong.py` | `save_trace=True` | DMA/MXU overlap in NTFF |

---

## 7. Quick-Start Commands

```bash
# Phase timing
python simple_nkipy_kernel_multicore_v5.py

# NTFF trace (single core)
python nki_gemm_tiled.py
# → opens with: neuron_profiler view <path_to>.ntff

# KERNEL_LOOP sweep (modify KERNEL_LOOP constant in v4 and rerun)
for kl in 1 2 5 10 20 50 100; do
    KERNEL_LOOP=$kl python simple_nkipy_kernel_multicore_v4.py 2>&1 \
        | grep "Wall time / GEMM"
done

# NRT queue depth sweep
for qd in 1 8 16 32 63; do
    NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=$qd \
        python simple_nkipy_kernel_multicore_v1.py 2>&1 \
        | grep "Aggregate throughput"
done

# Ping-pong vs tiled comparison
python nki_gemm_tiled.py       # baseline: no overlap
python nki_gemm_pingpong.py    # optimized: DMA/compute overlap
```
