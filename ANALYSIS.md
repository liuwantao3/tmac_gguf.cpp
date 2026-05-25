# Performance Analysis: Metal GPU Acceleration for TMAC GGUF Inference

## Model: Qwen2-0.5B (151936 vocab, 896 hidden, 24 layers)
## Hardware: Apple M1 Pro (16 GPU cores, 200 GB/s unified bandwidth)

---

## 1. Performance Journey Summary

| Date | Change | Process Prompt (1 tok) | Gen Throughput | Per-Tok Gen | Notes |
|------|--------|----------------------|----------------|-------------|-------|
| Baseline | CPU (-O3) | - | ~2.94s/tok | - | Reference baseline |
| Step 1 | Scalar Metal kernels (naive) | ~500ms | - | - | One-thread-scan-all-rows per quant matmul |
| Step 2 | SIMD kernels (32-thread tg, nr0=2) for Q4_K/Q5_0/Q6_K | ~295ms | ~5 t/s | ~200ms | Batch-size-1 dispatch per matmul |
| Step 3 | + Command buffer batching (QKV/gate+up grouped) | ~295ms | ~5 t/s | ~200ms | Reduced commits from 7→4 per layer |
| Step 4 | + Q8_0 SIMD kernel + buffer caching + params pool | ~295ms | ~11 t/s | ~100ms | Last quant type ported; overhead reduced |
| Step 5 | + 256-thread threadgroups (8×32, 16 rows/TG) | ~79ms | ~17 t/s | ~60ms | 40% improvement from threadgroup sizing |

**Current total time**: ~265ms for 1 prompt token + 3 generation tokens (warm).

---

## 2. The Real Bottleneck: Overhead Analysis

### Current profile breakdown (forward pass, 3 gen tokens)

| Category | Time | Share | Details |
|----------|------|-------|---------|
| Quantized matmuls (GPU) | ~10ms | ~7% | 6 matmuls × 24 layers × 3 tokens = 432 dispatches |
| CPU ops (rope, attn, silu, norms) | ~5ms | ~3% | Element-wise, trivially parallel |
| **Sync + dispatch overhead** | **~136ms** | **~90%** | 96 commit+wait cycles, encoder creation, buffer binding |
| **Total forward_all_layers** | **~151ms** | **100%** | |

Each sync point (commit + waitUntilCompleted) costs ~1.4ms of GPU idle + CPU overhead. 4 sync points per layer × 24 layers = 96 sync points per forward pass. **This is the single largest performance problem — 90% of time spent not computing.**

### Bandwidth utilization

- Model weight size: 374 MB (all quant tensors)
- M1 Pro peak bandwidth: 200 GB/s
- Theoretical minimum to read all weights: 374 MB / 200 GB/s = **1.9ms**
- Actual time per token: ~**60ms**
- Bandwidth utilization: **~3%**

---

## 3. Cross-Reference: llama.cpp Metal Backend

Source: `/Users/arctic/llama.cpp/ggml/src/ggml-metal/` (10,699 lines MSL, 4,622 lines C++ ops, 739 lines ObjC context)

### 3.1 Key Architectural Differences

| Aspect | Our Implementation | llama.cpp | Impact |
|--------|-------------------|-----------|--------|
| **RoPE, attention, SILU** | On CPU (creates sync points) | On GPU (Metal kernels) | **Critical**: forcing 4 sync points per layer vs 0 |
| **Command buffers / forward pass** | 96 (one per batch, commit+wait each) | 2-3 (split by node index, async encoding) | **Critical**: 96× GPU drain vs 2-3× |
| **Encoding** | Synchronous, single-threaded | Parallel via GCD (dispatch_apply) | Reduces CPU encoding latency |
| **Threadgroup size (mat-vec)** | 256 (8 simdgroups) | 64 (2 simdgroups) uniformly | Excessive SG count hurts occupancy |
| **Cross-SG reduction** | simd_sum only | simd_sum + shmem barrier + simd_sum | More threads idle in large TG |
| **Elements/thread/block (Q5_0)** | 1 | 16 (4× more work per thread) | Much lower compute-to-overhead ratio |
| **Mat-mul path (batch>8)** | N/A (always mat-vec) | Shared-memory simdgroup mat-mul (128 tg) | Activates for >8 batch; irrelevant for gen |
| **Attention** | CPU dot product + softmax | GPU tiled flash attention (4-8 SG, 128-256 tg) | Eliminates sync; scales with KV cache |
| **Kernel specialization** | Runtime branching on params | Function constants (compile-time) | Marginal; runtime branches cheap on GPU |

### 3.2 llama.cpp Kernel Organization (ggml-metal.metal)

#### Quantized mat-vec (`mul_vec_q_n_f32_impl`)
- `ggml-metal.metal:3388`
- Template: `<block_q_type, NR0, args_t>` — NR0 = rows per simdgroup
- Uses 2 simdgroups (64 threads) uniformly across all quant types
- Each simdgroup handles NR0 rows (varies by type: Q5_0=4, Q8_0=2, Q4_K=2, etc.)
- Each thread processes NQ elements per block (varies by type: Q8_0=8, Q4_0=16)
- Cross-simdgroup reduction via `helper_mv_reduce_and_write` (shmem+barrier+simd_sum)

#### Rope (`kernel_rope_norm/neox/multi`)
- `ggml-metal.metal:4363-4633`
- `nth = min(1024, ne00)` threads per threadgroup
- Each thread processes 2 elements per iteration (one even-odd pair)
- Strided loop: `for (i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x)`

#### Unary ops (SILU) (`kernel_unary_impl`)
- `ggml-metal.metal:1017-1175`
- SINGLE kernel for all unary ops (SILU, GELU, RELU, TANH, SIGMOID, etc.)
- Dispatched via function constant `FC_unary_op`: switch-free, compile-time specialized
- SILU: `dst[i0] = x / (1 + exp(-x))` — standard sigmoid(x)*x

#### Flash attention (`kernel_flash_attn_ext_impl`)
- `ggml-metal.metal:5853-6490+`
- Tiled online softmax flash attention
- `Q=8` queries per simdgroup, `C=64` KV items per simdgroup
- `NSG` = 4 or 8 simdgroups (selected by head dimension: ≥512 → 8, else 4)
- Threadgroup memory: query data, output accumulator, score buffer, K/V scratch, mask
- simdgroup_float8x8/half8x8 matrix multiply for Q*K^T
- Iterates over KV cache in blocks of C, loading K tiles into shared memory

#### Command buffer pipeline (`ggml-metal-context.m:438-721`)
1. Main thread creates cmd_buf[n_cb], encodes first ~10% of graph nodes, commits immediately
2. Creates n_cb additional cmd_bufs (default 1-2, max 16), enqueues them
3. Dispatches parallel encoding via `dispatch_apply(n_cb, queue, encode_async)`
4. Each thread encodes its node range, commits when done
5. No waitUntilCompleted (pipelines asynchronously)
6. Uses `commandBufferWithUnretainedReferences` for lower overhead

### 3.3 Threadgroup Constants (ggml-metal-impl.h)

```
N_SIMDWIDTH = 32  (threads per simdgroup)
N_R0_Q5_0 = 4     (rows per simdgroup, Q5_0)
N_R0_Q8_0 = 2     (rows per simdgroup, Q8_0)
N_R0_Q4_K = 2     (rows per simdgroup, Q4_K)
N_SG = 2          (simdgroups per threadgroup) — uniform across ALL quant types
```

**Key insight**: llama.cpp uses 64 threads per threadgroup (2×32), NOT 256. Our 256-thread change actually moves AWAY from llama.cpp's proven configuration.

---

## 4. What Worked and What Didn't — Verified by Measurement

### ✅ What Worked

#### 1. SIMD collaborative kernels (replacing scalar)
- **Result**: 500ms → 295ms prefill (1.7×)
- **Why**: 32 threads share block work instead of 1 thread scanning all cols

#### 2. Command buffer batching (grouping independent matmuls)
- **Result**: Reduces sync from 7 to 4 per layer → 24 fewer commits/forward
- **Why**: Fewer GPU drain cycles, less PCI-e/encoder overhead

#### 3. Weight buffer caching + params pool
- **Result**: Eliminates per-call allocation overhead
- **Why**: newBufferWithBytesNoCopy called once per tensor, not per dispatch

#### 4. Q8_0 SIMD kernel
- **Result**: Enabled full GPU path for all quant types (was last remaining)
- **Why**: Uses same 32-lane collaborative pattern as other quant types

#### 5. 256-thread threadgroups (8×32, 16 rows/TG)
- **Result**: 100ms → 60ms per gen token (~40%)
- **Why**: Matches larger GPU work units for dispatch → fewer TGs for large row counts (e.g., 151936-row Q8_0 logits matmul gets better occupancy)
- **Limitation**: llama.cpp uses 64-thread (2 SG), not 256-thread — so 256 may be excessive. Our improvement may be from the specific 2→8 SG scaling for large matrices, not from hitting an optimal point.

### ❌ What Didn't Work (Reverted)

#### 1. sumy precomputation in Q4_K/Q6_K
- **Result**: No measurable benefit, reverted
- **Why**: Kernels are memory-bandwidth bound, not compute bound. Adding computation doesn't help when the bottleneck is reading weight data.

#### 2. Moving to fewer than 32 threads per SIMD group
- **Result**: Degraded performance
- **Why**: 32 is the SIMD width on Apple Silicon; any smaller underutilizes the SIMD unit.

### 🔬 What Wasn't Worth the Complexity

#### 1. Dedicated per-type dispatch functions (vs shared dispatch)
- We created `matmul_q8_0/matmul_q5_0/matmul_q4_k/matmul_q6_k` wrappers
- Minimal overhead difference vs a single shared function
- Kept for clarity/maintenance, not performance

---

## 5. Root Cause Analysis: Why 90% Overhead?

### The sync chain (per layer)

```
metal_batch_begin()
  Q matmul dispatch    [GPU, ~26μs]
  K matmul dispatch    [GPU, ~12μs]
  V matmul dispatch    [GPU, ~12μs]
metal_batch_end()      [COMMIT + WAIT ~1.4ms] ← GPU idle, CPU blocks
rope/attention (CPU)   [CPU, ~11μs]
metal_batch_begin()
  attn_out dispatch    [GPU, ~24μs]
metal_batch_end()      [COMMIT + WAIT ~1.4ms] ← GPU idle, CPU blocks
silu (CPU)             [CPU, ~11μs]
metal_batch_begin()
  gate dispatch        [GPU, ~22μs]
  up dispatch          [GPU, ~13μs]
metal_batch_end()      [COMMIT + WAIT ~1.4ms] ← GPU idle, CPU blocks
metal_batch_begin()
  ffn_down dispatch    [GPU, ~24μs]
metal_batch_end()      [COMMIT + WAIT ~1.4ms] ← GPU idle, CPU blocks
```

**Each commit+wait cycle:**
1. `endEncoding` — marks encoder done (~5μs)
2. `commit` — submits command buffer to GPU (~10μs)
3. `waitUntilCompleted` — blocks CPU until GPU finishes all work in buffer (~1.4ms GPU idle!)

**Why so expensive?** GPU has startup latency per command buffer (scheduling, cache warm, power-up). With only 50μs of actual work per batch, the GPU spends 96% of the time on startup overhead.

### Theoretical minimum per token

| Component | Time | Assumptions |
|-----------|------|-------------|
| Read 374 MB weights at 200 GB/s | 1.9ms | Bandwidth-saturating kernel |
| Attention (softmax Q·K^T·V, CPU) | ~0.3ms | For short context |
| Element-wise ops (rope, silu, norms) | ~0.1ms | 896 hidden dim, trivial |
| GPU dispatch overhead (amortized) | ~0.5ms | 1 command buffer, ~50 dispatches |
| **Theoretical minimum** | **~2.8ms** | **≈ 357 t/s** |

Current gap from theoretical: **60ms / 2.8ms = 21×**

---

## 6. Recommendations: Next Steps (Ranked by Impact)

### Priority 1: Move CPU ops to GPU (RoPE, attention, SILU) → Single CB forward pass
**Rationale**: Eliminates 3 sync points per layer (72/96 = 75% of overhead).
- Write 3 simple Metal kernels (~100 lines total)
- Encode ALL operations for ALL 24 layers into ONE command buffer
- Zero commit+wait cycles during forward pass
- Estimated gain: **60ms → ~10ms per token** (6×)

### Priority 2: Reduce threadgroup size from 256 to 64 (2 SG)
**Rationale**: llama.cpp uniformly uses 64 threads (2 SG) for mat-vec. 256 threads (8 SG) is excessive for small row counts (e.g., V-proj: 128 rows → 1 TG with 256 threads, 128 idle). Also reduces register pressure.
- Estimated gain: marginal alone, but compounds with Priority 1

### Priority 3: Use function constants for kernel specialization
**Rationale**: Eliminates runtime branching for batch dimensions (ne12, ne13) and simdgroup count (nsg). llama.cpp compiles a unique pipeline for each combination.
- Estimated gain: marginal (runtime branches cheap on GPU ALU)

### Priority 4: Multi-CB async encoding (GCD dispatch_apply)
**Rationale**: Overlaps CPU encoding with GPU execution. Important only after sync points are eliminated (Priority 1).
- Estimated gain: 5-10% after Priority 1 is done

### Priority 5: Flash attention on GPU (full tiled)
**Rationale**: Our current attention is basic Q·K^T·V with softmax. As KV cache grows, this will become a bottleneck. Flash attention scales sub-linearly.
- Estimated gain: Important for long context, irrelevant for short gen

---

## 7. llama.cpp Kernel Parameters Reference

### Mat-Vec (kernel_mul_mv)

| Quant | N_R0 | N_SG | NW | NQ | Total Threads | Total Rows/TG |
|-------|:----:|:----:|:--:|:--:|:-------------:|:-------------:|
| Q1_0 | 8 | 2 | 32 | 8 | 64 | 16 |
| Q4_0 | 4 | 2 | 32 | 16 | 64 | 8 |
| Q4_1 | 4 | 2 | 32 | 16 | 64 | 8 |
| Q5_0 | 4 | 2 | 32 | 4 | 64 | 8 |
| Q5_1 | 4 | 2 | 32 | 4 | 64 | 8 |
| Q8_0 | 2 | 4 | 32 | 8 | 128 | 8 |
| Q2_K | 4 | 2 | 32 | - | 64 | 8 |
| Q3_K | 2 | 2 | 32 | - | 64 | 4 |
| Q4_K | 2 | 2 | 32 | - | 64 | 4 |
| Q6_K | 2 | 2 | 32 | - | 64 | 4 |

### Mat-Mat (kernel_mul_mm) — tensor ops, A14+

| Parameter | Value |
|-----------|-------|
| N_MM_BLOCK_X | 4 |
| N_MM_BLOCK_Y | 2 |
| N_MM_SIMD_GROUP_X | 2 |
| N_MM_SIMD_GROUP_Y | 2 |
| Total simdgroups | 4 |
| Total threads | 128 |
| N_MM_NK_TOTAL | 32 (K tile) |
| Output tile NRA×NRB | 64×128 |

### Flash Attention (kernel_flash_attn_ext)

| Parameter | Small head (<512) | Large head (>=512) |
|-----------|:-----------------:|:------------------:|
| NSG | 4 | 8 |
| Threads | 128 | 256 |
| Queries/SG (Q) | 8 | 8 |
| KV items/SG (C) | 64 | 64 |
| K accumulator type | half4x4 simdgroup | half4x4 simdgroup |

---

*Generated from direct analysis of tmac_gguf.cpp and llama.cpp source at `/Users/arctic/llama.cpp/ggml/src/ggml-metal/`.*
