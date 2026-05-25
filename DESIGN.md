# Design: Eliminate All 96 Sync Points — Single Command Buffer Forward Pass

## Problem

96 commit+wait cycles per forward pass (4 per layer × 24 layers). Each cycle costs ~1.4ms of GPU drain. The GPU executes ~50μs of actual work, then idles for ~1.4ms waiting for CPU to process intermediate results.

## Root Cause

CPU-in-the-middle operations between GPU matmul batches:

```
GPU: QKV mats → wait → CPU: rope, attn → GPU: attn_out → wait → CPU: rmsnorm → GPU: gate,up → wait → CPU: silu → GPU: ffn_down → wait → CPU: add
```

Each `→ wait →` is a `metal_batch_end()` with commit + waitUntilCompleted.

## Solution

Move ALL CPU-in-the-middle operations to Metal kernels. Encode ALL operations for ALL 24 layers into ONE command buffer. Zero commit+wait during forward pass.

---

## Model Constants (Qwen2-0.5B)

| Constant | Value | Notes |
|----------|-------|-------|
| HIDDEN_DIM | 896 | Hidden state size |
| INTER_DIM | 4864 | FFN intermediate size |
| NUM_HEADS | 14 | Query heads |
| NUM_KV_HEADS | 2 | Key/value heads (GQA) |
| HEAD_DIM | 64 | Dimensions per head |
| K_DIM | 128 | NUM_KV_HEADS × HEAD_DIM |
| V_DIM | 128 | Same as K_DIM |
| MAX_SEQ_LEN | 256 | Max context length |
| Q_PER_KV | 7 | NUM_HEADS / NUM_KV_HEADS |
| rope_base | 10000.0 | Rotary base frequency |

---

## New Metal Kernels

### 1. `kernel_add_bias` — Add bias to matmul output

**Purpose**: Replace CPU bias-add after Q, K, V matmuls.

```
Params: [dim]
Buffers: [0] data (in/out), [1] bias (in)
Threads: dim (one per element)
Body:   data[tid] += bias[tid]
```

### 2. `kernel_rope` — In-place rotary position embedding on Q and K

**Purpose**: Replace `apply_rope(q, k, pos)`. Applied to Q and K separately.

**Q rope**:
```
Params: [n_heads, head_dim, pos, rope_base]
Buffer: [0] data (in/out)  — Q [n_heads × head_dim]
Threads: n_heads × head_dim/2  (one thread per even-odd pair)
Body:
    int head = tid / (head_dim/2);
    int pair = tid % (head_dim/2);
    int idx = head * head_dim + pair * 2;
    float theta = 1.0 / pow(rope_base, (float)(pair*2) / head_dim);
    float angle = (float)pos * theta;
    float cos_a = cos(angle), sin_a = sin(angle);
    float x0 = data[idx], x1 = data[idx+1];
    data[idx]   = x0*cos_a - x1*sin_a;
    data[idx+1] = x0*sin_a + x1*cos_a;
```

**K rope**: Same kernel, but K_DIM=128 = 2 heads, so 2×32=64 threads.

Dispatch: Q rope = 14×32=448 threads, K rope = 2×32=64 threads → use same kernel with n_heads=14 and n_heads=2 respectively.

### 3. `kernel_cache_write` — Write K or V into KV cache slot

**Purpose**: Replace `g_k_cache[layer][pos][i] = k_new[i]` and same for V.

```
Params: [pos, dim]
Buffer: [0] src (in), [1] cache (out)
Threads: dim (one per element)
Body:   cache[pos * dim + tid] = src[tid]
```

### 4. `kernel_attention` — Single-token GQA attention with KV cache

**Purpose**: Replace CPU attention: Q·K_cache^T → softmax → ·V_cache.

This is the most complex kernel. Design:

**Thread decomposition**: 14 threadgroups (one per Q head), 32 threads each (head_dim/2 = 32 pairs). 1 simdgroup per TG → no cross-SG barrier needed.

**Threadgroup memory**: `scores[MAX_SEQ_LEN]` float array (256×4=1KB, fits in 32KB SRAM).

```
Inputs:
  [0] Q       [n_head × head_dim]           — 14×64=896 floats
  [1] K_cache [MAX_SEQ_LEN × n_kv_head × head_dim] — 256×2×64=32K floats
  [2] V_cache [MAX_SEQ_LEN × n_kv_head × head_dim] — same
  [3] output  [n_head × head_dim]           — 14×64=896 floats
  [4] params  [n_head, n_kv_head, head_dim, pos, q_per_kv]

Threadgroup memory:
  threadgroup float scores[MAX_SEQ_LEN];  // 256 × 4 = 1KB

Per-thread group (head h):
    int kv_head = h / q_per_kv;   // GQA: 7 Q heads share 1 KV head
    int d0 = tiisg * 2;            // even dim
    int d1 = tiisg * 2 + 1;        // odd dim
    float my_q0 = Q[h * head_dim + d0];
    float my_q1 = Q[h * head_dim + d1];

    // ── Phase 1: scores[p] = Q[h] · K_cache[p][kv] / sqrt(head_dim) ──
    for (int p = 0; p <= pos; p++) {
        float k0 = K_cache[p * n_kv_head * head_dim + kv_head * head_dim + d0];
        float k1 = K_cache[p * n_kv_head * head_dim + kv_head * head_dim + d1];
        float prod = my_q0 * k0 + my_q1 * k1;
        float score = simd_sum(prod) * (1.0 / sqrt(head_dim));
        if (tiisg == 0) scores[p] = score;
    }

    // ── Phase 2: online softmax ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Find max across all positions (strided)
    float my_max = -INFINITY;
    for (int p = tiisg; p <= pos; p += 32)
        my_max = max(my_max, scores[p]);
    float global_max = simd_max(my_max);
    // Compute exp and sum (strided)
    float my_sum = 0;
    for (int p = tiisg; p <= pos; p += 32) {
        float e = exp(scores[p] - global_max);
        scores[p] = e;
        my_sum += e;
    }
    float total = simd_sum(my_sum);
    // Normalize (strided)
    for (int p = tiisg; p <= pos; p += 32)
        scores[p] /= total;

    // ── Phase 3: weighted sum of V_cache ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float my_r0 = 0, my_r1 = 0;
    for (int p = tiisg; p <= pos; p += 32) {
        float v0 = V_cache[p * n_kv_head * head_dim + kv_head * head_dim + d0];
        float v1 = V_cache[p * n_kv_head * head_dim + kv_head * head_dim + d1];
        my_r0 += scores[p] * v0;
        my_r1 += scores[p] * v1;
    }
    float r0 = simd_sum(my_r0);
    float r1 = simd_sum(my_r1);
    output[h * head_dim + d0] = r0;
    output[h * head_dim + d1] = r1;
```

**Key invariants**:
- `pos` is inclusive (covers current token position 0..pos)
- Score is divided by `sqrt(head_dim)` = 8 before softmax
- Phase 2 normalizes in-place in threadgroup memory
- Phase 3 reads normalized scores from threadgroup memory (after barrier)
- V_cache read in Phase 3 is strided: thread t reads positions t, t+32, t+64, ...
- Each thread accumulates for its 2 dims, simd_sum combines across all 32 threads → full dim sum

### 5. `kernel_residual_add` — hidden += residual

**Purpose**: Replace `hidden[i] = original_hidden[i] + attn_out[i]` and `hidden[i] += ffn_out[i]`.

```
Params: [dim]
Buffers: [0] hidden (in/out), [1] residual (in)
Threads: dim
Body:   hidden[tid] += residual[tid]
```

### 6. `kernel_rmsnorm` — In-place RMS normalization

**Purpose**: Replace `rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t)`.

```
Params: [dim]
Buffers: [0] data (in/out), [1] weight (in)
Threads: 32 (one SG), threadgroup memory for partial sums

Body:
    // Sum of squares (strided)
    float my_sum = 0;
    for (int i = tiisg; i < dim; i += 32)
        my_sum += data[i] * data[i];
    float total = simd_sum(my_sum);
    float rms = sqrt(total / dim + 1e-6f);
    // Normalize (strided)
    for (int i = tiisg; i < dim; i += 32)
        data[i] = (data[i] / rms) * weight[i];
```

### 7. `kernel_silu_x_up` — SiLU activation and element-wise multiply

**Purpose**: Replace `silu(gate, gate, INTER_DIM)` and `gate[i] *= up[i]` fused.

```
Params: [dim]
Buffers: [0] gate (in/out), [1] up (in)
Threads: dim
Body:   gate[tid] = (gate[tid] / (1 + exp(-gate[tid]))) * up[tid]
```

---

## Forward Pass Modifications (tmac_gguf.cpp)

### Modified `forward_layer_with_cache` — All-GPU single command buffer

```cpp
void forward_layer_with_cache_gpu(float* hidden, int layer, int pos) {
    char name[128]; Tensor* t;
    float original_hidden[HIDDEN_DIM];
    memcpy(original_hidden, hidden, HIDDEN_DIM * sizeof(float));

    // Pre-wrap all fixed buffers once
    // (hidden, attn_norm_out, q_vec, k_new, v_new, context, attn_out,
    //  ffn_norm_out, gate, up, ffn_out, k_cache, v_cache)

    metal_batch_begin();

    // ── [STEP 0] attn_norm: hidden → attn_norm_out ──
    // CPU rms_norm (or GPU kernel_rmsnorm)

    // ── [STEP 1] QKV matmuls ──
    matmul(attn_norm → q_vec);  // pipe: Q5_0
    matmul(attn_norm → k_new);  // pipe: Q5_0 or Q8_0
    matmul(attn_norm → v_new);  // pipe: Q5_0 or Q8_0

    // ── [STEP 2] bias adds on Q, K, V ──
    kernel_add_bias(q_vec, bias_q);
    kernel_add_bias(k_new, bias_k);
    kernel_add_bias(v_new, bias_v);

    // ── [STEP 3] rope on Q and K ──
    kernel_rope(q_vec, n_heads=14);
    kernel_rope(k_new, n_heads=2);

    // ── [STEP 4] write K, V to cache ──
    kernel_cache_write(k_new → k_cache[layer][pos]);
    kernel_cache_write(v_new → v_cache[layer][pos]);

    // ── [STEP 5] attention ──
    kernel_attention(q_vec, k_cache[layer], v_cache[layer] → context);

    // ── [STEP 6] attn_out matmul ──
    matmul(context → attn_out);  // pipe: Q5_0

    // ── [STEP 7] residual add + rmsnorm ──
    kernel_residual_add(hidden, attn_out);  // hidden += attn_out
    kernel_rmsnorm(hidden, rms_weight);     // in-place on hidden

    // ── [STEP 8] FFN gate + up matmuls ──
    matmul(hidden → gate);  // pipe: Q5_0
    matmul(hidden → up);    // pipe: Q5_0

    // ── [STEP 9] silu_x_up ──
    kernel_silu_x_up(gate, up);  // gate = silu(gate) * up, in-place

    // ── [STEP 10] ffn_down matmul ──
    matmul(gate → ffn_out);  // pipe: Q6_K

    // ── [STEP 11] final residual ──
    kernel_residual_add(hidden, ffn_out);  // hidden += ffn_out

    metal_batch_end();  // ONE commit + wait for the whole layer
}
```

For 24 layers: call `forward_layer_with_cache_gpu` 24 times → 24 commit+wait total.
For generation: 3 tokens × 24 commits = 72 commits (down from 288).

Wait, this still has one commit+wait per layer! The problem is that each layer depends on the previous layer's `hidden` output. We can't batch ALL 24 layers into one command buffer because each layer reads and writes `hidden` — and while the GPU handles this correctly within a single command buffer, the CPU side still needs to pass different weights for each layer.

Actually, we CAN put all 24 layers in ONE command buffer! The `metal_batch_begin/end` is just about creating a command buffer and encoders. We can put all layers' operations in sequence within the same command buffer. The GPU will execute them sequentially.

But there's a problem: `metal_batch_dispatch` resets the encoder each time the pipe changes. For 24 layers, we have:
- ~5 pipe switches per layer (Q5_0, Q8_0, elem, attn, Q6_K)
- 24 × 5 = 120 encoder switches
- But ALL within one command buffer → zero GPU sync

So the forward pass looks like:

```cpp
metal_batch_begin();
for (int layer = 0; layer < NUM_LAYERS; layer++) {
    encode_layer_operations_gpu(layer, pos);
}
metal_batch_end(); // ONE commit + wait for all 24 layers
```

This is **1 commit+wait** for the entire forward pass (down from 96).

The key insight: `metal_batch_ensure_encoder` already handles pipe switching within a single command buffer. We just need to not call `metal_batch_end()` between layers.

### Threadgroup Configurations Summary

| Kernel | TGs | Threads/TG | SIMDgroups | Threadgroup Mem |
|--------|:---:|:----------:|:----------:|:---------------:|
| Q5_0 matmul | ceil(rows/16) | 256 | 8 | 0 |
| Q8_0 matmul | ceil(rows/16) | 256 | 8 | 0 |
| Q6_K matmul | ceil(rows/16) | 256 | 8 | 0 |
| Q4_K matmul | ceil(rows/16) | 256 | 8 | 0 |
| add_bias | 1 (or ceil(dim/256)) | 256 | - | 0 |
| rope | 1 (or ceil(threads/256)) | 256 | - | 0 |
| cache_write | 1 | 256 | - | 0 |
| attention | 14 | 32 | 1 | 1KB (scores) |
| residual_add | 1 (or ceil(dim/256)) | 256 | - | 0 |
| rmsnorm | 1 | 32 | 1 | 0 (register only) |
| silu_x_up | 1 (or ceil(dim/256)) | 256 | - | 0 |

---

## Buffer Management

### Wrapping existing CPU arrays as Metal buffers

All intermediate buffers are already stack-allocated in `forward_layer_with_cache`:
```cpp
float attn_norm_out[HIDDEN_DIM];
float q_vec[HIDDEN_DIM];
float k_new[K_DIM], v_new[V_DIM];
float context[HIDDEN_DIM];
float attn_out[HIDDEN_DIM];
float ffn_norm_out[HIDDEN_DIM];
float gate[INTER_DIM], up[INTER_DIM];
float ffn_out[HIDDEN_DIM];
```

These are `newBufferWithBytesNoCopy` wrapped per-dispatch at ~1μs each. For the all-GPU approach, we should cache these buffer wrappers too, to avoid redundant ObjC message sends.

Add a general-purpose buffer cache (keyed by pointer):
```cpp
static std::unordered_map<const void*, id<MTLBuffer>> g_buf_cache;
```

### KV cache buffers

The KV cache is `float g_k_cache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM]` — a single contiguous 3D array. Wrap once:
```cpp
id<MTLBuffer> buf_k_cache = wrap_buffer(g_k_cache, sizeof(g_k_cache));
id<MTLBuffer> buf_v_cache = wrap_buffer(g_v_cache, sizeof(g_v_cache));
```

Total KV cache size: 2 × 24 × 256 × 128 × 4 = 6,291,456 bytes = 6 MB — negligible.

### RMS norm weight buffers

Each layer has separate rms weight tensors (attn_norm, ffn_norm). These need to be wrapped per use.

---

## Implementation Plan

### Phase 1 — Infrastructure (metal_backend.hpp)

1. Add `kernel_source` entries for all 7 new kernels
2. Add pipeline state objects to `Context` (pipe_elem, pipe_attn, pipe_rmsnorm)
3. Add dispatch helper functions for each new kernel
4. Add general buffer cache (`g_buf_cache`)
5. Modify `metal_batch_dispatch` to accept pre-existing buffer references (to avoid re-wrapping per call)

### Phase 2 — Kernel Implementation (metal_backend.hpp)

1. `kernel_add_bias` — 10 lines
2. `kernel_rope` — 25 lines  
3. `kernel_cache_write` — 8 lines
4. `kernel_attention` — 55 lines (most complex)
5. `kernel_residual_add` — 5 lines
6. `kernel_rmsnorm` — 15 lines
7. `kernel_silu_x_up` — 8 lines

### Phase 3 — Forward Pass Restructuring (tmac_gguf.cpp)

1. Add `forward_layer_with_cache_gpu` that encodes all ops for one layer
2. Add `forward_all_layers_gpu` that wraps all 24 layers in one batch scope
3. Wire up buffer wrapping for all intermediates
4. Remove old `metal_batch_begin/end` calls from existing code
5. Verify correctness by comparing output tokens

### Phase 4 — Cleanup

1. Remove unused code paths (scalar kernels, old dispatch API)
2. Update profiler to measure the new ops
3. Update ANALYSIS.md with final results

---

## Edge Cases & Risks

| Issue | Risk | Mitigation |
|-------|------|------------|
| KV cache truncated reads | pos < 0 (first token) | Guard with `if (pos < 0) return;` in attention |
| Past_len = 0 | No cached K/V for first token | Attention kernel handles pos=0 as single position |
| Threadgroup memory overflow | scores array > 32KB | MAX_SEQ_LEN=256 → 1KB, well within 32KB limit |
| Large past_len > MAX_SEQ_LEN | Buffer overflow | Already capped at 256 by model config |
| GQA score normalization | Wrong KV head mapping | q_per_kv = 7, integer division for mapping |
| Rope `pow` per thread | Expensive on GPU | Pre-compute via `precise::pow` — <10μs total |
| Bias tensors not present | Null pointer | Skip add_bias if bias tensor is null |
| Q5_0 matmul bias | Matmul + bias not fused | Separate kernel, negligible overhead |
