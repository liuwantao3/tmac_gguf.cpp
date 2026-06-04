# Architecture

## Design: Eliminating Sync Points — Single Command Buffer Forward Pass

### Problem

96 commit+wait cycles per forward pass (4 per layer × 24 layers). Each cycle cost ~1.4ms of GPU drain — the GPU executed ~50μs of actual work then idled for ~1.4ms.

### Root Cause

CPU-in-the-middle operations between GPU matmul batches:

```
GPU: QKV mats → wait → CPU: rope, attn → GPU: attn_out → wait → CPU: rmsnorm → GPU: gate,up → wait → CPU: silu → GPU: ffn_down → wait → CPU: add
```

### Solution

Move ALL CPU-in-the-middle operations to Metal kernels. Encode ALL operations for ALL 24 layers into ONE command buffer. Zero commit+wait during forward pass.

```
BEFORE (96 sync points):
  CPU: ── m_b ── mat ── m_e ──CPU── m_b ── mat ── m_e ──...
  GPU:      [exec]   IDLE      [exec]   IDLE
              ^ 96 cycles of drain/refill

AFTER (1 sync point):
  CPU: ── m_b ────────────────────────────────────────── m_e ──
  GPU:      [exec L0]──[exec L1]──[...]──[exec L23]
              ^ no idle between layers
```

```cpp
metal_batch_begin();
for (int layer = 0; layer < NUM_LAYERS; layer++) {
    encode_layer_operations_gpu(layer, pos);
}
metal_batch_end();  // ONE commit + wait for all 24 layers
```

The key insight: `metal_batch_ensure_encoder` already handles pipe switching within a single command buffer. We just don't call `metal_batch_end()` between layers.

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

## Execution Flow: Single Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      METAL COMMAND BUFFER (1 per forward pass)          │
│                                                                         │
│  ┌─ Encoder[0]: Q5_0 matmul ─────────────────────────────────────────┐  │
│  │  hidden ──► mul_mat_q5_0(rows=896,cols=896) ──► [q_vec]           │  │
│  │  hidden ──► mul_mat_q5_0(rows=128,cols=896) ──► [k_new]           │  │
│  │  hidden ──► mul_mat_q8_0(rows=128,cols=896) ──► [v_new]           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[1]: Element-wise ops ────────────────────────────────────┐  │
│  │  [q_vec] ──► add_bias(dim=896) ──► [q_vec]  (w/ bias_q)          │  │
│  │  [k_new] ──► add_bias(dim=128) ──► [k_new]  (w/ bias_k)          │  │
│  │  [v_new] ──► add_bias(dim=128) ──► [v_new]  (w/ bias_v)          │  │
│  │  [q_vec] ──► rope(n_heads=14, head_dim=64, pos) ──► [q_vec]      │  │
│  │  [k_new] ──► rope(n_heads= 2, head_dim=64, pos) ──► [k_new]      │  │
│  │  [k_new] ──► cache_write(pos, dim=128) ──► [K_cache[layer]]      │  │
│  │  [v_new] ──► cache_write(pos, dim=128) ──► [V_cache[layer]]      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[2]: Attention ───────────────────────────────────────────┐  │
│  │  [q_vec] ──── kernel_attention or kernel_flash_attn ────► [context]│  │
│  │               ◄─── K_cache[layer] ────► V_cache[layer]             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[3]: Q5_0 matmul ────────────────────────────────────────┐  │
│  │  [context] ──► mul_mat_q5_0(rows=896,cols=896) ──► [attn_out]    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[4]: Element-wise ────────────────────────────────────────┐  │
│  │  [hidden] ──► residual_add ──► rmsnorm ──► [hidden]               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[5]: Q5_0 matmul ────────────────────────────────────────┐  │
│  │  [hidden] ──► mul_mat_q5_0(rows=4864,cols=896) ──► [gate]        │  │
│  │  [hidden] ──► mul_mat_q5_0(rows=4864,cols=896) ──► [up]          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[6]: Element-wise ────────────────────────────────────────┐  │
│  │  [gate] ──► silu_x_up(dim=4864) ──► [gate]   (silu*up in-place)   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[7]: Q6_K matmul ────────────────────────────────────────┐  │
│  │  [gate] ──► mul_mat_q6_k(rows=896,cols=4864) ──► [ffn_out]       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Encoder[8]: Element-wise ────────────────────────────────────────┐  │
│  │  [hidden] ──► residual_add(hidden + ffn_out) ──► [hidden] (done!) │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  (output hidden flows into Layer 1 as input)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Two Attention Paths

### `kernel_attn` — Online softmax, 1 head/threadgroup

- 14 threadgroups (one per Q head), 32 threads each (head_dim/2)
- 1 simdgroup per TG → no cross-SG barrier
- Online softmax: tiles of 8 KV positions, single pass
- No threadgroup memory, no barriers

### `kernel_flash_attn` — Flash attention, 8 heads/threadgroup

- `(n_head+7)/8` threadgroups, 64 threads (2 simdgroups)
- 4 heads per simdgroup, GQA-correct per-head KV mapping
- Process ALL KV positions in tiles of 16 using online softmax
- Each thread handles 2 dims: `d0=tiisg*2, d1=tiisg*2+1`
- Enables V cache quantization (requires flash attention in llama.cpp)

## Threadgroup Configurations

| Kernel | TGs | Threads/TG | SIMDgroups | Notes |
|--------|:---:|:----------:|:----------:|-------|
| Q5_0 matmul | ceil(rows/4) | 64 | 2 | Branchless nibble |
| Q8_0 scalar | ceil(rows/4) | 64 | 2 | |
| Q8_0 simdtc | ceil(rows/8) | 64 | 2 | simdgroup_float8x8 |
| Q4_K matmul | ceil(rows/4) | 64 | 2 | Branchless |
| Q6_K matmul | ceil(rows/4) | 64 | 2 | |
| fused_qkv | ceil(sum_rows/4) | 64 | 2 | Q+K+V combined |
| attention | n_head | 32 | 1 | kernel_attn |
| flash_attn | ceil(n_head/8) | 64 | 2 | kernel_flash_attn |
| rope | 1 (or ceil/64) | 64 | — | |
| rmsnorm | 1 | 32 | 1 | |
| elem ops | ceil(dim/64) | 64 | — | add, silu, copy |

## Buffer Management

### Wrapping existing CPU arrays as Metal buffers

All intermediate buffers are stack-allocated in the forward pass. They're wrapped via `newBufferWithBytesNoCopy` per use. A pointer-keyed cache (`g_buf_cache`) avoids re-wrapping.

### KV cache buffers

`float g_k_cache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM]` — contiguous 3D array. Wrapped once per layer. Total: 2 × 24 × 256 × 128 × 4 = 6 MB.

### Weight buffer cache

Per-tensor-pointer MTLBuffer reuse via `g_w_cache` — weights are immutable, wrap once per tensor.

## Metal Kernels Internals

### Quantized matmul pattern (Q5_0, Q8_0, Q4_K, Q6_K)

```
Block: [scale (FP16)] [quantized values]
Threads: 32 per SIMD group × 2 groups = 64 per threadgroup
Each thread:
  for each block:
    xv = x[block_id * 32 + lane]          // one x element per lane
    for each row in local rows:
      scale = W[block_base] (FP16 → float)
      q_val = W[block_base + 2 + lane] (int8 → float)
      sum[row] += q_val * scale * xv
  simd_sum across lanes → full dot product
```

### Branchless nibble extraction (Q5_0, Q4_K, Q6_K)

Replaces ternary branches with per-lane shift amounts, eliminating SIMD divergence:

| Quant | Divergent pattern | Branchless |
|-------|------------------|------------|
| Q5_0 | `lane < 16 ? (qs & 0xF) : (qs >> 4)` | `(qs >> ((lane>>4)*4)) & 0xF` |
| Q4_K | `(sub & 1) ? (byte >> 4) : (byte & 0xF)` | `(byte >> ((sub & 1) * 4)) & 0xF` |

### Q8_0 simdgroup tensor core (mul_mat_q8_0_simdtc)

Uses Apple GPU `simdgroup_float8x8` matrix multiply unit:
- Dequantize W into threadgroup memory `sa[8×64]` (both simdgroups load 2 Q8_0 blocks)
- Broadcast x values: `sb[row×8 + 0..7] = x_val[row] × 8` (replicated)
- 8 tiles of `simdgroup_load` + `simdgroup_multiply_accumulate` per iteration
- Accumulate all blocks in registers, store once at end

---

## Per-Layer Operations List (One Forward Pass)

Each of the 24 layers executes these ~21 dispatches sequentially within a single command buffer. Operations are data-dependent — each reads the previous op's output.

### Layer Step-by-Step

```
Step   Op                  Kernel              Read          Write         Dims
────   ──────────────────  ──────────────────  ────────────  ────────────  ─────────
  1    Copy hidden         elem_op(copy)       hidden        scratch       896
  2    Attention RMS norm  rmsnorm_op          scratch      scratch       896
  3    Q matmul            fused_qkv_op        scratch      q_vec,         896×896
  4    K matmul            (Q+K+V combined)                  k_new,        128×896
  5    V matmul                                             v_new         128×896
  6    Q bias add          elem_op(add)        q_vec, bias_q q_vec         896
  7    K bias add          elem_op(add)        k_new, bias_k k_new         128
  8    V bias add          elem_op(add)        v_new, bias_v v_new         128
  9    Q RoPE              rope_op             q_vec         q_vec         14×64
 10    K RoPE              rope_op             k_new         k_new         2×64
 11    K cache write       elem_op(write)      k_new         K_cache[layer] 128
 12    V cache write       elem_op(write)      v_new         V_cache[layer] 128
 13    Attention           attention_op        q_vec,        context       14×64
                                               K_cache,
                                               V_cache
 14    Attn output matmul  matmul_q5_0         context       attn_out      896×896
 15    Residual add        elem_op(add)        hidden,       hidden        896
                                               attn_out
 16    FFN RMS norm        rmsnorm_op          hidden        scratch       896
 17    FFN gate matmul     matmul_q5_0         scratch       gate          4864×896
 18    FFN up matmul       matmul_q5_0         scratch       up            4864×896
 19    SiLU × up           elem_op(silu)       gate, up      gate          4864
 20    FFN down matmul     matmul_q4_k/q6_k    gate          ffn_out       896×4864
 21    Residual add        elem_op(add)        hidden,       hidden        896
                                               ffn_out
```

### Logits Projection (after all layers)

```
 22   Output norm         rmsnorm_op          hidden        scratch       896
 23   Logits matmul       matmul_q8_0         scratch       logits        151936×896
```

---

## Weight Quantization Reference (Qwen2-0.5B)

All tensors loaded from GGUF and converted to TMAC format. Original quantization types preserved per tensor.

### Global Tensors

| Tensor | Quant | Dimensions |
|--------|:-----:|:----------:|
| `token_embd.weight` | Q8_0 | 151936 × 896 |
| `output_norm.weight` | F32 | 1 × 896 |

### Per-Layer Weights (24 layers, same structure)

Each layer has 9 weight tensors. Quantization varies for `ffn_down.weight` and `attn_v.weight` across layers:

| Tensor | Quant | Dimensions | Notes |
|--------|:-----:|:----------:|-------|
| `attn_q.weight` | Q5_0 | 896 × 896 | |
| `attn_k.weight` | Q5_0 | 128 × 896 | |
| `attn_v.weight` | **Q8_0** or **Q5_0** | 128 × 896 | See layer table |
| `attn_output.weight` | Q5_0 | 896 × 896 | |
| `ffn_gate.weight` | Q5_0 | 4864 × 896 | |
| `ffn_up.weight` | Q5_0 | 4864 × 896 | |
| `ffn_down.weight` | **Q6_K** or **Q4_K** | 896 × 4864 | See layer table |
| `attn_norm.weight` | F32 | 1 × 896 | |
| `ffn_norm.weight` | F32 | 1 × 896 | |

Biases (all F32): `attn_q.bias` (896), `attn_k.bias` (128), `attn_v.bias` (128).

### Layer-by-Layer Variant Weights

`ffn_down.weight` and `attn_v.weight` alternate quantization types across layers (correlated pairs):

| Layers | ffn_down | attn_v |
|:------:|:--------:|:------:|
| 0, 1, 3, 6, 7, 8, 9, 10, 13, 16, 19, 21 | **Q6_K** | **Q8_0** |
| 2, 4, 5, 11, 12, 14, 15, 17, 18, 20, 22, 23 | **Q4_K** | **Q5_0** |

All other per-layer tensors are identical across all 24 layers.
