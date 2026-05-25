# Execution Flow: Single Command Buffer Forward Pass

## Legend

```
┌──────────┐  Encoder boundary (pipe switch)
│ KERNEL   │  GPU kernel dispatch
╞══════════╡  Threadgroup boundary within kernel
──────────►  Data flow / dependency
    [buf]    Intermediate buffer (shared memory)
```

---

## Layer 0 Internal Flow (Expandable)

All 24 layers share this pattern. Only Layer 0 shown at full detail.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      METAL COMMAND BUFFER                               │
│                                                                         │
│  ┌─ Encoder[0]: Q5_0 matmul ─────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  hidden ──► mul_mat_q5_0(rows=896,cols=896) ──► [attn_norm_out]   │  │
│  │  hidden ──► mul_mat_q5_0(rows=896,cols=896) ──► [q_vec]           │  │
│  │  hidden ──► mul_mat_q8_0(rows=128,cols=896) ──► [k_new]           │  │
│  │  hidden ──► mul_mat_q8_0(rows=128,cols=896) ──► [v_new]           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[1]: Element-wise ─────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [q_vec] ──► add_bias(dim=896) ──► [q_vec]  (w/ bias_q)          │  │
│  │  [k_new] ──► add_bias(dim=128) ──► [k_new]  (w/ bias_k)          │  │
│  │  [v_new] ──► add_bias(dim=128) ──► [v_new]  (w/ bias_v)          │  │
│  │  [q_vec] ──► rope(n_heads=14, head_dim=64, pos) ──► [q_vec]      │  │
│  │  [k_new] ──► rope(n_heads= 2, head_dim=64, pos) ──► [k_new]      │  │
│  │  [k_new] ──► cache_write(pos, dim=128) ──► [K_cache[layer]]      │  │
│  │  [v_new] ──► cache_write(pos, dim=128) ──► [V_cache[layer]]      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[2]: Attention ────────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [q_vec] ◄─── kernel_attention ────► K_cache[layer]               │  │
│  │    │              (14 TGs × 32 threads)                           │  │
│  │    │              tgmem: scores[256] floats                        │  │
│  │    ▼                       ▲                                       │  │
│  │  [context]          V_cache[layer]                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[3]: Q5_0 matmul ──────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [context] ──► mul_mat_q5_0(rows=896,cols=896) ──► [attn_out]     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[4]: Element-wise ─────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [hidden] ──► residual_add(original_hidden + attn_out) ──►[hidden]│  │
│  │  [hidden] ──► rmsnorm(dim=896, weight) ──► [hidden]               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[5]: Q5_0 matmul ──────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [hidden] ──► mul_mat_q5_0(rows=4864,cols=896) ──► [gate]         │  │
│  │  [hidden] ──► mul_mat_q5_0(rows=4864,cols=896) ──► [up]           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[6]: Element-wise ─────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [gate] ──► silu_x_up(dim=4864) ──► [gate]   (silu*up in-place)  │  │
│  │                        ▲                                           │  │
│  │                   [up] (read-only)                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[7]: Q6_K matmul ──────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [gate] ──► mul_mat_q6_k(rows=896,cols=4864) ──► [ffn_out]        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│  ┌─ Encoder[8]: Element-wise ─────┼─────────────────────────────────┐  │
│  │                                ▼                                  │  │
│  │  [hidden] ──► residual_add(hidden + ffn_out) ──► [hidden] (done!) │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│          (output hidden flows into Layer 1 as input)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Full 24-Layer Forward Pass (Generation, 1 token)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 0    (8 encoders, 16 dispatches — detailed above)         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 1    (8 encoders, 16 dispatches)                          │   │
│  │    hidden(1) ──► rmsnorm ──► Q,K,V ──► rope ──► cache_write     │   │
│  │    ──► attention ──► attn_out ──► residual ──► rmsnorm          │   │
│  │    ──► gate,up ──► silu_x_up ──► ffn_down ──► residual          │   │
│  │    ──► output: hidden(2)                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2 ...                                                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│               ...  (21 more layers)  ...                                │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 23   (last layer)                                        │   │
│  │    ... ──► residual ──► hidden(24)  ◄── FINAL OUTPUT             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ╔═══════════════════════════════════════════════════════════════════╗   │
│  ║           END ENCODING + COMMIT + waitUntilCompleted             ║   │
│  ║             1 sync point for all 24 layers                       ║   │
│  ╚═══════════════════════════════════════════════════════════════════╝   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Timing Estimate

| Segment | Encoder Switches | Kernel Dispatches | Est. GPU Time |
|---------|:-:|:-:|:---:|
| L0 QKV mats (3×) | 0 | 3 | ~60μs |
| L0 elem ops (bias, rope, write) | 1 | 5 | ~10μs |
| L0 attention | 1 | 1 | ~30μs (pos=5) to ~300μs (pos=255) |
| L0 attn_out mat | 1 | 1 | ~24μs |
| L0 post-attn (residual, rmsnorm) | 1 | 2 | ~5μs |
| L0 gate+up mats (2×) | 1 | 2 | ~50μs |
| L0 silu_x_up | 1 | 1 | ~8μs |
| L0 down mat | 1 | 1 | ~24μs |
| L0 final residual | 1 | 1 | ~3μs |
| **Total L0** | **8** | **17** | **~214-484μs** |
| **All 24 layers** | **192** | **408** | **~5-12ms** |
| + 1x commit/wait overhead | — | — | ~1ms |
| **Total per gen token** | | | **~6-13ms** |

**Current**: ~60ms per token → **~77-92% reduction** in forward pass time.

---

## Memory Access Pattern: Attention Kernel Detail

```
K_cache layout: [pos][kv_head][dim]  (contiguous in pos)
Q layout:       [head][dim]          (contiguous in dim)

Each TG (head h):
  Thread t reads Q[h][2t], Q[h][2t+1]  ──── register
  Phase 1 (pos loop):
    Thread t reads K_cache[p][kv][2t], K_cache[p][kv][2t+1]  ──── device memory
    simd_sum → score[p]                                        ──── tgmem write (thread 0)
  Phase 2 (softmax):
    strided read of scores[p] from tgmem → simd_max/sum
    strided write exp value back to tgmem
  Phase 3 (V weighted sum):
    Thread t reads V_cache[p][kv][2t], V_cache[p][kv][2t+1]  ──── device memory
    Thread t reads scores[p] from tgmem                       ──── tgmem (same val for all)
    simd_sum across threads → output[h][2t], output[h][2t+1]
```

---

## Comparison: Before vs After

```
BEFORE (96 sync points):
  CPU: ── m_b ── mat ── m_e ──CPU── m_b ── mat ── m_e ──CPU── m_b ── mat ── m_e ── ...
  GPU:      [exec]   IDLE      [exec]   IDLE      [exec]   IDLE
             ^ 96 cycles of drain/refill

AFTER (1 sync point):
  CPU: ── m_b ────────────────────────────────────────────────────────────────── m_e ──
  GPU:      [exec L0]──[exec L1]──[exec L2]──...──[exec L23]  (contiguous)
             ^ no idle between layers
```

```
m_b = metal_batch_begin()
m_e = metal_batch_end()  → commit + waitUntilCompleted
mat  = matmul / kernel dispatch
```
