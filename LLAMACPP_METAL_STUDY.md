# llama.cpp Metal Backend: Technical Study

> Based on analysis of `ggml/src/ggml-metal/ggml-metal.metal` (10,699 lines) and `ggml-metal-device.cpp`.  
> Target: adopt key techniques in tmac_gguf.cpp to close the 3.1× performance gap with llama.cpp (147 vs 47 t/s).

---

## 1. Threadgroup Sizing: 64 Threads (2 simdgroups)

**llama.cpp pattern:** Every kernel uses `num_threadsgroups = 64` with `[[threads_per_threadgroup(64)]]`.  
**Used in:** quantized matmuls (`kernel_mul_mat_q*_f32`), attention, RMS norm, rope, FFN.

**Why 64 not 256 (our choice)?**
- M1 has 32KB threadgroup memory per TG, 1024 max threads per TG.
- 64 threads = 2 simdgroups × 32 threads each. Fits register budget for quantized matmul (each thread unpacks ~20 registers for weights).
- 256 threads = 8 simdgroups — OK for large rows >1024 where occupancy helps hide latency.
- For small rows (896 in Qwen2 hidden dim), 64 threads keeps 8+ TGs in flight (max 16 per CU on M1), saturating the 16 CU cores.
- `nsg=2` (2 simdgroups) kernel variants exist for each quantization type, selected in `ggml_metal_choose_kernel()` based on tensor dimensions.

**Our problem:** We set TG=256 for simplicity. This wastes TGs on small-row ops (rmsnorm, rope, cache_kv, add_bias) and reduces occupancy. Also increases register pressure — each thread handles fewer elements but more threads compete for registers.

**Adoption plan:** Change `threadsPerThreadgroup` from `MTLSizeMake(256, 1, 1)` to `MTLSizeMake(64, 1, 1)` across all kernels, with `numThreadgroups = ceil_div(N, 64)` for 1D dispatch or `ceil_div(N, 2) × ceil_div(rows, 32)` for 2D dispatch.

---

## 2. Flash Attention: `kernel_flash_attn_ext_vec_f16_dk64_dv64`

**llama.cpp has a dedicated flash attention kernel** (lines ~5400-5730) that computes:
```
S = Q × K^T (score matrix)
P = softmax(S)
O = P × V
```
entirely on-chip, never materializing the full `seq_len × seq_len` attention matrix.

### Key implementation details:

**Tiling:**
- Tiles Q across queries (`nq = 2 or 4`), K/V across KV sequence (`nk = 8 or 16`).
- Inner tile: `nq × dk` for Q, `nk × dk` for K, `nk × dv` for V.
- Each tile loaded into threadgroup memory.

**Online softmax (safe softmax):**
- Maintains row-wise `m_prev = max(m_prev, row_max(S_tile))` and `d_prev = sum(exp(S_row - m_prev))` across tiles.
- `O = O * exp(m_prev - m_new) / d_new + P_tile * V_tile / d_new`.
- No separate reduction pass needed — single loop over KV tiles.

**Vectorized:**
- `vec_f16` loads 8×f16 values at once using `half8`.
- dk=64 fits exactly in 4×vec_f16 loads per thread.
- `[[function_constant(0)]]` for `DK` and `DV` specialization — avoids runtime branches.

**SIMDgroup math:**
- Uses `simdgroup_multiply_accumulate` for matrix multiply within a simdgroup (32 threads).
- Inter-TG communication via threadgroup memory for reductions.

**Our current attention:** naive O(seq²) math with threadgroup softmax, loading all KV into threadgroup memory. For seq_len=256 this is fine (256×256×64 = 4M FLOPs), but flash would improve memory efficiency and enable >256 context.

**Adoption plan:** Implement `kernel_flash_attn_vec_f16` with:
- `nq=2, nk=8` tiles for Qwen2 head_dim=64, kv_dim=128.
- Online softmax loop over KV tiles.
- Half-precision (`half`) for score computation.
- Function constants for `DK`, `DV`, `nq`, `nk`.

---

## 3. Fused Gated FFN: `gated_delta_net`

**llama.cpp fuses the entire FFN** (silu(x_up) * x_gate → x_down) into a single kernel.
- Present in kernel source (~lines 3150-3260 in the original, ~lines 6150-6260 in the expanded version).
- Dispatch: `{1, 1, 1}` threadgroups, `{32, 32, 1}` threads.
- Each threadgroup handles a group of output elements, loading weights from the same input row.

**Benefit:** One dispatch instead of 4 (up_proj, gate_proj → silu → mul → down_proj). Saves 3× launch overhead, 2× weight loads (weights read once from DRAM instead of twice).

**Our current FFN:** 5 dispatches (up_proj + gate_proj in one batch, silu_x_up, down_proj, add_residual). Fused would cut to 2 (attention, FFN) per layer.

**Adoption plan:**
- Implement `kernel_fused_ffn_silu_gate` taking `x`, `W_up`, `W_gate`, `W_down`, `bias_up`, `bias_gate`, `bias_down` (all already loaded in GPU buffers).
- Single dispatch: `{1, 1, 1}` threadgroups × `{32, 32, 1}` threadgroup.
- TG handles sub-range of `INTER_DIM` output elements for up/gate, then the same TG does silu × gate multiply, then applies down_proj for the same range.

---

## 4. Function Constants for Quantization Specialization

**llama.cpp uses `[[function_constant(idx)]]` extensively** for:
- Quantization type (`FUNCTION_CONSTANT_QK` = quantized block type at index 0).
- Head dimensions for flash attention (`FUNCTION_CONSTANT_DK`, `FUNCTION_CONSTANT_DV`).
- Kernel variants (e.g., `FUNCTION_CONSTANT_NUM_K_QUANTS`).

**Usage in `ggml-metal-device.cpp`** (lines ~1150-1250):
```cpp
// For each quantization type, create a specialized pipeline:
for (auto qtype : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, ..., GGML_TYPE_Q8_0}) {
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&qtype type:MTLDataTypeInt atIndex:0];
    // Also set block size, block alignment, etc.
    auto function = [library newFunctionWithName:@"kernel_mul_mat_qX_f32"
                                  constantValues:constants error:&error];
    auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    // Cache by qtype.
}
```

**Result:** Zero runtime `if (qtype == ...)` branches inside the hot matmul kernel. Each pipeline has the quant type baked in, enabling better Metal compiler optimizations (constant folding, loop unrolling, dead code elimination).

**Our current approach:** Runtime `switch (qtype)` in `metal_mul_mat_q()` per batch element. Each thread evaluates the same switch repeatedly.

**Adoption plan:** Extend our function constant approach (already used for `ELEM_OP`, `RNS_OP`, `ATTN_OP`) to quantization type. Create pipeline cache keyed by `{qtype, op_type}`. Compile at load time for all supported quant types.

---

## 5. Graph Scheduling (Metal-GPU DAG)

**llama.cpp doesn't execute one layer at a time.** It builds a full DAG of all operations (all layers), then `ggml_graph_compute()` schedules:

1. **Concurrent ops:** Independent operations across different layers execute concurrently (e.g., RMS norm in layer 3 while attention matmul in layer 2).
2. **Out-of-order:** Memory writes are tracked via `ggml_tensor` lifetimes; the scheduler reorders ops based on data dependencies, not source order.
3. **Batch submission:** Multiple command buffers are filled and committed concurrently, then waited.

**Our approach:** Sequential layers with explicit syncs (even in fused path, layers within the CB are sequential). No cross-layer concurrency.

**Adoption complexity:** High. Requires DAG infrastructure (or porting ggml). Medium-term optimization.

---

## 6. Multi-Token Generation (Prefill/Batch)

**llama.cpp prefill:** Matrix-matrix multiplication for prompt processing. Processes all prompt tokens in parallel using a single matmul (K = prompt_embeds, QKV projection is [prompt_len × hidden] × [hidden × 3*hidden]).

**Our prefill:** Same single-token forward loop as generation. Each prompt token is processed identically to a generation step, with KV cache management but no parallelism gain.

**Adoption plan:** Implement batched prefill as a separate path. For prompt_len > 1, use matmul-based QKV projection with prompt_len as batch dimension; similarly for FFN. This would speed prefill from 29ms/tok to ~2ms/tok for a 256-token prompt.

---

## 7. Smaller Details

### 7a. RMS Norm + Scale Kernel
llama.cpp fuses `rmsnorm(x) * weight` into a single kernel. We already do this (`kernel_rmsnorm` writes scaled output). No change needed.

### 7b. ROPE with Complex Multiplication
llama.cpp uses a single kernel for rope that computes both cos/sin and applies them via complex multiplication. We do the same.

### 7c. Cache Line Awareness
llama.cpp aligns KV cache blocks to 16 bytes (half8 alignment) and uses vectorized writes. Our cache write kernel already fuses `cache_k + cache_v` and adds previous values, but doesn't use vectorized stores.

### 7d. No Redundant `clear` or Memset
llama.cpp never clears buffers before writing. We also don't (`newBuffer` with zero-filled is avoided). Good.

### 7e. Dedicated INT4/INT8 Quantized Matmul Kernels
Each quantization type has its own `kernel_mul_mat_qX_f32`. No generic loop over block types. We already have separate `kernel_mul_mat_q4_0_f32`, `kernel_mul_mat_q4_k_f32`, `kernel_mul_mat_q6_k_f32` functions. Good.

---

## 8. Performance Bottleneck Analysis for tmac_gguf.cpp

| Bottleneck | Impact | Fix | Difficulty |
|---|---|---|---|
| 256-thread TGs (low occupancy) | ~1.5× slowdown | Switch to 64 threads | Low |
| Separate-kernel dispatch overhead | ~2-3× (8 vs 1 CB) | Already done (fused) | Done |
| No flash attention | ~1.5× attention | Implement flash attn | Medium |
| No fused FFN | ~1.3× FFN | Implement fused FFN | Medium |
| No graph scheduling | ~1.2-1.5× | Port DAG scheduler | High |
| No batched prefill | ~10× prefill | Matmul-based prefill | Medium |
| Runtime quant type switch | ~1.05-1.1× | Function constants | Low |
| Suboptimal memory layout | Unknown | Profile + tune | Ongoing |

**Priority for next session (highest impact → lowest effort):**

1. **[Low effort] Switch to 64-thread TGs** — change `MTLSize(256,...)` → `MTLSize(64,...)` and adjust grid sizes. Expected gain: ~1.5× (est. 14ms → 9.5ms).
2. **[Medium effort] Flash attention** — implement `kernel_flash_attn_vec_f16`. Expected gain: ~1.5× on attention (est. 4ms → 2.7ms).
3. **[Medium effort] Fused FFN** — merge silu_x_up + down_proj + residual_add into one kernel. Expected gain: ~1.3× (est. 8ms → 6ms).
4. **[High effort] Batched prefill** — separate path for prompt processing with matmul dispatch. Expected gain: ~10× prefill speed.
5. **[Low/Med effort] Function constants for quant types** — compile specialized pipelines at init. Expected gain: ~5-10% on quant matmul.

**Target after items 1-3:** ~6ms/tok (comparable to llama.cpp's 6.8ms for generation).

---

## 9. Reference: Key llama.cpp File Locations

| File | Lines | Content |
|---|---|---|
| `ggml-metal.metal` | 1-320 | Quantized block types, shared structs, utility functions |
| `ggml-metal.metal` | 321-1900 | Quantized dequant functions (q4_0, q4_1, q5_0, q5_1, q8_0, q2_K through q6_K) |
| `ggml-metal.metal` | 1901-3400 | Quantized matmul kernels (`kernel_mul_mat_q4_0_f32`, ..., `kernel_mul_mat_q6_K_f32`) |
| `ggml-metal.metal` | 3401-4700 | Dequant + matmul (grouped by quant type), RMS norm, rope |
| `ggml-metal.metal` | 4701-5400 | `kernel_mul_mv` (GQA split), `get_rows` |
| `ggml-metal.metal` | 5401-5730 | `kernel_flash_attn_ext_vec_f16_dk64_dv64` |
| `ggml-metal.metal` | 5731-6200 | `kernel_flash_attn_ext_f16_hz`, cross-attention variants |
| `ggml-metal.metal` | 6201-6400 | Fused gated delta net, dequant + mul_mat for various types |
| `ggml-metal.metal` | 6401-7300 | More quantized matmul, softmax, `kernel_alibi` |
| `ggml-metal.metal` | 7301-8500 | `kernel_mul_mat_id` (expert routing), MoE kernels |
| `ggml-metal.metal` | 8501-10699 | `kernel_flash_attn_ext_vec_f16_dk128_dv128`, other large-head variants |
| `ggml-metal-device.cpp` | 50-240 | Device discovery, buffer management, `ggml_backend_metal_buffer_type` |
| `ggml-metal-device.cpp` | 241-900 | Kernel selection (`ggml_metal_choose_kernel`), pipeline compilation, constant specialization |
| `ggml-metal-device.cpp` | 901-1300 | Graph scheduling, dispatch logic for tensors |
| `ggml-metal-device.cpp` | 1301-1550 | `ggml_backend_metal_graph_compute` — main compute dispatcher |
| `ggml-metal-device.cpp` | 1551-1925 | Buffer ops, synchronization, tensor lifetime management |

---

## 10. Analysis of 64-Thread vs 256-Thread Performance

**Hypothesis for why our naive 64-thread attempt regressed:**

We simply changed `num_threads` from 256 to 64 and `num_groups` from N/256 to N/64. This doesn't account for:

1. **Register pressure:** With 256 threads, each thread handles `ceil_div(256, 8*256) = 1` block per thread = trivial. With 64 threads, each thread might need `ceil_div(256, 8*64) = 1` but the TG has 4× fewer threads to cover the same work. The real issue is the *inner loop structure* — llama.cpp's kernels have specific unrolling factors tuned for 64 threads.

2. **SIMDgroup utilization:** At 64 threads (2 simdgroups), the Metal compiler can assign each simdgroup a contiguous chunk of the output. At 256 threads, inter-simgroup synchronization is needed more often.

3. **Memory access patterns:** llama.cpp's kernels load data in specific patterns for 64 threads — each thread loads 4 consecutive blocks, ensuring coalesced access. A direct thread-count swap without adjusting load patterns breaks coalescing.

**Proper approach:** 
- Copy llama.cpp's exact kernel structure for the quant types we use (Q4_K, Q6_K).
- Match their block dispatch stride (`nth = thread_index * 4` for Q4_K).
- Keep `nsg=2` variant (or detect and compile both).

---

## 11. Conclusion

tmac_gguf.cpp's fused single-CB approach already matches llama.cpp in architectural ambition (eliminating dispatch overhead). The remaining 3.1× gap comes from:

- **Threadgroup sizing** (64 vs 256 threads) — estimated 1.5× gain
- **Missing flash attention** — estimated 1.5× gain (multiplicative with above)
- **Missing fused FFN** — estimated 1.3× gain
- **Suboptimal quantized matmul inner loops** — estimated 1.2-1.5× gain
- **No graph-level optimization** — marginal for single-CB path

Combined: 1.5 × (1/0.67) × (1/0.77) ≈ 3.4× improvement possible, implying ~6.2ms/tok — close to llama.cpp's 6.8ms/tok.

**Recommendation:** Implement items 1-3 (64-thread TGs, flash attention, fused FFN) in order. After each, benchmark against llama.cpp to measure actual vs estimated gain. Then decide if batched prefill or graph scheduling are needed.
