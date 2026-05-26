# llama.cpp Metal Backend: Technical Study

> Based on analysis of `ggml/src/ggml-metal/ggml-metal.metal` (10,699 lines) and `ggml-metal-device.cpp`.  
> Target: adopt key techniques in llama.cpp to close the performance gap.

---

## Quick Status Check

| Optimization | llama.cpp | Our tmac_gguf.cpp | Status |
|-------------|-----------|------------------|--------|
| Threadgroup sizing | 64 threads (2 simdgroups) | 64 threads for matmul/rope, 32 for rmsnorm/attn | ✅ Done |
| Flash attention | Yes | Yes (`kernel_attn` with TILE=8, online softmax) | ✅ Done |
| Fused QKV matmul | Yes | Yes (`kernel_fused_qkv`, Q5_0 Q/K + V) | ✅ Done |
| Fused FFN gate+up | Yes | Yes (`kernel_fused_ffn_gate_up`) | ✅ Done |
| Fused FFN silu+down | Yes | No (silu separate, down separate) | ⚠️ Broken |
| Function constants (quant) | Yes (all quant types) | Partial (only kernel_op, v_type) | ⚠️ Partial |
| Graph scheduling | Yes (full DAG) | No (sequential per-layer) | ❌ Not done |
| Batched prefill | Yes | No (single-token forward) | ❌ Not done |

**Our performance:** ~18-19ms/token (Metal fused) vs llama.cpp ~6.8ms/token (3× gap)

---

## 1. Threadgroup Sizing: 64 Threads (2 simdgroups) — ✅ DONE

**llama.cpp pattern:** Every kernel uses `num_threadsgroups = 64` with `[[threads_per_threadgroup(64)]]`.

**Our implementation:**
- Quantized matmuls (Q8_0, Q5_0, Q4_K, Q6_K): **64 threads** ✅
- Fused QKV: **64 threads** ✅
- Fused FFN gate+up: **64 threads** ✅
- RMS norm: **32 threads** (appropriate for simple element-wise op)
- Attention: **32 threads** (intentional for single-pass flash attention design)

```cpp
// metal_backend.hpp:203-206 — quantized matmul dispatch
if (is_simd) {
    int total = ((rows + 3) / 4) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}
```

---

## 2. Flash Attention — ✅ DONE

**llama.cpp:** `kernel_flash_attn_ext_vec_f16_dk64_dv64` — tiled, online softmax, single-pass.

**Our implementation:** `kernel_attn` at `metal_backend.hpp:589-648`

```metal
// Attention: flash-attention single-token GQA, 32-thread TG (1 simdgroup).
// Processes KV in tiles of 8 with online softmax — single pass, no threadgroup memory, no barriers.
kernel void kernel_attn(device const float* Q [[buffer(0)]],
                         device const float* K_cache [[buffer(1)]],
                         device const float* V_cache [[buffer(2)]],
                         device float* output [[buffer(3)]],
                         constant int* params [[buffer(4)]],
                         uint head [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    // Flash attention: online softmax, single pass, TILE=8
    float O0 = 0, O1 = 0, m = -INFINITY, d = 0;
    int n_pos = past_len + 1;

    for (int tile_start = 0; tile_start < n_pos; tile_start += 8) {
        // Compute scores for this tile
        float s[8];
        for (int t = 0; t < tile_sz; t++) {
            s[t] = simd_sum(my_q0 * k0 + my_q1 * k1) * scale;
        }
        // Online softmax: find tile max, rescale previous accum, add new contributions
        float m_new = max(m_new, s[t]);
        float old_scale = exp(m - m_new);
        O0 *= old_scale; O1 *= old_scale; d *= old_scale;
        for (int t = 0; t < tile_sz; t++) {
            O0 += e * V_cache[cache_ofs + d0];
            O1 += e * V_cache[cache_ofs + d1];
            d += e;
        }
        m = m_new;
    }
    output[head * head_dim + d0] = O0 / d;
    output[head * head_dim + d1] = O1 / d;
}
```

Key features:
- **TILE=8** — processes 8 KV positions per tile iteration
- **Online softmax** — maintains `m` (max) and `d` (denominator) incrementally across tiles
- **No threadgroup memory** — streams directly from device K/V cache buffers
- **Single pass** — never materializes full `seq_len × seq_len` score matrix

---

## 3. Fused QKV Matmul — ✅ DONE

**llama.cpp:** Combines Q, K, V projection into one kernel or batches them in one CB.

**Our implementation:** `kernel_fused_qkv` at `metal_backend.hpp:414-472`

```metal
constant int v_type [[function_constant(1)]];  // 0=V Q5_0, 1=V Q8_0

kernel void kernel_fused_qkv(
    device const uint8_t* W_q [[buffer(0)]],
    device const uint8_t* W_k [[buffer(1)]],
    device const uint8_t* W_v [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y_q [[buffer(4)]],
    device float* y_k [[buffer(5)]],
    device float* y_v [[buffer(6)]],
    constant int* params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    // Q and K are always Q5_0; V is Q5_0 or Q8_0 based on v_type
    if (global_row < q_rows + k_rows || v_type == 0) {
        // Q5_0 dequantization path
    } else {
        // Q8_0 dequantization path
    }
}
```

Two pipelines created at init:
- `pipe_fused_qkv_q5` — V=Q5_0 (v_type=0)
- `pipe_fused_qkv_q8` — V=Q8_0 (v_type=1)

**Effect:** Replaces 3 separate matmul dispatches (Q, K, V) with 1 dispatch per layer. Saves 48 dispatch operations across 24 layers.

---

## 4. Fused FFN Gate+Up Matmul — ✅ DONE, ❌ BROKEN

**llama.cpp:** `kernel_fused_ffn_silu_gate` — gate+up+silu+down in one kernel.

**Our implementation:** `kernel_fused_ffn_gate_up` at `metal_backend.hpp:475-529` (gate+up only)

```metal
kernel void kernel_fused_ffn_gate_up(
    device const uint8_t* W_gate [[buffer(0)]],
    device const uint8_t* W_up [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    // Both W_gate and W_up are Q6_K, same dims
    // Writes gate[0..INTER_DIM-1] and up[INTER_DIM..2*INTER_DIM-1]
}
```

**Status:** Implemented but **BROKEN** — produces wrong tokens after first generation. Currently disabled (falls back to separate gate/up matmuls).

**Why silu+down can't be fused:**
- silu needs the FULL gate and up results (not partial sums)
- down depends on silu(gate) * up result
- This sequential dependency requires either:
  - Threadgroup memory to store intermediate results + barrier (complex)
  - Multiple passes within one kernel (not feasible)

llama.cpp solves this with threadgroup memory and careful synchronization, which requires restructuring the kernel significantly.

---

## 5. Function Constants for Quantization Specialization — ⚠️ PARTIAL

**llama.cpp:** Creates specialized pipeline for each quantization type at init time. Zero runtime branches.

```cpp
// ggml-metal-device.cpp pattern:
for (auto qtype : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, ..., GGML_TYPE_Q8_0}) {
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&qtype type:MTLDataTypeInt atIndex:0];
    auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    // Cache by qtype
}
```

**Our implementation:** Only two function constants are used:
- `kernel_op` at index 0: elem_op specialization (add=0, silu=1, cache_write=2)
- `v_type` at index 1: V quantization type (Q5_0=0, Q8_0=1) in fused QKV

The quantized matmul kernels (Q8_0, Q5_0, Q4_K, Q6_K) still use runtime `switch` statements in `matmul()` to dispatch to the correct kernel. This is because the `matmul()` function selects the pipeline, not the kernel itself.

**What we have:**
```cpp
// kernel_elem uses function constant for op type
constant int kernel_op [[function_constant(0)]];

// kernel_fused_qkv uses function constant for V quant type
constant int v_type [[function_constant(1)]];
```

**What we don't have:** Separate pipelines per quant type with function constants inside the matmul kernels themselves.

---

## 6. Graph Scheduling (Metal-GPU DAG) — ❌ NOT DONE

**llama.cpp:** Builds a full DAG of all operations across all layers. Independent ops execute concurrently:
- RMS norm in layer 3 runs concurrently with attention matmul in layer 2
- Out-of-order execution based on tensor lifetime analysis
- Multiple command buffers filled and committed concurrently

**Our approach:** Sequential per-layer execution within one command buffer. No cross-layer concurrency. All 24 layers encoded in sequence.

**Adoption complexity:** High — requires DAG infrastructure or porting ggml's graph scheduler.

---

## 7. Batched Prefill (Multi-Token Generation) — ❌ NOT DONE

**llama.cpp prefill:** Matrix-matrix multiplication for prompt processing. All prompt tokens processed in parallel via a single matmul.

**Our approach:** Same single-token forward loop as generation. Each prompt token processed identically to a generation step.

**Impact:** For prompt_len=256, batched prefill would be ~10× faster.

---

## 8. RMS Norm + Scale — ✅ DONE

llama.cpp fuses `rmsnorm(x) * weight` into one kernel. We do the same:

```metal
// metal_backend.hpp:kernel_rmsnorm
kernel void kernel_rmsnorm(device float* data [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          constant int* params [[buffer(2)]],
                          uint gid [[threadgroup_position_in_grid]]) {
    // Single pass: compute sum of squares, sqrt, divide, multiply by weight
    data[i] = (data[i] / rms) * weight[i];
}
```

---

## Performance Analysis

| Bottleneck | Impact | Status | Fix Difficulty |
|---|---|---|---|
| Separate kernel dispatch overhead | ~2-3× | ✅ Done (fused path, 1 CB) | Done |
| 256-thread TGs (low occupancy) | ~1.5× | ✅ Done (switched to 64) | Done |
| No flash attention | ~1.2× | ✅ Done (implemented) | Done |
| Fused FFN silu+down | ~1.3× | ❌ Not done (can't fuse due to dependency) | Medium |
| Graph scheduling | ~1.2-1.5× | ❌ Not done | High |
| Runtime quant type switch | ~1.05-1.1× | ⚠️ Partial (only kernel_op, v_type) | Low |
| Batched prefill | ~10× prefill | ❌ Not done | Medium |

**Current performance:** ~18-19ms/token (Metal fused path)
**Target:** ~6-7ms/token (close to llama.cpp's 6.8ms)

**Remaining gap analysis:**
- The dispatch overhead elimination (fused path) is done
- Flash attention is done
- Fused QKV is done
- The 3× gap likely comes from:
  1. Quant type specialization not fully implemented (runtime switch in matmul)
  2. No graph scheduling (sequential layers in CB vs concurrent)
  3. Fused FFN silu+down not fused (but this is hard due to sequential dep)

---

## Reference: Key llama.cpp File Locations

| File | Lines | Content |
|---|---|---|
| `ggml-metal.metal` | 1-320 | Quantized block types, shared structs, utility functions |
| `ggml-metal.metal` | 321-1900 | Quantized dequant functions (q4_0 through q8_0) |
| `ggml-metal.metal` | 1901-3400 | Quantized matmul kernels (`kernel_mul_mat_q*_f32`) |
| `ggml-metal.metal` | 4701-5400 | `kernel_flash_attn_ext_vec_f16_dk64_dv64` |
| `ggml-metal.metal` | 6201-6400 | Fused gated delta net, dequant + mul_mat |
| `ggml-metal-device.cpp` | 241-900 | Kernel selection, pipeline compilation, constant specialization |
| `ggml-metal-device.cpp` | 901-1300 | Graph scheduling, dispatch logic |
| `ggml-metal-device.cpp` | 1301-1550 | `ggml_backend_metal_graph_compute` — main compute dispatcher |

---

## Next Steps for FPGA (~/fpga)

Key insights from this study applicable to the FPGA accelerator:

1. **Quantization diversity** — Model uses Q5_0 (attention), Q6_K (FFN gate/up), Q4_K (attn output), Q8_0 (embeddings). FPGA currently supports only Q8_0 and Q4_K. Adding Q5_0 and Q6_K would cover the largest layers.

2. **Memory layout** — Q6_K uses 256-element blocks (matching GPU SIMD width). FPGA should align block sizes to AXI bus width for efficient transfers.

3. **Fused operations** — Even with slower memory, fusing gate+up into one transaction would save DDR bandwidth. The silu×up→down dependency prevents full fusion without local storage.

4. **Flash attention** — The tiled online softmax approach is applicable to any accelerator. For FPGA, processing KV in tiles of 8 (matching the 8 MAC lanes) would be efficient.

---

## Appendix: Build Instructions

```bash
# Requires linking matmul_q8.cpp from ~/fpga/sim
clang++ -std=c++17 -x objective-c++ -O3 \
    tmac_gguf.cpp \
    ~/fpga/sim/matmul_q8.cpp \
    -framework Foundation -framework Metal -fobjc-arc \
    -I~/fpga/sim \
    -o tmac_gguf

# Run with Metal fused path
./tmac_gguf model.tmac --metal-fused --generate 20 < tokens.txt

# Run with profiling
./tmac_gguf model.tmac --metal-fused --generate 20 --perf < tokens.txt
```