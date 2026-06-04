# tmac_gguf.cpp — Metal GPU-Accelerated Qwen2-0.5B Inference

High-performance inference engine for Qwen2-0.5B-Instruct on Apple Silicon Metal GPU, with companion FPGA accelerator (`~/fpga`).

## Quick Start

```bash
# Build (requires ~/fpga/sim for logits projection)
clang++ -std=c++17 -x objective-c++ -O3 tmac_gguf.cpp metal_backend.cpp \
    ~/fpga/sim/matmul_q8.cpp \
    -framework Foundation -framework Metal -fobjc-arc \
    -I~/fpga/sim -o tmac_gguf

# Run
echo "0 1 2 3" | ./tmac_gguf model.tmac --metal-fused --generate 30
```

## Run Modes

| Flag | Path | Sync Points | Speed |
|------|------|:-----------:|:-----:|
| `--metal` | Per-layer CB (4 CBs/layer) | 96 commit+wait | ~71 ms/tok |
| `--metal-fused` | **Single CB for all layers** | **1 commit+wait** | **~17 ms/tok** |
| `--fpga-q8` | CPU FPGA simulation (INT8) | — | Slow |
| (none) | CPU matmul | — | ~2.9 s/tok |

### Profiling Flags

| Flag | Effect | Overhead |
|------|--------|:--------:|
| `--perf` | Chrome trace + per-layer CB timing | ~15-40% |
| `--perf-granular` | Per-op GPU timing via GPUStartTime/GPUEndTime | ~2-5× |
| `--gpu-capture` | Xcode GPU frame capture | Minimal |

## Model: Qwen2-0.5B-Instruct

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Hidden dim | 896 |
| Query heads | 14 |
| KV heads (GQA) | 2 |
| Head dim | 64 |
| FFN intermediate | 4864 |
| Vocab | 151936 |
| Max seq len | 256 |

### Quantization Mix (all tensors converted to Q8_0 at load time)

| Original Type | Count | Usage |
|:------------:|:-----:|-------|
| Q5_0 | 132 | QKV projections, FFN gate/up |
| Q4_K | 12 | Attention output, FFN down (half) |
| Q6_K | 12 | FFN down (half) |
| Q8_0 | 13 | Token embeddings, output projection |
| F32 | 121 | Norm weights, bias |

## Metal Kernels

| Kernel | Type | Description |
|--------|------|-------------|
| `mul_mat_q8_0` | Q8_0 | Scalar matvec, 4 rows/TG, 64 threads |
| `mul_mat_q8_0_simd` | Q8_0 | simd_sum-based (standalone test only) |
| `mul_mat_q8_0_simdtc` | Q8_0 | **simdgroup_float8x8 tensor core**, 8 rows/TG |
| `mul_mat_q5_0` | Q5_0 | Branchless nibble extraction |
| `mul_mat_q4_k` | Q4_K | 256-element blocked, branchless |
| `mul_mat_q6_k` | Q6_K | 256-element blocked |
| `kernel_fused_qkv` | Q5_0/Q8_0 | Fused Q+K+V matmul |
| `kernel_attn` | — | GQA attention, online softmax, 1 head/TG |
| `kernel_flash_attn` | — | **Flash attention**, GQA-correct, 8 heads/TG |
| `kernel_rope` | — | Rotary position embedding |
| `kernel_rmsnorm` | — | RMS normalization |
| `kernel_elem` | — | Add, SiLU×up, cache-write |

## Performance (M1 Pro, fused path, 30 gen tokens)

| Metric | Value |
|--------|:-----:|
| **Per-token wall-clock** | **16.9 ms** |
| **Throughput** | **59 tok/s** |
| GPU time (24 layers) | ~15.6 ms (~650 μs/layer) |
| Logits matmul (151936×896) | ~530 μs |
| CPU encoding overhead | ~0.8 ms |

**Bottleneck**: GPU compute throughput (~12% bandwidth utilization). Within each layer, ~21 dispatches execute sequentially due to data dependencies.

## Project Structure

```
tmac_gguf.cpp          — Main inference engine
metal_backend.cpp      — Metal init + pipeline loading
metal_backend.hpp      — Dispatch helpers, batch infrastructure
kernels/               — 15 Metal Shading Language kernel files
  mul_mat_q8_0.metal   — Scalar Q8 matvec
  mul_mat_q8_0_simd.metal      — simd_sum Q8 (reference)
  mul_mat_q8_0_simdtc.metal    — Tensor core Q8 (production)
  mul_mat_q5_0.metal, mul_mat_q4_k.metal, mul_mat_q6_k.metal
  kernel_flash_attn.metal, kernel_attn.metal
  kernel_fused_qkv.metal, kernel_elem.metal, ...
test_*.mm              — Standalone test programs (kernel validation)
scripts/               — TMAC converter, profiling script
ARCHITECTURE.md        — Design, execution flow, kernel internals
BENCHMARKS.md          — Performance data, profiling methodology
LLAMACPP_METAL_STUDY.md — Comparative analysis vs llama.cpp
```

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — inspiration for quantization formats and Metal backend
- [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) — model architecture
- `~/fpga` — companion FPGA accelerator with Verilog RTL
