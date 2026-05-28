# tmac_gguf.cpp — Metal GPU-Accelerated Qwen2-0.5B Inference

A high-performance Metal GPU inference engine for Qwen2-0.5B-Instruct, inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp). Built to explore GPU optimization techniques that can inform an FPGA accelerator in a companion project (`~/fpga`).

## Purpose

This project serves as a **performance optimization playground** for model inference:

1. **Exercise Metal GPU optimization** — implement and benchmark techniques from llama.cpp on Apple Silicon
2. **Validate against a reference** — ensure correctness via CPU baseline before optimizing
3. **Collect insights for FPGA** — the techniques developed here inform the hardware-software co-design in `~/fpga`

The companion FPGA project (`~/fpga`) implements a Zynq 7010 accelerator with Verilog RTL. The quantization formats (Q5_0, Q6_K, Q8_0, Q4_K) and matmul patterns here mirror what the FPGA needs to support.

## Model

**Qwen2-0.5B-Instruct** — 24 layers, 896 hidden dim, 14 heads (GQA with 2 KV heads), 64 head dim.

Quantized formats used:
- **Q5_0**: attention Q/K/V weights
- **Q5_0**: attention Q/K/V weights, FFN gate/up weights
- **Q6_K**: FFN down weights (half of layers, 896 × 4864)
- **Q4_K**: attention output projection, FFN down weights (half of layers)
- **Q4_K**: attention output projection
- **Q8_0**: token embeddings, output projection

## Features

- [x] 5 quantization types (Q5_0, Q6_K, Q8_0, Q4_K)
- [x] KV-cached autoregressive generation
- [x] Metal GPU acceleration with command buffer batching
- [x] **Fused QKV matmul** — single dispatch computes Q, K, V together (~6% speedup)
- [x] Chrome trace profiling (`--perf`)
- [x] CPU fallback path (verifies correctness)
- [x] **Fused FFN gate+up matmul** — single dispatch for Q5_0 gate+up (FFN gate/up are Q5_0 in Qwen2-0.5B)
- [ ] Flash attention (studied but not implemented)

## Build

```bash
# Requires linking matmul_q8.cpp from ~/fpga/sim (provides logits projection)
clang++ -std=c++17 -x objective-c++ -O3 \
    tmac_gguf.cpp \
    ~/fpga/sim/matmul_q8.cpp \
    -framework Foundation -framework Metal -fobjc-arc \
    -I~/fpga/sim \
    -o tmac_gguf
```

## Run

```bash
# CPU baseline
./tmac_gguf model.tmac --generate 20 < tokens.txt

# Metal with per-layer sync (4 command buffers per layer)
./tmac_gguf model.tmac --metal --generate 20 < tokens.txt

# Metal fused path (1 command buffer for entire forward pass) — RECOMMENDED
./tmac_gguf model.tmac --metal-fused --generate 20 < tokens.txt

# With profiling
./tmac_gguf model.tmac --metal-fused --generate 20 --perf < tokens.txt
```

## Performance

Single-token generation latency on M1 Pro:

| Path | ms/token | Notes |
|------|----------|-------|
| CPU | ~21ms | 24 layers × 4 CPU sync points |
| Metal (per-layer) | ~19ms | 4 CBs/layer × 24 layers |
| Metal (fused QKV) | ~18ms | 1 CB, saves 48 dispatches |

The fused QKV path saves 2 dispatch operations per layer (Q, K, V matmuls → 1 fused dispatch).

## Architecture

### Metal Kernels

Each quantization type has a specialized SIMD matmul kernel using 64-thread threadgroups (2 SIMD groups × 32 lanes):

| Kernel | Quantization | Notes |
|--------|-------------|-------|
| `mul_mat_q8_0` | Q8_0 | 34 bytes/block, 32 elements |
| `mul_mat_q5_0` | Q5_0 | 22 bytes/block, 32 elements |
| `mul_mat_q4_k` | Q4_K | Blocked, 2×16 sub-blocks |
| `mul_mat_q6_k` | Q6_K | 210 bytes/block, 256 elements |
| `kernel_rope` | — | In-place rotary embedding |
| `kernel_rmsnorm` | — | RMS normalization |
| `kernel_attn` | — | GQA attention with softmax |
| `kernel_elem` | — | Add, SiLU×up, cache-write |
| `kernel_fused_qkv` | Q5_0/Q8_0 | Fused Q+K+V matmul |

### Command Buffer Batching

Two modes:

1. **Per-layer sync** (`--metal`): 4 command buffers per layer, CPU sync after each
2. **Fused path** (`--metal-fused`): 1 command buffer for entire forward pass, zero CPU sync

## Insights for FPGA Accelerator

Key optimization techniques from this project applicable to `~/fpga`:

1. **Fused operations** — combining multiple independent matmuls reduces protocol overhead
   - On GPU: dispatch overhead ~30µs per kernel
   - On FPGA: AXI transaction setup similarly costs cycles

2. **Quantization diversity** — the model uses 4+ quantization types; the FPGA supports only Q8_0 and Q4_K
   - Missing: Q5_0 (attention Q/K/V), Q6_K (FFN gate/up)
   - These are the largest layers and biggest performance opportunities

3. **Batch dispatch** — encoding multiple operations into one batch reduces per-op overhead
   - GPU: one command buffer vs. 96 commit/wait cycles
   - FPGA: similar gains from batching AXI transactions

4. **Memory layout** — Q6_K uses 256-element blocks (matching GPU SIMD width); FPGA memory access patterns should align to bus width

## Project Structure

```
tmac_gguf.cpp          — Main inference engine
metal_backend.hpp      — Metal kernels + dispatch helpers (embedded MSL source)
fpga_sim.hpp           — Symlink to ~/fpga/sim/fpga_sim.hpp
DESIGN.md              — Detailed design document
LLAMACPP_METAL_STUDY.md — Analysis of llama.cpp techniques
FLOW.md                — Forward pass data flow
ANALYSIS.md            — Performance analysis
```

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — inspiration for quantization formats and Metal backend
- [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) — model architecture
- `~/fpga` — companion FPGA accelerator project with Verilog RTL implementation