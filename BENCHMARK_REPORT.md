# Benchmark: tmac_gguf.cpp — Wall-Clock Measurements

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen2-0.5B-Instruct (151936 vocab, 896 hidden, 24 layers) |
| **Quant Mix** | Q5_0 (132), Q4_K (12), Q6_K (12), Q8_0 (13), F32 (121) |
| **Hardware** | Apple M1 Pro (16 GPU cores, 200 GB/s unified bandwidth) |
| **Backend** | Metal GPU (`--metal-fused` — fused path, 2 command buffers) |
| **Measurement** | Wall-clock via `[BENCH]` timer, 30 gen tokens, 4-token prompt |
| **Warmup** | 0 tokens of warmup (cold start) |

## Results

| Metric | tmac_gguf (measured) |
|--------|:--------------------:|
| **Per-token generation** | **16.9 ms** |
| **Throughput** | **59 tok/s** |
| **Total GPU time (24 layers)** | ~15.6 ms (steady-state: ~650 μs/layer) |
| **Logits matmul (151936×896)** | ~530 μs |
| **CPU encoding + overhead** | ~0.8 ms |

### Comparison with Projections

| Metric | Projected (prior) | Measured | Delta |
|--------|:-----------------:|:--------:|:-----:|
| ms/tok | 6.6 | **16.9** | 2.6× slower |
| tok/s | 150 | **59** | 2.5× less |

The 6.6 ms projection was optimistic: it assumed all 7 matmuls per layer run in parallel on 16 GPU cores. In reality, per-layer ops are data-dependent and execute sequentially within a single command buffer.

## Per-Layer GPU Time Breakdown

Measured via `--perf` (per-layer CB timing, steady-state after ~12 dispatches):

| Layer | GPU time |
|:-----:|:--------:|
| Cold (first 3-4 tokens) | ~1000 μs |
| Steady-state | ~650 μs |

### Composition of ~650 μs per layer (~21 dispatches)

| Dispatch type | Count | Approx. GPU time |
|--------------|:-----:|:----------------:|
| Quant matmuls (Q, K, V, attn_out, gate, up, down) | 7 | ~330 μs |
| Element-wise (norm×2, bias×3, rope×2, copy×2, silu, residual×2, KV cache×2) | ~13 | ~200 μs |
| Flash attention (online softmax, seq_len=4) | 1 | ~120 μs |

## Per-Operator Speedup (Branchless SIMD)

| Kernel | Quant | Dimensions | Before | After | Speedup |
|--------|-------|-----------|--------|-------|---------|
| ffn_gate | Q5_0 | 4864×896 | 552 μs | 170 μs | **3.2×** |
| attn_q | Q5_0 | 896×896 | 218 μs | 67 μs | **3.3×** |
| ffn_down | Q4_K | 4864×896 | 575 μs | 170 μs | **3.4×** |

Measured via `--perf-granular` GPUStartTime/GPUEndTime after 12+ warmup passes.

## Warmup Profile

Branchless SIMD kernels stabilize after ~12 GPU dispatches:

| Pass | ffn_gate (Q5_0) | attn_q (Q5_0) | ffn_down (Q4_K) |
|------|:---------------:|:--------------:|:----------------:|
| Cold | 552 μs | 218 μs | 575 μs |
| 3 | 333 μs | 135 μs | 340 μs |
| 5 | 278 μs | 103 μs | 282 μs |
| 7 | 224 μs | 85 μs | 230 μs |
| 12+ | **170 μs** | **67 μs** | **170 μs** |

## Key Architectural Differences vs llama.cpp

| Aspect | tmac_gguf | llama.cpp |
|--------|-----------|-----------|
| **Command buffers / forward** | 2 (1 layers + 1 logits) | 2-3 |
| **Threadgroup size (mat-vec)** | 256 (8 SG) | 64 (2 SG) |
| **Elements/thread/block (Q5_0)** | 1 | 16 |
| **Attention** | Flash attention (GPU, single-pass) | Flash attention (GPU, tiled) |
| **Cross-SG reduction** | simd_sum | simd_sum + shmem + simd_sum |
| **Branchless nibble** | Yes (Q5_0, Q4_K, Q6_K) | No (ternary branches) |
| **Encoding** | Synchronous, single-threaded | Parallel via GCD (dispatch_apply) |
| **Kernel specialization** | Runtime branching | Function constants (compile-time) |

## Measurement Methodology

```bash
# Wall-clock (always-on [BENCH] timer, no --perf overhead):
./tmac_gguf ~/fpga/models/model.tmac --metal-fused --generate 30 < tokens.txt

# Expected output:
# [BENCH] Generated 30 tokens in 507.0 ms — 16.9 ms/tok, 59 tok/s

# Per-layer GPU breakdown (adds ~1% overhead from 24 separate CBs):
./tmac_gguf ~/fpga/models/model.tmac --metal-fused --perf --generate 30 < tokens.txt
# Read [GPU PER-CB TIMING] for per-layer GPUStartTime/GPUEndTime

# Per-op GPU timing (adds ~2-5× overhead from per-op commit+wait):
./tmac_gguf ~/fpga/models/model.tmac --metal-fused --perf-granular --generate 30 < tokens.txt
# Read [GPU PER-CB TIMING] for individual operation timestamps
```
