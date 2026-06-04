# Benchmarks & Performance Analysis

**Hardware**: Apple M1 Pro (16 GPU cores, 200 GB/s unified bandwidth)
**Model**: Qwen2-0.5B-Instruct (151936 vocab, 896 hidden, 24 layers)

---

## 1. Current Measurements

| Metric | Value |
|--------|:-----:|
| **Per-token wall-clock** | **16.9 ms** |
| **Throughput** | **59 tok/s** |
| **GPU time (24 layers)** | ~15.6 ms (~650 μs/layer) |
| **Logits matmul (151936×896)** | ~530 μs |
| **CPU encoding overhead** | ~0.8 ms |

Measured: `[BENCH]` timer, fused single-CB forward pass, 30 gen tokens, no `--perf`.

### Per-Layer GPU Time Breakdown (~650 μs)

| Dispatch type | Count | GPU time |
|--------------|:-----:|:--------:|
| Quant matmuls (Q, K, V, attn_out, gate, up, down) | 7 | ~330 μs |
| Element-wise (norm×2, bias×3, rope×2, copy×2, silu, residual×2, KV cache×2) | ~13 | ~200 μs |
| Flash attention (seq_len small) | 1 | ~120 μs |

### Comparison with Projections

| Metric | Projected (prior) | Measured | Delta |
|--------|:-----------------:|:--------:|:-----:|
| ms/tok | 6.6 | **16.9** | 2.6× slower |
| tok/s | 150 | **59** | 2.5× less |

The 6.6ms projection assumed matmuls within a layer run in parallel on 16 GPU cores. In reality, ops are data-dependent and execute sequentially.

---

## 2. Performance Journey

| Step | Change | Gen Throughput | Per-Tok | Notes |
|:----:|--------|:--------------:|:-------:|-------|
| — | CPU baseline (-O3) | ~0.3 tok/s | ~2.94s | Reference |
| 1 | Scalar Metal kernels | — | — | One-thread-scan-all-rows |
| 2 | SIMD kernels (32-thread) for Q4_K/Q5_0/Q6_K | ~5 t/s | ~200ms | Batch-size-1 per matmul |
| 3 | + CB batching (QKV/gate+up grouped) | ~5 t/s | ~200ms | 7→4 commits/layer |
| 4 | + Q8_0 SIMD + buffer caching + params pool | ~11 t/s | ~100ms | Last quant type ported |
| 5 | + 64-thread TGs (2×32, 4 rows/TG) | ~17 t/s | ~60ms | +40% from threadgroup sizing |
| 6 | + Fused forward pass + fused QKV | — | — | Single-CB forward |
| **7** | **+ Branchless SIMD nibble extraction** | **~59 t/s** | **~16.9ms** | **3.2× per-op** |

### Bottleneck Analysis

- **GPU compute throughput** is the bottleneck (~92% of time)
- Bandwidth utilization: ~12% (374 MB weights / 200 GB/s peak = 1.9ms theoretical read time; actual GPU time ~16.1ms)
- Sequential execution within each layer prevents parallel GPU core utilization

### Theoretical Minimum

| Component | Time |
|-----------|:----:|
| Read 374 MB at 200 GB/s | 1.9 ms |
| Attention (GPU flash) | ~0.3 ms |
| Element-wise ops | ~0.1 ms |
| Dispatch overhead | ~0.5 ms |
| **Theoretical** | **~2.8 ms (~357 t/s)** |

Current gap from theoretical: **6×** — primarily due to sequential matmuls within layers.

---

## 3. Per-Operator Speedup (Branchless SIMD)

| Kernel | Quant | Dimensions | Before | After | Speedup |
|--------|-------|-----------|--------|-------|:-------:|
| ffn_gate | Q5_0 | 4864×896 | 552 μs | 170 μs | **3.2×** |
| attn_q | Q5_0 | 896×896 | 218 μs | 67 μs | **3.3×** |
| ffn_down | Q4_K | 4864×896 | 575 μs | 170 μs | **3.4×** |

Measured via `--perf-granular` GPUStartTime/GPUEndTime after 12+ warmup passes.

### Warmup Profile

Branchless SIMD kernels stabilize after ~12 GPU dispatches (instruction/cache warmup):

| Pass | ffn_gate (Q5_0) | attn_q (Q5_0) | ffn_down (Q4_K) |
|:----:|:---------------:|:--------------:|:----------------:|
| Cold | 552 μs | 218 μs | 575 μs |
| 3 | 333 μs | 135 μs | 340 μs |
| 5 | 278 μs | 103 μs | 282 μs |
| 7 | 224 μs | 85 μs | 230 μs |
| 12+ | **170 μs** | **67 μs** | **170 μs** |

---

## 4. Next Steps (Ranked by Impact)

1. **Extend simdgroup matmul to Q5_0/Q4_K/Q6_K** — Currently only Q8_0 uses simdgroup_float8x8 tensor core. Adapting for other quant types could provide 2-4× on largest matmuls (4864×896).
2. **Reduce dispatch count** — Fused FFN gate+up (Q5_0) already merged; further fusing possible.
3. **Multi-CB async encoding** — Overlap CPU encoding with GPU execution via GCD dispatch_apply.
4. **Flash attention at long context** — Current flash attention benefit compounds at seq_len > 4096.

---

## 5. Profiling Instruments

### Flag Reference

| Flag | Effect | Overhead |
|------|--------|:--------:|
| (none) | Wall-clock `[BENCH]` timer | Negligible |
| `--perf` | Chrome trace + per-layer CB timing | ~15-40% (layer_cb_size=1) |
| `--perf-granular` | Per-op GPU timing via GPUStartTime/GPUEndTime | ~2-5× (per-op sync) |
| `--gpu-capture` | Xcode GPU frame capture | Minimal |

### Instrument Details

#### Chrome Trace Profiler (`--perf`)

CPU-side tracing generating `/tmp/pipeline_trace.json` for Chrome Trace Viewer.
- `PROFILE_SCOPE(name)` — RAII CPU timestamps
- `--perf-granular` adds Metal `metal_batch_checkpoint()` per operation → GPU sync
- View at https://ui.perfetto.dev/

**Critical**: Both `--perf` and `--perf-granular` change `layer_cb_size` from `NUM_LAYERS` to 1, splitting the fused forward pass into per-layer command buffers. Results are NOT representative of normal execution.

#### GPU Command Buffer Timing

Uses Metal `GPUStartTime`/`GPUEndTime` timestamps captured by Metal Profiler:
```cpp
g_cb_ms[g_cb_count] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
```

Output: `[GPU PER-CB TIMING]` section per-layer breakdown.

#### GPU Frame Capture (`--gpu-capture`)

Apple Metal Profiler API capture for Xcode GPU Debugger:
```bash
./tmac_gguf --gpu-capture --prompt "..."
# Attach Xcode: Debug → Attach to Process
```

### Warmup Caution

First 3-4 tokens run at pre-optimization speed (~1000 μs/layer). Steady-state at ~650 μs/layer after ~12 dispatches. Use ≥12 warmup passes for stable measurements.

---

## 6. Key Differences vs llama.cpp

| Aspect | tmac_gguf | llama.cpp |
|--------|-----------|-----------|
| **Command buffers / forward** | 2 (1 layers + 1 logits) | 2-3 |
| **Threadgroup size (mat-vec)** | 64 (2 SG) | 64 (2 SG) |
| **Elements/thread/block (Q5_0)** | 1 | 16 |
| **Attention** | Flash attention (GPU, single-pass) | Flash attention (GPU, tiled) |
| **Cross-SG reduction** | simd_sum | simd_sum + shmem + simd_sum |
| **Branchless nibble** | Yes (Q5_0, Q4_K, Q6_K) | No (ternary) |
| **Encoding** | Synchronous, single-threaded | Parallel via GCD |
| **Kernel specialization** | Runtime branching | Function constants |

## Measurement Methodology

```bash
# Wall-clock (no overhead):
./tmac_gguf model.tmac --metal-fused --generate 30 < tokens.txt
# Output: [BENCH] Generated 30 tokens in 507.0 ms — 16.9 ms/tok, 59 tok/s

# Per-layer GPU breakdown (~1% overhead):
./tmac_gguf model.tmac --metal-fused --perf --generate 30 < tokens.txt
# Read [GPU PER-CB TIMING]

# Per-op GPU timing (~2-5× overhead):
./tmac_gguf model.tmac --metal-fused --perf-granular --generate 30 < tokens.txt
```
