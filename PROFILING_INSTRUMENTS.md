# Performance Analysis & Profiling Instruments

This document systematically catalogs the performance analysis, tracking, and profiling instruments used in this project.

## Overview

| Instrument | Type | Enable Flag | Output | Runtime Impact |
|------------|------|-------------|--------|----------------|
| Chrome Trace (CPU) | Sampling profiler | `--perf` | CPU timestamps | ~15-40% slower (layer_cb_size=1) |
| GPU CB Timing | Per-dispatch timing | `--perf-granular` | GPU exec time + CB labels | ~2-5× slower (per-op sync) |
| GPU Frame Capture | Xcode Instruments | `--gpu-capture` | Xcode GPU debugger | Minimal |

---

## 1. Chrome Trace Profiler (`--perf`)

**Source:** `tmac_gguf.cpp` (lines 16-96)

A custom CPU-side tracing infrastructure that generates Chrome-compatible trace JSON files for visualization in [Chrome Trace Viewer](https://ui.perfetto.dev/).

### Trace Event Structure

```cpp
struct TraceEvent {
    const char* name;
    int64_t ts_us;
    char ph; // 'B' = begin, 'E' = end
};
```

### Global State

```cpp
static std::vector<TraceEvent> g_trace_events;
static bool g_perf_enabled = false;
static bool g_perf_granular = false;
```

### Key Functions

- `trace_begin(name)` / `trace_end(name)` - Record begin/end events
- `dump_chrome_trace(path)` - Writes JSON trace to `/tmp/pipeline_trace.json`
- `print_trace_summary()` - Aggregates and prints timing statistics

### RAII Scoping Macros

#### `--perf` Mode
- `PROFILE_SCOPE(name)` - Records CPU timestamps on scope enter/exit

```cpp
struct TraceScope {
    TraceScope(const char* name) : name_(name) { trace_begin(name_); }
    ~TraceScope() { trace_end(name_); }  // CPU-only, no GPU sync
};
```

#### `--perf-granular` Mode (Additional GPU Synchronization)
- `PROFILE_SCOPE(name)` - Commits Metal command buffer with labeled checkpoint, waits for GPU completion

```cpp
struct TraceScope {
    TraceScope(const char* name) : name_(name) { trace_begin(name_); }
    ~TraceScope() {
        if (g_perf_granular && g_mtl_ctx_ptr) {
            metal_backend::metal_batch_checkpoint(g_mtl_ctx_ptr, strdup(name_));
        }
        trace_end(name_);
    }
};
```

- `PROFILE_SCOPE_GPU(name)` - Explicit GPU batch commit/wait cycle

```cpp
struct TraceScopeGPU {
    TraceScopeGPU(const char* name) {
        if (g_perf_granular) metal_backend::metal_batch_commit();
    }
    ~TraceScopeGPU() {
        if (g_perf_granular) {
            metal_backend::metal_batch_wait_all();
            g_trace_events.push_back({name_, now_us(), 'B'});
            g_trace_events.push_back({name_, now_us(), 'E'});
        }
    }
};
```

### Critical: `--perf` / `--perf-granular` Change Execution Behavior

Both flags change `layer_cb_size` in the fused forward path:

| Flag | `layer_cb_size` | CBs per forward pass | Impact |
|------|:---------------:|:--------------------:|--------|
| None | `NUM_LAYERS` (1 CB for all 24 layers) | 1 | **Maximum fusion** — no inter-layer sync |
| `--perf` | 1 (per layer) | 24 + 1 logits = 25 | ~15-40% slower; enables per-layer breakdown |
| `--perf-granular` | 1 (per layer) + per-op checkpoint | 25 + per-op commits | **Significantly slower** (2-5×); enables per-op GPU trace |

**This means profiling results are NOT representative of normal execution performance.** The `--perf` flag trades performance for granularity.

### Key Difference

| Aspect | `--perf` | `--perf-granular` |
|--------|----------|-------------------|
| GPU Sync | None (batched wait_all at end) | Per-op CB commit + wait |
| Timing Accuracy | PROFILE_SCOPE timestamps | GPUEndTime - GPUStartTime per CB |
| CB Labeling | Per-layer labels | Per-operation labels |
| Overhead | ~15-40% slowdown | ~2-5× slowdown |
| layer_cb_size | 1 (per-layer) | 1 (per-layer) |
| Use Case | Per-layer breakdown | Micro-architecture GPU timing |

### Usage

```bash
# Basic CPU tracing
./tmac_gguf --perf --prompt "..."

# Granular GPU-aware profiling
./tmac_gguf --perf-granular --prompt "..."
```

View the trace:
1. Open [Chrome Trace Viewer](https://ui.perfetto.dev/)
2. Load `/tmp/pipeline_trace.json`

---

## 2. GPU Command Buffer Timing (`--perf`)

**Source:** `metal_backend.hpp` (lines 75-182)

Metal command buffer-level GPU timing using `GPUStartTime`/`GPUEndTime` timestamps captured by the Metal Profiler.

### Timing State

```cpp
static bool g_trace_enabled = false;
static const char* g_cb_labels[MAX_CBS];
static double g_cb_ms[MAX_CBS];
```

### Key Functions

- `metal_trace_begin()` - Initialize tracing
- `metal_trace_end()` - Stop tracing
- `metal_trace_record(cb)` - Record CB timing
- `metal_trace_report()` - Print per-CB timing summary
- `metal_batch_checkpoint()` - Commit CB and record GPU timing
- `metal_batch_wait_all()` - Wait for committed CBs and record timings

### Timing Calculation

```cpp
g_cb_ms[g_cb_count] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
```

### GPU Warmup Effect

The `--perf-granular` per-op GPU timing is sensitive to GPU warmup state. Branchless SIMD kernels require ~12 warmup passes to reach steady-state performance:

| Pass | ffn_gate (Q5_0 4864×896) | attn_q (Q5_0 896×896) |
|------|-------------------------|----------------------|
| Cold | 552 μs | 218 μs |
| 3    | 333 μs | 135 μs |
| 5    | 278 μs | 103 μs |
| 7    | 224 μs | 85 μs |
| 12+  | **170 μs** (3.2×) | **67 μs** (3.3×) |

This warmup is GPU instruction/cache warmup — branchless code gets cached in the GPU's instruction cache, and the SIMD pipeline fills optimally after a few dispatches.

---

## 3. GPU Frame Capture (`--gpu-capture`)

**Source:** `metal_backend.hpp` (lines 994-1007)

Uses Apple's Metal Profiler API to capture GPU traces for Xcode Instruments GPU Debugger.

### Capture Function

```cpp
inline bool capture_start(Context& ctx) {
    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
    if (![mgr supportsDestination:MTLCaptureDestinationDeveloperTools]) return false;
    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = ctx.device;
    desc.destination = MTLCaptureDestinationDeveloperTools;
    NSError* err;
    return [mgr startCaptureWithDescriptor:desc error:&err];
}
```

### Usage

```bash
./tmac_gguf --gpu-capture --prompt "..."
```

1. Run the app with `--gpu-capture`
2. Open Xcode
3. Debug -> Attach to Process -> select the running process
4. Use GPU Debugger to analyze frames

---

## 4. FPGA Simulation Timing

**Source:** `tmac_gguf.cpp` (multiple locations)

Timing infrastructure for FPGA simulation path, tracking MAC operations, tiles, CPU time, and FPGA cycles.

### Tracked Metrics via `fpga_sim::g_timing`

| Metric | Description |
|--------|-------------|
| `total_mac_ops` | Total multiply-accumulate operations |
| `total_tiles` | Number of tile operations |
| `cpu_ms` | CPU time in milliseconds |
| `total_fpga_cycles` | FPGA cycle count |

### Locations

- MAC ops tracking: lines 542, 606, 752, 830, 906
- Tile tracking: lines 577, 721, 801, 879, 975
- CPU timing: lines 583, 727, 811, 888, 985
- FPGA cycles: line 1886

### Reporting

```cpp
fpga_sim::g_timing.report();
```

---

## 5. High-Resolution Clock Timing

**Source:** `tmac_gguf.cpp` (line 32-36)

Uses `std::chrono::high_resolution_clock` for precise CPU-side timing.

```cpp
static int64_t now_us() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}
```

---

## 6. Metal Trace Sampling

**Source:** `metal_backend.hpp`

Periodic sampling points in the Metal backend for performance analysis.

### Sampling Locations

- Line 280: In elem_op dispatch
- Lines 1065, 1080, 1096: In fused forward path
- Lines 1119, 1137, 1171, 1196: In layer processing

---

## 7. Integration Points

### Initialization (tmac_gguf.cpp:1829)

```cpp
if (g_perf_enabled) metal_backend::metal_trace_begin();
```

### Finalization (tmac_gguf.cpp:1887-1896)

```cpp
if (g_perf_enabled) {
    if (g_use_metal) {
        metal_backend::metal_trace_end();
        metal_backend::metal_trace_report();
    }
    print_trace_summary();
    dump_chrome_trace("/tmp/pipeline_trace.json");
}
```

---

## 8. CLI Flags Reference

| Flag | Description | Performance Impact |
|------|-------------|-------------------|
| `--perf` | Chrome trace + GPU CB timing | ~15-40% slower (layer_cb_size: all → per-layer) |
| `--perf-granular` | Per-op GPU timing via GPUStartTime/GPUEndTime | ~2-5× slower (per-op commit+wait) |
| `--gpu-capture` | Xcode GPU frame capture | Minimal during capture |
| (none — always on) | End-to-end wall-clock timer around `generate()` | Negligible (2 chrono reads) |

**Note on warmup**: Branchless SIMD kernels require ~12 GPU dispatches to reach steady-state performance. The first tokens run at pre-optimization speed. Use multiple warmup passes before measuring.

---

## 9. End-to-End Wall-Clock Timer (`[BENCH]`)

**Source:** `tmac_gguf.cpp` (main, around `generate()` call)

Added to always print real generation throughput without requiring `--perf`:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
generate(hidden, logits, (int)tokens.size(), generate_n, 40);
auto t1 = std::chrono::high_resolution_clock::now();
double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
printf("\n[BENCH] Generated %d tokens in %.1f ms — %.1f ms/tok, %.0f tok/s\n",
       generate_n, total_ms, total_ms / generate_n, generate_n / (total_ms / 1000.0));
```

### Output Example (from actual M1 Pro run)
```
[BENCH] Generated 30 tokens in 507.0 ms — 16.9 ms/tok, 59 tok/s
```

### How to Use
```bash
# Basic measurement (no --perf, no overhead)
./tmac_gguf ~/fpga/models/model.tmac --metal-fused --generate 30 < tokens.txt

# With --perf (per-layer CB breakdown, negligible overhead)
./tmac_gguf ~/fpga/models/model.tmac --metal-fused --perf --generate 30 < tokens.txt
# Read [GPU PER-CB TIMING] for per-layer GPUStartTime/GPUEndTime
```

### Measured Results (M1 Pro, Qwen2-0.5B, fused path)

| Metric | Value | Source |
|--------|:-----:|--------|
| **Per-token wall-clock** | **16.9 ms** | `[BENCH]` timer (30 tokens, no `--perf`) |
| **Throughput** | **59 tok/s** | `[BENCH]` timer |
| **GPU time per layer** | ~650 μs (steady-state) | `--perf` GPU per-CB timing |
| **Total GPU time** | ~16.1 ms | 24 × 650μs + logits 530μs |
| **CPU encoding overhead** | ~0.8 ms | forward_and_logits_fused CPU timing minus GPU time |

**Breakdown**: Within each ~650μs layer CB, ~21 dispatches execute sequentially (data-dependent). The 7 quant matmuls account for ~330μs, element-wise ops ~200μs, flash attention ~120μs. There is no hidden 90% CPU overhead — the old bottleneck analysis described the non-fused path that used per-layer commit+wait cycles.

---

## 10. Related Documentation

- `ANALYSIS.md` - Performance Journey, Bottleneck Analysis, Root cause analysis
- `BENCHMARK_REPORT.md` - Measured wall-clock benchmark data
- `LLAMACPP_METAL_STUDY.md` - Comparative analysis vs llama.cpp, Threadgroup sizing optimization

---

## 11. Build Configuration

- Optimization: `-O3`
- No explicit `-pg` compile flags
- Profiling achieved through runtime flags, not compile-time instrumentation
