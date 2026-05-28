// T-MAC Inference for Zynq 7010 - KV Cache Version

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include "fpga_sim.hpp"
#include "metal_backend.hpp"
// ===========================================================================
// CHROME TRACE EVENT PROFILER
// ===========================================================================
struct TraceEvent {
    const char* name;
    int64_t ts_us;
    char ph; // 'B' = begin, 'E' = end
};

static std::vector<TraceEvent> g_trace_events;
static bool g_perf_enabled = false;
static bool g_perf_granular = false;

namespace metal_backend { struct Context; }
static metal_backend::Context* g_mtl_ctx_ptr;

static int64_t now_us() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

static void trace_begin(const char* name) {
    if (!g_perf_enabled) return;
    g_trace_events.push_back({name, now_us(), 'B'});
}

static void trace_end(const char* name) {
    if (!g_perf_enabled) return;
    g_trace_events.push_back({name, now_us(), 'E'});
}

struct TraceScope {
    const char* name_;
    TraceScope(const char* name) : name_(name) { trace_begin(name_); }
    ~TraceScope() {
        if (g_perf_granular && g_mtl_ctx_ptr) {
            char label[64];
            snprintf(label, sizeof(label), "%s", name_);
            metal_backend::metal_batch_checkpoint(g_mtl_ctx_ptr, strdup(label));
        }
        trace_end(name_);
    }
};

#define PROFILE_SCOPE(name) TraceScope CONCAT(_trace_, __LINE__)(name)
#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

struct TraceScopeGPU {
    const char* name_;
    TraceScopeGPU(const char* name) : name_(name) {
        if (!g_perf_granular) return;
        metal_backend::metal_batch_commit();
    }
    ~TraceScopeGPU() {
        if (!g_perf_granular) return;
        metal_backend::metal_batch_wait_all();
        g_trace_events.push_back({name_, now_us(), 'B'});
        g_trace_events.push_back({name_, now_us(), 'E'});
    }
};
#define PROFILE_SCOPE_GPU(name) TraceScopeGPU CONCAT(_trace_gpu_, __LINE__)(name)

static void dump_chrome_trace(const char* path) {
    if (!g_perf_enabled || g_trace_events.empty()) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\"traceEvents\":[\n");
    for (size_t i = 0; i < g_trace_events.size(); i++) {
        auto& e = g_trace_events[i];
        fprintf(f, "{\"ph\":\"%c\",\"name\":\"%s\",\"cat\":\"pipeline\","
                "\"ts\":%lld,\"pid\":1,\"tid\":1}",
                e.ph, e.name, (long long)e.ts_us);
        if (i + 1 < g_trace_events.size()) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "]}\n");
    fclose(f);
    printf("[TRACE] Wrote Chrome trace: %s\n", path);
}

struct TraceAgg {
    const char* name;
    int64_t total_us;
    int64_t count;
};

static void print_trace_summary() {
    if (g_trace_events.empty()) return;

    // Build name → total duration map from begin/end pairs
    std::vector<TraceAgg> agg;
    auto find_agg = [&](const char* name) -> TraceAgg* {
        for (auto& a : agg) if (strcmp(a.name, name) == 0) return &a;
        return nullptr;
    };

    // Stack-based pairing
    struct Frame { const char* name; int64_t start_us; };
    Frame stack[64];
    int depth = 0;

    for (auto& e : g_trace_events) {
        if (e.ph == 'B') {
            if (depth < 64) stack[depth++] = {e.name, e.ts_us};
        } else if (e.ph == 'E' && depth > 0) {
            depth--;
            auto* a = find_agg(stack[depth].name);
            int64_t dur = e.ts_us - stack[depth].start_us;
            if (a) { a->total_us += dur; a->count++; }
            else agg.push_back({stack[depth].name, dur, 1});
        }
    }

    if (agg.empty()) return;

    // Sort by total time descending
    std::sort(agg.begin(), agg.end(),
              [](const TraceAgg& a, const TraceAgg& b) { return a.total_us > b.total_us; });

    int64_t total_us = 0;
    for (auto& a : agg) total_us += a.total_us;

    printf("\n[PERF — PIPELINE TIMING BREAKDOWN]\n");
    printf("  %-20s %12s %8s %10s %8s\n", "Operation", "Time (ms)", "Calls", "Avg (ms)", "Share");
    printf("  %s\n", std::string(65, '-').c_str());
    for (auto& a : agg) {
        double ms = a.total_us / 1000.0;
        double avg = ms / a.count;
        double pct = a.total_us * 100.0 / total_us;
        printf("  %-20s %10.2f %8lld %8.4f %6.1f%%\n",
               a.name, ms, (long long)a.count, avg, pct);
    }
    printf("  %s\n", std::string(65, '-').c_str());
    printf("  %-20s %10.2f\n", "TOTAL", total_us / 1000.0);

    if (!agg.empty()) {
        printf("\n  >> Hottest: %s (%.1f ms, %.1f%%)\n",
               agg[0].name, agg[0].total_us / 1000.0,
               agg[0].total_us * 100.0 / total_us);
    }
    printf("\n");
}

// ===========================================================================
// CONFIG - Qwen2-0.5B
// ===========================================================================
constexpr int HIDDEN_DIM = 896;
constexpr int INTER_DIM = 4864;
constexpr int VOCAB_SIZE = 151936;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_HEADS = 14;
constexpr int HEAD_DIM = 64;
constexpr int NUM_KV_HEADS = 2;
constexpr int K_DIM = NUM_KV_HEADS * HEAD_DIM;  // 128
constexpr int V_DIM = K_DIM;
constexpr int MAX_SEQ_LEN = 256;

constexpr size_t DDR_SIZE = 512 * 1024 * 1024;

// Tensor types
constexpr int TENSOR_F32 = 0;
constexpr int TENSOR_F16 = 1;
constexpr int TENSOR_Q8_0 = 8;
constexpr int TENSOR_Q6_K = 14;
constexpr int TENSOR_Q5_0 = 6;
constexpr int TENSOR_Q4_K = 12;

// ===========================================================================
// TENSOR STORAGE
// ===========================================================================
struct Tensor {
    char name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
};

static Tensor* g_tensors = nullptr;
static int g_ntensors = 0;
static uint8_t* g_ddr = nullptr;

// KV Cache
static float g_k_cache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
static float g_v_cache[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];
static int g_seq_len = 0;
static bool g_use_fpga = false;
static bool g_fpga_q8 = false;
static bool g_fpga_q4k = false;
static bool g_use_metal = false;
static bool g_gpu_capture = false;
static bool g_use_metal_fused = false;
static metal_backend::Context g_mtl_ctx;

// Cosimulation tile dump
static FILE* g_dump_file = nullptr;
static int g_dump_tiles_remaining = 0;
static constexpr int COSIM_TILE_SIZE = 4096 + 256 + 128 + 512; // weights + scales + vec + result

// ===========================================================================
// F16 TO F32
// ===========================================================================
float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    } else {
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (int)exp - 15);
    }
}

// ===========================================================================
// DEQUANTIZATION
// ===========================================================================

// Q8_0 block structure: [scale:FP16 (2B)][val:INT8 (32B)] per 32 elements
// Returns float value at element index idx
float dequant_q8_0(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t block_offset = block * 34;
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    float scale = f16_to_f32(scale_raw);
    int8_t val = (int8_t)data[block_offset + 2 + offset];
    return val * scale;
}

// FPGA Q8_0→INT16 dequantization: converts raw Q8_0 bytes to INT16 directly
// This simulates what the FPGA would do: per-32-element block, compute
// scale = max_abs / 32767, then val_int16 = round(val_q8 * 127 / max_abs)
// The FPGA would stream Q8_0 bytes and output INT16 without ever going to float.
// Returns the INT16 dequantized value.
inline fpga_sim::in16_t dequant_q8_to_int16(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t block_offset = block * 34;
    // Read scale (FP16) — same as Q8_0 decoder
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    (void)scale_raw;  // not used in FPGA path (FPGA recomputes max_abs per row)
    int8_t val_q8 = (int8_t)data[block_offset + 2 + offset];
    // Convert: Q8_0 is already scaled by scale, we want INT16 with new scale
    // New INT16 scale = max_abs / 32767 where max_abs comes from max Q8_0 value
    // Since Q8_0 stores val * scale, and we need val * new_scale where
    // new_scale = max_abs(Q8_0 values in block) / 32767
    // We approximate: scale_factor = 1.0 / scale * (max_abs / 32767)
    // For simplicity, FPGA would compute: val_int16 = round(val_q8 * (127/127))
    // which is just val_q8 directly if the scale is absorbed into the matmul scale.
    // Simpler still: the FPGA computes scale_inv = 32767.0 / computed_max
    // and multiplies each Q8_0 value directly.
    // Here we do the direct conversion: val * scale / new_scale, clamped to INT16 range.
    float max_abs = fabsf((float)val_q8);
    for (uint64_t i = 0; i < 32; i++) {
        int8_t v = (int8_t)data[block_offset + 2 + i];
        float a = fabsf((float)v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    float scale_factor = max_abs / 32767.0f;
    float val_f = (float)val_q8;  // Q8_0 val already in -127..127 range
    int16_t val_i16 = (int16_t)roundf(val_f / scale_factor);
    return val_i16;
}

// Per-block Q8_0→INT16 scale factors (for batch dequantization)
// Returns: scale_int16 = max_abs(Q8 block) / 32767 encoded as fp16-like value
// and the Q8_0 byte offset for the block
struct Q8BlockInfo {
    float scale_factor;  // max_abs / 32767 for this block
    uint64_t offset;     // byte offset to block data
};

// Get Q8_0 block info for a given element index
inline Q8BlockInfo get_q8_block_info(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t block_offset = block * 34;
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    float scale = f16_to_f32(scale_raw);
    float max_abs = 0;
    for (uint64_t i = 0; i < 32; i++) {
        int8_t v = (int8_t)data[block_offset + 2 + i];
        float a = fabsf((float)v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    float scale_factor = max_abs / 32767.0f;
    return {scale_factor, block_offset + 2};  // offset points to first INT8 value
}

float dequant_q5_0(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t base = block * 22;
    uint16_t d_raw = data[base] | (data[base + 1] << 8);
    float d = f16_to_f32(d_raw);
    uint32_t qh = data[base + 2] | (data[base + 3] << 8) |
                  (data[base + 4] << 16) | (data[base + 5] << 24);
    // ggml Q5_0: pairs elements (j, j+16) per qs byte
    // qs[j] lower nibble = element j, upper nibble = element j+16
    // qh bit j = high bit for element j, qh bit j+12 = high bit for element j+16
    uint64_t j = offset < 16 ? offset : offset - 16;
    uint8_t qs_byte = data[base + 6 + j];
    uint8_t ql = (offset < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
    uint8_t qh_bit = (qh >> offset) & 1;
    int8_t q = ((qh_bit << 4) | ql) - 16;
    return d * (float)q;
}

float dequant_q6_k(const uint8_t* data, uint64_t idx) {
    const uint64_t QK_K = 256;
    const uint64_t block = idx / QK_K;
    const uint64_t offset = idx % QK_K;
    const uint64_t base = block * 210;
    uint16_t d_raw = data[base + 208] | (data[base + 209] << 8);
    float d = f16_to_f32(d_raw);
    uint64_t half = offset / 128;
    uint64_t pos = offset % 128;
    uint64_t l = pos % 32;
    uint64_t sub = pos / 32;
    uint64_t ql_base = base + half * 64;
    uint64_t qh_base = base + 128 + half * 32;
    uint64_t sc_base = base + 192 + half * 8;
    uint8_t ql_byte;
    uint8_t qh_shift;
    if (sub == 0) {
        ql_byte = data[ql_base + l];
        qh_shift = 0;
    } else if (sub == 1) {
        ql_byte = data[ql_base + l + 32];
        qh_shift = 2;
    } else if (sub == 2) {
        ql_byte = data[ql_base + l];
        qh_shift = 4;
    } else {
        ql_byte = data[ql_base + l + 32];
        qh_shift = 6;
    }
    uint8_t ql_nibble = (sub == 0 || sub == 1) ? (ql_byte & 0xF) : (ql_byte >> 4);
    uint8_t qh_bits = (data[qh_base + l] >> qh_shift) & 0x3;
    int8_t q = (int8_t)((qh_bits << 4) | ql_nibble) - 32;
    uint64_t is = l / 16;
    uint64_t scale_idx = is + sub * 2;
    int8_t scale = (int8_t)data[sc_base + scale_idx];
    return d * scale * q;
}

float dequant_q4_k(const uint8_t* data, uint64_t idx) {
    const uint64_t QK_K = 256;
    const uint64_t block = idx / QK_K;
    const uint64_t offset = idx % QK_K;
    const uint64_t base = block * 144;

    uint16_t d_raw = data[base] | (data[base + 1] << 8);
    float d = f16_to_f32(d_raw);
    uint16_t dmin_raw = data[base + 2] | (data[base + 3] << 8);
    float dmin = f16_to_f32(dmin_raw);

    uint64_t sub_block = offset / 32;
    uint64_t q_pos = offset % 32;
    uint64_t qs_byte_idx = (sub_block / 2) * 32 + q_pos;

    uint8_t qs_byte = data[base + 16 + qs_byte_idx];
    uint8_t q4 = (sub_block % 2 == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);

    const uint8_t* scales = data + base + 4;
    uint8_t sc, m;
    if (sub_block < 4) {
        sc = scales[sub_block] & 63;
        m = scales[sub_block + 4] & 63;
    } else {
        sc = (scales[sub_block + 4] & 0xF) | ((scales[sub_block - 4] >> 6) << 4);
        m = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
    }

    return d * (float)sc * (float)q4 - dmin * (float)m;
}

// ===========================================================================
// TENSOR ACCESS
// ===========================================================================
float read_tensor(const Tensor* t, uint64_t row, uint64_t col) {
    // GGUF/GGML stores 2D weights column-major: ne[0]=input_dim, ne[1]=output_dim
    // Element at logical W[output][input] is at flat index: input + output * input_dim
    // t->cols = ne[0] = input_dim, t->rows = ne[1] = output_dim
    // For 1D tensors (cols==1): flat index = row
    uint64_t idx = (t->cols == 1) ? row : (col + row * t->cols);
    if (t->type == TENSOR_F32) return ((float*)t->data)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)t->data)[idx]);
    if (t->type == TENSOR_Q8_0) return dequant_q8_0(t->data, idx);
    if (t->type == TENSOR_Q5_0) return dequant_q5_0(t->data, idx);
    if (t->type == TENSOR_Q6_K) return dequant_q6_k(t->data, idx);
    if (t->type == TENSOR_Q4_K) return dequant_q4_k(t->data, idx);
    return 0.0f;
}

float read_embedding(const Tensor* t, uint64_t token_id, uint64_t feature) {
    // GGUF token_embd.weight: shape [vocab_size, hidden_dim] = [151936, 896]
    // GGML stores ne[0]=896 (hidden), ne[1]=151936 (vocab)
    // ggml_get_rows interprets ne[1] as rows — row token_id starts at token_id*896
    // Element (token_id, feature) = data[token_id * hidden_dim + feature]
    // t->cols = 151936 (vocab_size), but we use HIDDEN_DIM=896 for stride
    uint64_t idx = token_id * HIDDEN_DIM + feature;
    if (t->type == TENSOR_Q8_0) return dequant_q8_0(t->data, idx);
    if (t->type == TENSOR_F32) return ((float*)t->data)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)t->data)[idx]);
    if (t->type == TENSOR_Q5_0) return dequant_q5_0(t->data, idx);
    if (t->type == TENSOR_Q6_K) return dequant_q6_k(t->data, idx);
    if (t->type == TENSOR_Q4_K) return dequant_q4_k(t->data, idx);
    return 0.0f;
}

Tensor* get_tensor_info(const char* name) {
    for (int i = 0; i < g_ntensors; i++) {
        if (strcmp(g_tensors[i].name, name) == 0) return &g_tensors[i];
    }
    return nullptr;
}

extern void q8_logits_matmul_with_tensor(const Tensor* emb_t, const float* x, float* y, int rows, int cols);

// ===========================================================================
// RMS NORM
// ===========================================================================
float rms(const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum / n);
}

void rms_norm(float* o, const float* x, int n, const Tensor* t) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float mean = sum / n;
    float scale = 1.0f / sqrtf(mean + 1e-6f);
    float* w = (float*)t->data;
    for (int i = 0; i < n; i++) o[i] = x[i] * w[i] * scale;
}

// ===========================================================================
// SILU
// ===========================================================================
void silu(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

// ===========================================================================
// MATMUL
// ===========================================================================
void matmul_fpga_int16(const Tensor* A, const float* x, float* y, int rows, int cols);
void matmul_fpga_q8(const Tensor* A, const float* x, float* y, int rows, int cols, const float* row_max_abs = nullptr);
void matmul_fpga_q4_k(const Tensor* A, const float* x, float* y, int rows, int cols);
void matmul_fpga_q5_0(const Tensor* A, const float* x, float* y, int rows, int cols);
void matmul_fpga_q6_k(const Tensor* A, const float* x, float* y, int rows, int cols);
void get_logits_q8(float* logits, const float* hidden);

void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    if (g_use_metal) {
        switch (A->type) {
            case TENSOR_Q8_0:
                metal_backend::matmul_q8_0(g_mtl_ctx, rows, cols, A->data, x, y);
                return;
            case TENSOR_Q5_0:
                metal_backend::matmul_q5_0(g_mtl_ctx, rows, cols, A->data, x, y);
                return;
            case TENSOR_Q4_K:
                metal_backend::matmul_q4_k(g_mtl_ctx, rows, cols, A->data, x, y);
                return;
            case TENSOR_Q6_K:
                metal_backend::matmul_q6_k(g_mtl_ctx, rows, cols, A->data, x, y);
                return;
            default:
                break; // fall through to CPU
        }
    }
    if (g_use_fpga) {
        if (g_fpga_q4k && A->type == TENSOR_Q5_0) {
            matmul_fpga_q5_0(A, x, y, rows, cols);
        } else if (g_fpga_q4k && A->type == TENSOR_Q6_K) {
            matmul_fpga_q6_k(A, x, y, rows, cols);
        } else if (g_fpga_q4k && A->type == TENSOR_Q4_K) {
            matmul_fpga_q4_k(A, x, y, rows, cols);
        } else if (g_fpga_q8 && A->type == TENSOR_Q8_0) {
            matmul_fpga_q8(A, x, y, rows, cols);
        } else {
            matmul_fpga_int16(A, x, y, rows, cols);
        }
        return;
    }
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) sum += read_tensor(A, i, j) * x[j];
        y[i] = sum;
    }
}

void matmul_transposed(const Tensor* A, const float* x, float* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) sum += read_tensor(A, j, i) * x[j];
        y[i] = sum;
    }
}

// ===========================================================================
// FPGA MATMUL (SIMULATED INT8 SYSTOLIC ACCELERATOR)
// ===========================================================================
// FPGA MATMUL INT16 — CPU quantizes weights to INT16, INT16×INT16 matmul
// ===========================================================================
void matmul_fpga_int16(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    for (int r = 0; r < rows; r += fpga_sim::N) {
        int r_size = std::min(fpga_sim::N, rows - r);

        std::vector<float> W_buf(r_size * cols);
        for (int i = 0; i < r_size; i++)
            for (int j = 0; j < cols; j++)
                W_buf[i * cols + j] = read_tensor(A, r + i, j);

        float row_scale[fpga_sim::N];
        for (int i = 0; i < r_size; i++) {
            float m = 0;
            for (int j = 0; j < cols; j++) m = fmaxf(m, fabsf(W_buf[i * cols + j]));
            row_scale[i] = (m < 1e-10f) ? 1.0f : m / 32767.0f;
        }

        for (int c = 0; c < cols; c += fpga_sim::N) {
            int c_size = std::min(fpga_sim::N, cols - c);

            fpga_sim::in16_t W_q[fpga_sim::N][fpga_sim::N] = {{0}};
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < c_size; k++)
                    W_q[k][i] = (fpga_sim::in16_t)roundf(
                        W_buf[i * cols + c + k] / row_scale[i]);

            fpga_sim::in16_t vec[fpga_sim::N] = {0};
            for (int k = 0; k < c_size; k++) vec[k] = x_q[c + k];

            fpga_sim::acc16_t result[fpga_sim::N] = {0};
            fpga_sim::axi_vecmul_tile_int16(vec, W_q, result);

            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * row_scale[i];

            fpga_sim::g_timing.total_tiles++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// FPGA MATMUL Q8_0 DIRECT PATH
// FPGA receives Q8_0 bytes + row scales, does Q8→INT16 internally.
// This moves the dequantization workload from CPU to FPGA.
// ===========================================================================
void matmul_fpga_q8(const Tensor* A, const float* x, float* y, int rows, int cols, const float* row_max_abs) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    constexpr int Q8_BLOCK_BYTES = 34;
    constexpr int Q8_BLOCK_SIZE = 32;

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    for (int r = 0; r < rows; r += fpga_sim::N) {
        int r_size = std::min(fpga_sim::N, rows - r);

        // Read pure INT8 weight values from Q8_0 tensor (stripping FP16 headers)
        std::vector<uint8_t> W_q8(r_size * cols);
        for (int i = 0; i < r_size; i++)
            for (int j = 0; j < cols; j++) {
                uint64_t idx = (uint64_t)(r + i) * cols + j;
                uint64_t block = idx / Q8_BLOCK_SIZE;
                uint64_t off = idx % Q8_BLOCK_SIZE;
                uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + off;
                W_q8[i * cols + j] = A->data[block_off];
            }

        // Compute per-row scales
        float row_scale[fpga_sim::N];
        if (row_max_abs) {
            for (int i = 0; i < r_size; i++) {
                float max_abs = row_max_abs[r + i];
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        } else {
            for (int i = 0; i < r_size; i++) {
                float max_abs = 0.0f;
                for (int j = 0; j < cols; j++) {
                    uint64_t idx = (uint64_t)(r + i) * cols + j;
                    uint64_t block = idx / Q8_BLOCK_SIZE;
                    uint64_t off = idx % Q8_BLOCK_SIZE;
                    uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + off;
                    int8_t val_q8 = (int8_t)A->data[block_off];
                    uint64_t block_base = block * Q8_BLOCK_BYTES;
                    uint16_t scale_raw = (uint16_t)A->data[block_base] | ((uint16_t)A->data[block_base + 1] << 8);
                    float dequant_val = val_q8 * f16_to_f32(scale_raw);
                    float a = fabsf(dequant_val);
                    if (a > max_abs) max_abs = a;
                }
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        }
        for (int i = r_size; i < fpga_sim::N; i++) row_scale[i] = 1.0f;

        for (int c = 0; c < cols; c += fpga_sim::N) {
            int c_size = std::min(fpga_sim::N, cols - c);

            // Extract pure INT8 tile (64×64) from W_q8
            uint8_t q8_tile[fpga_sim::N][fpga_sim::N];
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < c_size; k++)
                    q8_tile[k][i] = W_q8[i * cols + c + k];
            for (int i = r_size; i < fpga_sim::N; i++)
                for (int k = 0; k < fpga_sim::N; k++)
                    q8_tile[k][i] = 0;
            for (int i = 0; i < r_size; i++)
                for (int k = c_size; k < fpga_sim::N; k++)
                    q8_tile[k][i] = 0;

            // Precompute combined UQ8.8 scales from original tensor FP16 headers
            uint16_t combined_scales[fpga_sim::N * 2] = {0};
            for (int i = 0; i < r_size; i++) {
                for (int blk = 0; blk < 2; blk++) {
                    uint64_t first_col = c + blk * Q8_BLOCK_SIZE;
                    if (first_col >= (uint64_t)cols) { combined_scales[i * 2 + blk] = 0; continue; }
                    uint64_t flat_idx = first_col + (uint64_t)(r + i) * cols;
                    uint64_t block_num = flat_idx / Q8_BLOCK_SIZE;
                    uint64_t block_base = block_num * Q8_BLOCK_BYTES;
                    uint16_t f16_raw = (uint16_t)A->data[block_base] | ((uint16_t)A->data[block_base + 1] << 8);
                    float block_scale = f16_to_f32(f16_raw);
                    float row_s = row_scale[i];
                    float row_inv = (row_s < 1e-10f) ? 1.0f : (1.0f / row_s);
                    float combined = block_scale * row_inv;
                    if (combined >= 256.0f) combined = 255.996f;
                    if (combined < 0.0f) combined = 0.0f;
                    combined_scales[i * 2 + blk] = (uint16_t)(combined * 256.0f + 0.5f);
                }
            }

            fpga_sim::in16_t vec[fpga_sim::N] = {0};
            for (int k = 0; k < c_size; k++) vec[k] = x_q[c + k];

            fpga_sim::acc16_t result[fpga_sim::N] = {0};

            // Cosimulation tile dump
            if (g_dump_file && g_dump_tiles_remaining > 0) {
                // Write tile to dump file: weights + scales + vec, then compute + write result
                fwrite(q8_tile, 1, sizeof(q8_tile), g_dump_file);
                fwrite(combined_scales, 2, fpga_sim::N * 2, g_dump_file);
                fwrite(vec, 2, fpga_sim::N, g_dump_file);
                // Compute reference result via fpga_sim
                fpga_sim::axi_vecmul_tile_q8(
                    (const uint8_t*)q8_tile,
                    combined_scales,
                    vec,
                    result
                );
                fwrite(result, 8, fpga_sim::N, g_dump_file);
                g_dump_tiles_remaining--;
                if (g_dump_tiles_remaining == 0) {
                    fclose(g_dump_file);
                    g_dump_file = nullptr;
                    printf("[COSIM] Dumped all tiles to /tmp/cosim_tiles.bin\n");
                }
            } else {
                fpga_sim::axi_vecmul_tile_q8(
                    (const uint8_t*)q8_tile,
                    combined_scales,
                    vec,
                    result
                );
            }

            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * row_scale[i];

            fpga_sim::g_timing.total_tiles++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// Q4_K FPGA PATH — FPGA receives Q4_K blocks, does Q4_K→INT16 internally.
// Uses per-element dequant to extract 64×64 tiles from Q4_K tensor.
// Uses raw Q4_K block path: CPU sends Q4_K blocks to FPGA, FPGA does block
// decode + row-normalized INT16 matmul internally.
// Tile structure: 896 rows × 64 cols = 224 blocks (stride 76 in tensor)
// row_scale[row] = UQ16.8 = round(32767 / row_max_abs * 256)
// ===========================================================================
// Q4_K down_proj path: [56 rows × 256 cols], 144-byte blocks.
// ===========================================================================
void matmul_fpga_q4_k(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;
    // == 896×256 Path: for tensors with cols >= 1024 (down_proj [896, 4864]) ==
        // == 896×256 Path: for tensors with cols >= 1024 (down_proj [896, 4864]) ==
        // Each Q4_K block covers 1 row × 256 cols, stride = cols/256 = 19.
        // Tile: 56 rows × 256 cols = 56 blocks × 144 = 8064 bytes (fits in 8192 weight_buf).
        // 16 row-batches × 19 column-groups = 304 tiles.

        float row_inv[896];
        for (int row = 0; row < rows; row++) {
            float max_abs = 0.0f;
            uint64_t nblocks = ((uint64_t)cols + 255) / 256;
            uint64_t base_block = (uint64_t)row * cols / 256;
            uint64_t row_start_flat = (uint64_t)row * cols;
            float block_f32[256];
            for (uint64_t b = 0; b < nblocks; b++) {
                fpga_sim::dequant_q4k_block_to_float(
                    A->data + (base_block + b) * 144, block_f32);
                for (int i = 0; i < 256; i++) {
                    uint64_t flat_idx = (base_block + b) * 256 + i;
                    uint64_t rel_idx = flat_idx - row_start_flat;
                    if (rel_idx >= (uint64_t)cols) break;
                    float absv = fabsf(block_f32[i]);
                    if (absv > max_abs) max_abs = absv;
                }
            }
            max_abs = (max_abs < 1e-10f) ? 1.0f : max_abs;
            row_inv[row] = 32767.0f / max_abs;
        }

        int64_t tile_result[896] = {0};
        int block_stride = cols / 256;

        for (int c = 0; c < cols; c += 256) {
            int block_base_offset = c / 256;

            for (int row0 = 0; row0 < rows; row0 += 56) {
                int nblocks = std::min(56, rows - row0);

                uint8_t blocks[56 * fpga_sim::Q4K_BLOCK_BYTES];
                for (int bi = 0; bi < nblocks; bi++) {
                    uint64_t block_idx = (uint64_t)(row0 + bi) * block_stride + block_base_offset;
                    memcpy(blocks + bi * fpga_sim::Q4K_BLOCK_BYTES,
                           A->data + block_idx * fpga_sim::Q4K_BLOCK_BYTES,
                           fpga_sim::Q4K_BLOCK_BYTES);
                }

                fpga_sim::axi_vecmul_tile_q4k_axilite(
                    blocks, nblocks, x_q.data() + c, 256, row_inv, tile_result, row0);

                fpga_sim::g_timing.total_tiles++;
            }
        }

        for (int i = 0; i < rows; i++) {
            y[i] += (double)tile_result[i] * x_scale / (double)row_inv[i];
        }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// Q5_0 path: [8 rows × 896 cols], 22-byte blocks (32 vals/block).
// Handles all Q5_0 tensors (FFN gate/up, attn Q/K/O) where cols=896.
// ===========================================================================
void matmul_fpga_q5_0(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    constexpr int TILE_ROWS = 8;
    constexpr int BLOCKS_PER_ROW = 28;     // 896 / 32 = 28 blocks per row
    constexpr int BLOCKS_PER_TILE = 224;   // 8 rows × 28

    uint64_t row_stride_blocks = (uint64_t)cols / fpga_sim::Q5_0_BLOCK_SIZE;  // 28

    // Pre-compute row_inv (FP32) for all rows
    std::vector<float> row_inv(rows);
    for (int row = 0; row < rows; row++) {
        float max_abs = 0.0f;
        uint64_t base_block = (uint64_t)row * row_stride_blocks;
        for (uint64_t b = 0; b < BLOCKS_PER_ROW; b++) {
            const uint8_t* blk = A->data + (base_block + b) * fpga_sim::Q5_0_BLOCK_BYTES;
            float d = fpga_sim::read_f16(blk);
            uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                          ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
            for (int wi = 0; wi < 32; wi++) {
                uint64_t j = wi < 16 ? wi : wi - 16;
                uint8_t qs_byte = blk[6 + j];
                uint8_t ql = (wi < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
                uint8_t qh_bit = (qh >> wi) & 1;
                int q = ((qh_bit << 4) | ql) - 16;
                float ab = fabsf(d * (float)q);
                if (ab > max_abs) max_abs = ab;
            }
        }
        max_abs = (max_abs < 1e-10f) ? 1.0f : max_abs;
        row_inv[row] = 32767.0f / max_abs;
    }

    std::vector<int64_t> tile_result(rows, 0);

    for (int row0 = 0; row0 < rows; row0 += TILE_ROWS) {
        int r_size = std::min(TILE_ROWS, rows - row0);

        // Load 224 Q5_0 blocks
        uint8_t blocks[fpga_sim::Q5_0_224BLOCK_BYTES];
        uint64_t tile_base_block = (uint64_t)row0 * row_stride_blocks;
        for (int bi = 0; bi < BLOCKS_PER_TILE; bi++) {
            memcpy(blocks + bi * fpga_sim::Q5_0_BLOCK_BYTES,
                   A->data + (tile_base_block + bi) * fpga_sim::Q5_0_BLOCK_BYTES,
                   fpga_sim::Q5_0_BLOCK_BYTES);
        }

        fpga_sim::axi_vecmul_tile_q5_0_8x896_axilite(
            blocks, x_q.data(), row_inv.data() + row0, tile_result.data(), row0);

        fpga_sim::g_timing.total_tiles++;
    }

    for (int i = 0; i < rows; i++) {
        y[i] += (double)tile_result[i] * x_scale / (double)row_inv[i];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// Q6_K down_proj path: [32 rows × 256 cols], 210-byte blocks.
// ===========================================================================
void matmul_fpga_q6_k(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    // Pre-compute row_inv (FP32) for all rows
    float row_inv[896];
    int block_stride = cols / fpga_sim::Q6_K_BLOCK_SIZE;  // = 19 for [896, 4864]

    for (int row = 0; row < rows; row++) {
        float max_abs = 0.0f;
        uint64_t nblocks = ((uint64_t)cols + fpga_sim::Q6_K_BLOCK_SIZE - 1) / fpga_sim::Q6_K_BLOCK_SIZE;
        uint64_t base_block = (uint64_t)row * cols / fpga_sim::Q6_K_BLOCK_SIZE;
        uint64_t row_start_flat = (uint64_t)row * cols;

        for (uint64_t b = 0; b < nblocks; b++) {
            const uint8_t* blk = A->data + (base_block + b) * fpga_sim::Q6_K_BLOCK_BYTES;
            float super_scale = fpga_sim::read_f16(blk + 208);
            for (int wi = 0; wi < fpga_sim::Q6_K_BLOCK_SIZE; wi++) {
                uint64_t flat_idx = (base_block + b) * fpga_sim::Q6_K_BLOCK_SIZE + wi;
                uint64_t rel_idx = flat_idx - row_start_flat;
                if (rel_idx >= (uint64_t)cols) break;

                int half = wi / 128;
                int pos = wi % 128;
                int l = pos % 32;
                int sub = pos / 32;

                int L_off = half * 64 + l + (sub % 2) * 32;
                uint8_t ql_byte = blk[L_off];
                uint8_t ql_nibble = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

                int H_off = 128 + half * 32 + l;
                int qh_shift = sub * 2;
                uint8_t qh_bits = (blk[H_off] >> qh_shift) & 0x3;

                int q6 = ((qh_bits << 4) | ql_nibble) - 32;

                int sc_off = 192 + half * 8 + (l / 16) + sub * 2;
                float scale_f = (float)(int8_t)blk[sc_off];

                float absv = fabsf(super_scale * scale_f * (float)q6);
                if (absv > max_abs) max_abs = absv;
            }
        }

        max_abs = (max_abs < 1e-10f) ? 1.0f : max_abs;
        row_inv[row] = 32767.0f / max_abs;
    }

    int64_t tile_result[896] = {0};
    constexpr int TILE_ROWS = 32;
    constexpr int BLOCKS_PER_TILE = 32;

    for (int c = 0; c < cols; c += fpga_sim::Q6_K_BLOCK_SIZE) {
        int block_base_offset = c / fpga_sim::Q6_K_BLOCK_SIZE;

        for (int row0 = 0; row0 < rows; row0 += TILE_ROWS) {
            int nblocks = std::min(TILE_ROWS, rows - row0);

            uint8_t blocks[TILE_ROWS * fpga_sim::Q6_K_BLOCK_BYTES];
            for (int bi = 0; bi < nblocks; bi++) {
                uint64_t block_idx = (uint64_t)(row0 + bi) * block_stride + block_base_offset;
                memcpy(blocks + bi * fpga_sim::Q6_K_BLOCK_BYTES,
                       A->data + block_idx * fpga_sim::Q6_K_BLOCK_BYTES,
                       fpga_sim::Q6_K_BLOCK_BYTES);
            }

            fpga_sim::axi_vecmul_tile_q6_k_axilite(
                blocks, nblocks, x_q.data() + c, fpga_sim::Q6_K_BLOCK_SIZE,
                row_inv, tile_result, row0);

            fpga_sim::g_timing.total_tiles++;
        }
    }

    for (int i = 0; i < rows; i++) {
        y[i] += (double)tile_result[i] * x_scale / (double)row_inv[i];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// ROTARY EMBEDDING
// ===========================================================================
void apply_rope(float* q, float* k, int pos) {
    const float rope_base = 1000000.0f;
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float q0 = q[idx], q1 = q[idx + 1];
            q[idx] = q0 * cos_val - q1 * sin_val;
            q[idx + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float k0 = k[idx], k1 = k[idx + 1];
            k[idx] = k0 * cos_val - k1 * sin_val;
            k[idx + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ===========================================================================
// FORWARD LAYER with KV Cache
// ===========================================================================
void forward_layer_with_cache(float* hidden, int layer, int pos) {
    char name[128];
    Tensor* t;
    float original_hidden[HIDDEN_DIM];
    memcpy(original_hidden, hidden, HIDDEN_DIM * sizeof(float));

    float attn_norm_out[HIDDEN_DIM];
    float q_vec[HIDDEN_DIM];
    float k_new[K_DIM];
    float v_new[V_DIM];
    float context[HIDDEN_DIM];
    float attn_out[HIDDEN_DIM];
    float ffn_norm_out[HIDDEN_DIM];
    float gate[INTER_DIM];
    float up[INTER_DIM];
    float ffn_out[HIDDEN_DIM];

    // Attention norm
    {
        PROFILE_SCOPE("attn_norm");
        snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
        t = get_tensor_info(name);
        rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);
    }

    // Q, K, V matmuls (batched — same input, independent outputs)
    metal_backend::metal_batch_begin(g_mtl_ctx);
    {
        PROFILE_SCOPE("attn_q");
        snprintf(name, 128, "blk.%d.attn_q.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, q_vec, HIDDEN_DIM, HIDDEN_DIM);
    }
    {
        PROFILE_SCOPE("attn_k");
        snprintf(name, 128, "blk.%d.attn_k.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, k_new, K_DIM, HIDDEN_DIM);
    }
    {
        PROFILE_SCOPE("attn_v");
        snprintf(name, 128, "blk.%d.attn_v.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, v_new, V_DIM, HIDDEN_DIM);
    }
    metal_backend::metal_batch_end(); // Q, K, V results ready

    // Bias adds (depend on GPU output)
    {
        snprintf(name, 128, "blk.%d.attn_q.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q_vec[i] += b[i]; }
    }
    {
        snprintf(name, 128, "blk.%d.attn_k.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k_new[i] += b[i]; }
    }
    {
        snprintf(name, 128, "blk.%d.attn_v.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v_new[i] += b[i]; }
    }

    {
        PROFILE_SCOPE("rope");
        apply_rope(q_vec, k_new, pos);
    }

    for (int i = 0; i < K_DIM; i++) g_k_cache[layer][pos][i] = k_new[i];
    for (int i = 0; i < V_DIM; i++) g_v_cache[layer][pos][i] = v_new[i];

    {
        PROFILE_SCOPE("attention");
        memset(context, 0, HIDDEN_DIM * sizeof(float));
        int q_per_kv = NUM_HEADS / NUM_KV_HEADS;

        for (int qh = 0; qh < NUM_HEADS; qh++) {
            int kv = qh / q_per_kv;
            float* qh_data = q_vec + qh * HEAD_DIM;
            float* ctx_h = context + qh * HEAD_DIM;

            float scores[MAX_SEQ_LEN];
            for (int p = 0; p <= pos; p++) scores[p] = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                float q_val = qh_data[d];
                for (int p = 0; p <= pos; p++) {
                    float* k_cached = g_k_cache[layer][p] + kv * HEAD_DIM;
                    scores[p] += q_val * k_cached[d];
                }
            }
            for (int p = 0; p <= pos; p++) scores[p] /= sqrtf((float)HEAD_DIM);

            float max_score = scores[0];
            for (int p = 1; p <= pos; p++) if (scores[p] > max_score) max_score = scores[p];
            float exp_sum = 0;
            for (int p = 0; p <= pos; p++) exp_sum += expf(scores[p] - max_score);
            float log_sum = logf(exp_sum) + max_score;

            for (int p = 0; p <= pos; p++) {
                float* v_cached = g_v_cache[layer][p] + kv * HEAD_DIM;
                float exp_score = expf(scores[p] - log_sum);
                for (int d = 0; d < HEAD_DIM; d++) {
                    ctx_h[d] += exp_score * v_cached[d];
                }
            }
        }
    }

    // Attention output (standalone)
    metal_backend::metal_batch_begin(g_mtl_ctx);
    {
        PROFILE_SCOPE("attn_output");
        snprintf(name, 128, "blk.%d.attn_output.weight", layer);
        t = get_tensor_info(name);
        matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);
    }
    metal_backend::metal_batch_end(); // attn_out ready

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];

    // FFN norm
    {
        PROFILE_SCOPE("ffn_norm");
        snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
        t = get_tensor_info(name);
        rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);
    }

    // FFN gate + up (batched — same input, independent outputs)
    metal_backend::metal_batch_begin(g_mtl_ctx);
    {
        PROFILE_SCOPE("ffn_gate");
        snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
        t = get_tensor_info(name);
        matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);
    }
    {
        PROFILE_SCOPE("ffn_up");
        snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
        t = get_tensor_info(name);
        matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);
    }
    metal_backend::metal_batch_end(); // gate, up ready

    {
        PROFILE_SCOPE("silu_x_up");
        silu(gate, gate, INTER_DIM);
        for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];
    }

    // FFN down (standalone)
    metal_backend::metal_batch_begin(g_mtl_ctx);
    {
        PROFILE_SCOPE("ffn_down");
        snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
        t = get_tensor_info(name);
        matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);
    }
    metal_backend::metal_batch_end(); // ffn_out ready

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
}

// ===========================================================================
// FORWARD (no cache - single token)
// ===========================================================================
void forward_layer(float* hidden, int layer, int pos) {
    char name[128];
    Tensor* t;
    float original_hidden[HIDDEN_DIM];
    memcpy(original_hidden, hidden, HIDDEN_DIM * sizeof(float));

    // Attention norm
    snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
    t = get_tensor_info(name);
    float attn_norm_out[HIDDEN_DIM];
    rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);

    // Q, K, V
    snprintf(name, 128, "blk.%d.attn_q.weight", layer);
    t = get_tensor_info(name);
    float q[HIDDEN_DIM];
    matmul(t, attn_norm_out, q, HIDDEN_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_q.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q[i] += b[i]; }

    snprintf(name, 128, "blk.%d.attn_k.weight", layer);
    t = get_tensor_info(name);
    float k[K_DIM];
    matmul(t, attn_norm_out, k, K_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_k.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k[i] += b[i]; }

    snprintf(name, 128, "blk.%d.attn_v.weight", layer);
    t = get_tensor_info(name);
    float v[V_DIM];
    matmul(t, attn_norm_out, v, V_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_v.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v[i] += b[i]; }

    apply_rope(q, k, pos);

    // GQA attention (single token)
    float context[HIDDEN_DIM] = {0};
    int q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    for (int qh = 0; qh < NUM_HEADS; qh++) {
        int kv = qh / q_per_kv;
        float* qh_data = q + qh * HEAD_DIM;
        float* kh_data = k + kv * HEAD_DIM;
        float* vh_data = v + kv * HEAD_DIM;
        float* ctx_h = context + qh * HEAD_DIM;

        float score = 0;
        for (int d = 0; d < HEAD_DIM; d++) score += qh_data[d] * kh_data[d];
        score /= sqrtf((float)HEAD_DIM);
        float softmax_w = 1.0f;

        for (int d = 0; d < HEAD_DIM; d++) ctx_h[d] = softmax_w * vh_data[d];
    }

    // Attention output projection
    snprintf(name, 128, "blk.%d.attn_output.weight", layer);
    t = get_tensor_info(name);
    float attn_out[HIDDEN_DIM];
    matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);

    // Residual
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];

    // FFN norm
    snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
    t = get_tensor_info(name);
    float ffn_norm_out[HIDDEN_DIM];
    rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);

    // FFN gate
    snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
    t = get_tensor_info(name);
    float gate[INTER_DIM];
    matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);

    // FFN up
    snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
    t = get_tensor_info(name);
    float up[INTER_DIM];
    matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);

    // SwiGLU
    silu(gate, gate, INTER_DIM);
    for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];

    // FFN down
    snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
    t = get_tensor_info(name);
    float ffn_out[HIDDEN_DIM];
    matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);

    // Final residual
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
}

// ===========================================================================
// GET LOGITS — Q8_0 PATH
// FPGA receives hidden (1×896 FP32) once, streams Q8_0 embeddings,
// does Q8→INT16 dequant internally, returns raw logits (151936 floats).
// This bypasses CPU dequantization for the token embedding lookup.
// ===========================================================================
void get_logits_q8(float* logits, const float* hidden) {
#ifdef Q8_DEBUG
    printf("[LOGITS_Q8] get_logits_q8 called\n"); fflush(stdout);
#endif
    Tensor* emb_t = get_tensor_info("token_embd.weight");
#ifdef Q8_DEBUG
    printf("[LOGITS_Q8] emb_t=%p\n", (void*)emb_t); fflush(stdout);
#endif
    if (!emb_t) { printf("[ERROR] No token_embd.weight\n"); return; }

    Tensor* norm_t = get_tensor_info("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    {
        PROFILE_SCOPE("output_norm");
        if (norm_t) {
            rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
            hidden = norm_hidden;
        }
    }

    {
        PROFILE_SCOPE("logits");
        if (g_use_metal) {
            metal_backend::metal_batch_begin(g_mtl_ctx);
            matmul(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
            metal_backend::metal_batch_end();
        } else {
            extern void q8_logits_matmul_with_tensor(const Tensor* t, const float* x, float* y, int rows, int cols);
            q8_logits_matmul_with_tensor(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
        }
    }
}

// ===========================================================================
// GET LOGITS — DEFAULT PATH
// ===========================================================================
void get_logits(float* logits, const float* hidden) {
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) { printf("[ERROR] No token_embd.weight\n"); return; }

    Tensor* norm_t = get_tensor_info("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    {
        PROFILE_SCOPE("output_norm");
        if (norm_t) {
            rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
            hidden = norm_hidden;
        }
    }

    {
        PROFILE_SCOPE("logits");
        matmul(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
    }
}

// ===========================================================================
// TOP-K SAMPLING
// ===========================================================================
int sample_top_k(float* logits, int k) {
    std::vector<std::pair<float, int>> topk;
    for (int i = 0; i < VOCAB_SIZE; i++) topk.push_back({logits[i], i});
    std::partial_sort(topk.begin(), topk.begin() + k, topk.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0;
    for (int i = 0; i < k; i++) topk[i].first = expf(topk[i].first);
    for (int i = 0; i < k; i++) sum += topk[i].first;
    for (int i = 0; i < k; i++) topk[i].first /= sum;
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < k; i++) {
        cumsum += topk[i].first;
        if (r < cumsum) return topk[i].second;
    }
    return topk[k-1].second;
}

// ===========================================================================
// LOAD TMAC
// ===========================================================================
int load_tmac(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("[ERROR] Cannot open %s\n", path); return -1; }
    uint8_t magic[4];
    if (fread(magic, 1, 4, f) != 4) return -1;
#ifdef TMAC_DEBUG
    printf("[DEBUG] TMAC magic: %02x %02x %02x %02x\n", magic[0], magic[1], magic[2], magic[3]);
#endif
    uint64_t n_tensors;
    if (fread(&n_tensors, 8, 1, f) != 1) return -1;
    g_tensors = (Tensor*)malloc(n_tensors * sizeof(Tensor));
    g_ntensors = n_tensors;
    uint64_t offset = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len, rows, cols, n_bytes;
        uint32_t type;
        if (fread(&name_len, 8, 1, f) != 1) { printf("[ERROR] fread name_len failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (name_len >= 128) {
            printf("[WARN] name_len=%lu >= 128 for tensor %lu, truncating to 127\n", (unsigned long)name_len, (unsigned long)i);
            if (fread(g_tensors[i].name, 1, 127, f) != 127) { printf("[ERROR] fread name failed at tensor %lu\n", (unsigned long)i); return -1; }
            g_tensors[i].name[127] = 0;
            fseek(f, name_len - 127, SEEK_CUR);
        } else {
            if (fread(g_tensors[i].name, 1, name_len, f) != name_len) { printf("[ERROR] fread name failed at tensor %lu\n", (unsigned long)i); return -1; }
            g_tensors[i].name[name_len] = 0;
        }
        g_tensors[i].name[name_len] = 0;
        if (fread(&rows, 8, 1, f) != 1) { printf("[ERROR] fread rows failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&cols, 8, 1, f) != 1) { printf("[ERROR] fread cols failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&type, 4, 1, f) != 1) { printf("[ERROR] fread type failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&n_bytes, 8, 1, f) != 1) { printf("[ERROR] fread n_bytes failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (offset + n_bytes > DDR_SIZE) { printf("[ERROR] overflow at tensor %lu: offset=%lu n_bytes=%lu DDR_SIZE=%lu\n", (unsigned long)i, (unsigned long)offset, (unsigned long)n_bytes, (unsigned long)(uint64_t)DDR_SIZE); return -1; }
        g_tensors[i].rows = rows;
        g_tensors[i].cols = cols;
        g_tensors[i].type = type;
        g_tensors[i].n_bytes = n_bytes;
        g_tensors[i].data = g_ddr + offset;
        if (fread(g_tensors[i].data, 1, n_bytes, f) != n_bytes) { printf("[ERROR] fread data failed at tensor %lu: name=%s rows=%lu cols=%lu n_bytes=%lu\n", (unsigned long)i, g_tensors[i].name, (unsigned long)rows, (unsigned long)cols, (unsigned long)n_bytes); return -1; }
        offset += n_bytes;
#ifdef TMAC_DEBUG
        if (i < 5 || i >= n_tensors - 3) fprintf(stderr, "[LOAD] tensor[%lu] %s rows=%lu cols=%lu type=%u n_bytes=%lu offset_after=%lu\n", (unsigned long)i, g_tensors[i].name, (unsigned long)rows, (unsigned long)cols, type, (unsigned long)n_bytes, (unsigned long)offset);
#endif
    }
    fclose(f);
#ifdef TMAC_DEBUG
    fprintf(stderr, "[LOAD] done loading\n"); fflush(stderr);
#endif
    printf("[OK] Loaded %lu tensors, DDR: %.1f MB / %d MB\n",
           (unsigned long)n_tensors, offset / (1024.0 * 1024.0), (int)(DDR_SIZE / (1024 * 1024))); fflush(stdout);
#ifdef TMAC_DEBUG
    fprintf(stderr, "[LOAD] about to return\n"); fflush(stderr);
#endif
    return 0;
}

// ===========================================================================
// INFERENCE with CACHE
// ===========================================================================
void reset_cache() {
    g_seq_len = 0;
    memset(g_k_cache, 0, sizeof(g_k_cache));
    memset(g_v_cache, 0, sizeof(g_v_cache));
}

void process_embedding(float* hidden, int token_id) {
    PROFILE_SCOPE("embedding");
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) return;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = read_embedding(emb_t, token_id, i);
    }
}

void forward_all_layers(float* hidden, int pos) {
    PROFILE_SCOPE("forward_all_layers");
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        forward_layer_with_cache(hidden, layer, pos);
    }
}

// ── Fused all-GPU forward pass (one command buffer, zero CPU sync) ──
void forward_all_layers_fused(float* hidden, int pos) {
    PROFILE_SCOPE("forward_all_layers_fused");

    // Static buffers so wrap_buffer gets stable pointers
    static float scratch[HIDDEN_DIM];
    static float q_vec[HIDDEN_DIM];
    static float k_new[K_DIM];
    static float v_new[V_DIM];
    static float context[HIDDEN_DIM];
    static float attn_out[HIDDEN_DIM];
    static float gate_up[INTER_DIM * 2];
    static float ffn_out[HIDDEN_DIM];
    char name[128];

    // Per-layer CB when profiling
    int layer_cb_size = g_perf_enabled ? 1 : NUM_LAYERS;
    for (int layer_start = 0; layer_start < NUM_LAYERS; layer_start += layer_cb_size) {
        metal_backend::metal_batch_begin(g_mtl_ctx, layer_start == 0);

        int layer_end = std::min(layer_start + layer_cb_size, NUM_LAYERS);
        for (int layer = layer_start; layer < layer_end; layer++) {
            // ── Attention ──
            metal_backend::g_trace_name = "attn_copy";
            { PROFILE_SCOPE("L00_copy"); metal_backend::elem_op(g_mtl_ctx, 2, scratch, hidden, HIDDEN_DIM); }
            metal_backend::g_trace_name = "attn_rms";
            { PROFILE_SCOPE("L01_attn_norm");
              snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
              metal_backend::rmsnorm_op(g_mtl_ctx, scratch, (const float*)get_tensor_info(name)->data, HIDDEN_DIM); }
            metal_backend::g_trace_name = "attn_q";
            { PROFILE_SCOPE("L02_attn_q");
              snprintf(name, 128, "blk.%d.attn_q.weight", layer);
              matmul(get_tensor_info(name), scratch, q_vec, HIDDEN_DIM, HIDDEN_DIM); }
            metal_backend::g_trace_name = "attn_k";
            { PROFILE_SCOPE("L03_attn_k");
              snprintf(name, 128, "blk.%d.attn_k.weight", layer);
              matmul(get_tensor_info(name), scratch, k_new, K_DIM, HIDDEN_DIM); }
            metal_backend::g_trace_name = "attn_v";
            { PROFILE_SCOPE("L04_attn_v");
              snprintf(name, 128, "blk.%d.attn_v.weight", layer);
              matmul(get_tensor_info(name), scratch, v_new, V_DIM, HIDDEN_DIM); }
            metal_backend::g_trace_name = "attn_bias";
            { PROFILE_SCOPE("L05_bias");
              snprintf(name, 128, "blk.%d.attn_q.bias", layer); Tensor* t = get_tensor_info(name);
              if (t) metal_backend::elem_op(g_mtl_ctx, 0, q_vec, (const float*)t->data, HIDDEN_DIM);
              snprintf(name, 128, "blk.%d.attn_k.bias", layer); t = get_tensor_info(name);
              if (t) metal_backend::elem_op(g_mtl_ctx, 0, k_new, (const float*)t->data, K_DIM);
              snprintf(name, 128, "blk.%d.attn_v.bias", layer); t = get_tensor_info(name);
              if (t) metal_backend::elem_op(g_mtl_ctx, 0, v_new, (const float*)t->data, V_DIM); }
            metal_backend::g_trace_name = "rope";
            { PROFILE_SCOPE("L06_rope");
              metal_backend::rope_op(g_mtl_ctx, q_vec, NUM_HEADS, HEAD_DIM, pos, 1000000.0f);
              metal_backend::rope_op(g_mtl_ctx, k_new, NUM_KV_HEADS, HEAD_DIM, pos, 1000000.0f); }
            metal_backend::g_trace_name = "kv_cache";
            { PROFILE_SCOPE("L07_cache");
              metal_backend::elem_op(g_mtl_ctx, 2, &g_k_cache[layer][pos][0], k_new, K_DIM);
              metal_backend::elem_op(g_mtl_ctx, 2, &g_v_cache[layer][pos][0], v_new, V_DIM); }
            metal_backend::g_trace_name = "attention";
            { PROFILE_SCOPE("L08_attention");
              metal_backend::attention_op(g_mtl_ctx, q_vec, &g_k_cache[layer][0][0], &g_v_cache[layer][0][0], context, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, pos); }
            metal_backend::g_trace_name = "attn_out";
            { PROFILE_SCOPE("L09_attn_out");
              snprintf(name, 128, "blk.%d.attn_output.weight", layer);
              matmul(get_tensor_info(name), context, attn_out, HIDDEN_DIM, HIDDEN_DIM); }
            metal_backend::g_trace_name = "residual1";
            { PROFILE_SCOPE("L10_residual");
              metal_backend::elem_op(g_mtl_ctx, 0, hidden, attn_out, HIDDEN_DIM); }

            // ── FFN ──
            metal_backend::g_trace_name = "ffn_copy";
            { PROFILE_SCOPE("L11_ffn_copy"); metal_backend::elem_op(g_mtl_ctx, 2, scratch, hidden, HIDDEN_DIM); }
            metal_backend::g_trace_name = "ffn_rms";
            { PROFILE_SCOPE("L12_ffn_norm");
              snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
              metal_backend::rmsnorm_op(g_mtl_ctx, scratch, (const float*)get_tensor_info(name)->data, HIDDEN_DIM); }
            snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
            Tensor* t_gate = get_tensor_info(name);
            snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
            Tensor* t_up = get_tensor_info(name);
            if (t_gate && t_up) {
                bool q5 = t_gate->type == TENSOR_Q5_0 && t_up->type == TENSOR_Q5_0;
                if (q5) {
                    metal_backend::g_trace_name = "ffn_gate_up_q5";
                    metal_backend::fused_ffn_gate_up_op(g_mtl_ctx,
                        t_gate->data, t_up->data,
                        scratch, gate_up, INTER_DIM, HIDDEN_DIM,
                        g_mtl_ctx.pipe_fused_ffn_gate_up_q5,
                        ((size_t)INTER_DIM * HIDDEN_DIM + 31) / 32 * 22);
                } else {
                    goto fallback_ffn_fused;
                }
            } else {
            fallback_ffn_fused:
                metal_backend::g_trace_name = "ffn_gate";
                matmul(t_gate, scratch, gate_up, INTER_DIM, HIDDEN_DIM);
                metal_backend::g_trace_name = "ffn_up";
                matmul(t_up, scratch, gate_up + INTER_DIM, INTER_DIM, HIDDEN_DIM);
                metal_backend::g_trace_name = "silu";
                metal_backend::elem_op(g_mtl_ctx, 1, gate_up, nullptr, INTER_DIM);
            }
            metal_backend::g_trace_name = "ffn_down";
            { PROFILE_SCOPE("L16_ffn_down");
              snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
              matmul(get_tensor_info(name), gate_up, ffn_out, HIDDEN_DIM, INTER_DIM); }
            metal_backend::g_trace_name = "residual2";
            { PROFILE_SCOPE("L17_residual");
              metal_backend::elem_op(g_mtl_ctx, 0, hidden, ffn_out, HIDDEN_DIM); }
        }

        metal_backend::metal_batch_end();
    }
}

// ── Fused forward + logits (single CB: layers + output norm + embedding matmul) ──
void forward_and_logits_fused(float* hidden, float* logits, int pos) {
    PROFILE_SCOPE("forward_and_logits_fused");

    static float scratch[HIDDEN_DIM];
    static float q_vec[HIDDEN_DIM];
    static float k_new[K_DIM];
    static float v_new[V_DIM];
    static float context[HIDDEN_DIM];
    static float attn_out[HIDDEN_DIM];
    static float gate_up[INTER_DIM * 2];
    static float ffn_out[HIDDEN_DIM];
    char name[128];

    Tensor* emb_t = get_tensor_info("token_embd.weight");
    Tensor* norm_t = get_tensor_info("output_norm.weight");

    metal_backend::g_params_offset = 0;

    // Chunk layers: per-layer profiling when --perf
    int layer_cb_size = g_perf_enabled ? 1 : NUM_LAYERS;
    for (int layer_start = 0; layer_start < NUM_LAYERS; layer_start += layer_cb_size) {
        metal_backend::metal_batch_begin(g_mtl_ctx, false);

        int layer_end = std::min(layer_start + layer_cb_size, NUM_LAYERS);
        for (int layer = layer_start; layer < layer_end; layer++) {
            metal_backend::g_trace_name = "attn_copy";   metal_backend::elem_op(g_mtl_ctx, 2, scratch, hidden, HIDDEN_DIM);
            snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
            metal_backend::g_trace_name = "attn_rms";    metal_backend::rmsnorm_op(g_mtl_ctx, scratch,
                (const float*)get_tensor_info(name)->data, HIDDEN_DIM);
            snprintf(name, 128, "blk.%d.attn_q.weight", layer);
            Tensor* t_q = get_tensor_info(name);
            snprintf(name, 128, "blk.%d.attn_k.weight", layer);
            Tensor* t_k = get_tensor_info(name);
            snprintf(name, 128, "blk.%d.attn_v.weight", layer);
            Tensor* t_v = get_tensor_info(name);
            metal_backend::g_trace_name = "fused_qkv";
            metal_backend::fused_qkv_op(g_mtl_ctx,
                t_q->data, t_k->data, t_v->data,
                scratch, q_vec, k_new, v_new,
                HIDDEN_DIM, K_DIM, V_DIM, HIDDEN_DIM, t_v->type == TENSOR_Q8_0);
            snprintf(name, 128, "blk.%d.attn_q.bias", layer);
            Tensor* t = get_tensor_info(name);
            metal_backend::g_trace_name = "q_bias";      if (t) metal_backend::elem_op(g_mtl_ctx, 0, q_vec, (const float*)t->data, HIDDEN_DIM);
            snprintf(name, 128, "blk.%d.attn_k.bias", layer);
            t = get_tensor_info(name);
            metal_backend::g_trace_name = "k_bias";      if (t) metal_backend::elem_op(g_mtl_ctx, 0, k_new, (const float*)t->data, K_DIM);
            snprintf(name, 128, "blk.%d.attn_v.bias", layer);
            t = get_tensor_info(name);
            metal_backend::g_trace_name = "v_bias";      if (t) metal_backend::elem_op(g_mtl_ctx, 0, v_new, (const float*)t->data, V_DIM);
            metal_backend::g_trace_name = "rope_q";      metal_backend::rope_op(g_mtl_ctx, q_vec, NUM_HEADS, HEAD_DIM, pos, 1000000.0f);
            metal_backend::g_trace_name = "rope_k";      metal_backend::rope_op(g_mtl_ctx, k_new, NUM_KV_HEADS, HEAD_DIM, pos, 1000000.0f);
            metal_backend::g_trace_name = "kv_cache";    metal_backend::elem_op(g_mtl_ctx, 2, &g_k_cache[layer][pos][0], k_new, K_DIM);
            metal_backend::g_trace_name = "kv_cache";    metal_backend::elem_op(g_mtl_ctx, 2, &g_v_cache[layer][pos][0], v_new, V_DIM);
            metal_backend::g_trace_name = "attn";        metal_backend::attention_op(g_mtl_ctx, q_vec,
                &g_k_cache[layer][0][0], &g_v_cache[layer][0][0],
                context, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, pos);
            snprintf(name, 128, "blk.%d.attn_output.weight", layer);
            metal_backend::g_trace_name = "attn_out";    matmul(get_tensor_info(name), context, attn_out, HIDDEN_DIM, HIDDEN_DIM);
            metal_backend::g_trace_name = "residual1";   metal_backend::elem_op(g_mtl_ctx, 0, hidden, attn_out, HIDDEN_DIM);

            metal_backend::g_trace_name = "ffn_copy";    metal_backend::elem_op(g_mtl_ctx, 2, scratch, hidden, HIDDEN_DIM);
            snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
            metal_backend::g_trace_name = "ffn_rms";     metal_backend::rmsnorm_op(g_mtl_ctx, scratch,
                (const float*)get_tensor_info(name)->data, HIDDEN_DIM);
            snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
            Tensor* t_gate = get_tensor_info(name);
            snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
            Tensor* t_up = get_tensor_info(name);
            if (t_gate && t_up) {
                bool q5 = t_gate->type == TENSOR_Q5_0 && t_up->type == TENSOR_Q5_0;
                if (q5) {
                    metal_backend::g_trace_name = "ffn_gate_up_q5";
                    metal_backend::fused_ffn_gate_up_op(g_mtl_ctx,
                        t_gate->data, t_up->data,
                        scratch, gate_up, INTER_DIM, HIDDEN_DIM,
                        g_mtl_ctx.pipe_fused_ffn_gate_up_q5,
                        ((size_t)INTER_DIM * HIDDEN_DIM + 31) / 32 * 22);
                } else {
                    goto fallback_ffn;
                }
            } else {
            fallback_ffn:
                metal_backend::g_trace_name = "ffn_gate";
                matmul(t_gate, scratch, gate_up, INTER_DIM, HIDDEN_DIM);
                metal_backend::g_trace_name = "ffn_up";
                matmul(t_up, scratch, gate_up + INTER_DIM, INTER_DIM, HIDDEN_DIM);
                metal_backend::g_trace_name = "silu";
                metal_backend::elem_op(g_mtl_ctx, 1, gate_up, nullptr, INTER_DIM);
            }
            snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
            metal_backend::g_trace_name = "ffn_down";    matmul(get_tensor_info(name), gate_up, ffn_out, HIDDEN_DIM, INTER_DIM);
            metal_backend::g_trace_name = "residual2";   metal_backend::elem_op(g_mtl_ctx, 0, hidden, ffn_out, HIDDEN_DIM);
        }

        char layer_label[64];
        if (layer_end - layer_start == 1) {
            snprintf(layer_label, 64, "layer_%d", layer_start);
        } else {
            snprintf(layer_label, 64, "layers_%d-%d", layer_start, layer_end - 1);
        }
        metal_backend::g_trace_name = strdup(layer_label);
        metal_backend::metal_batch_commit();
    }

    // Output norm + logits (separate CB)
    {
        metal_backend::metal_batch_begin(g_mtl_ctx, false);
        if (norm_t) {
            metal_backend::g_trace_name = "out_copy";    metal_backend::elem_op(g_mtl_ctx, 2, scratch, hidden, HIDDEN_DIM);
            metal_backend::g_trace_name = "out_rms";     metal_backend::rmsnorm_op(g_mtl_ctx, scratch, (const float*)norm_t->data, HIDDEN_DIM);
            metal_backend::g_trace_name = "logits";      matmul(emb_t, scratch, logits, VOCAB_SIZE, HIDDEN_DIM);
        } else {
            metal_backend::g_trace_name = "logits";      matmul(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
        }
        metal_backend::g_trace_name = "logits_cb";
        metal_backend::metal_batch_commit();
    }

    metal_backend::metal_batch_wait_all();
}

int sample_token(float* logits, int top_k) {
    std::vector<std::pair<float, int>> candidates;
    for (int i = 0; i < VOCAB_SIZE; i++) candidates.push_back({logits[i], i});
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0;
    for (int i = 0; i < top_k; i++) candidates[i].first = expf(candidates[i].first);
    for (int i = 0; i < top_k; i++) sum += candidates[i].first;
    for (int i = 0; i < top_k; i++) candidates[i].first /= sum;
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < top_k; i++) {
        cumsum += candidates[i].first;
        if (r < cumsum) return candidates[i].second;
    }
    return candidates[0].second;
}

int process_prompt(float* hidden, const std::vector<int>& tokens, bool dump_layers) {
    reset_cache();
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) { printf("[ERROR] No embedding\n"); return -1; }

    for (size_t t = 0; t < tokens.size(); t++) {
        int token_id = tokens[t];
        process_embedding(hidden, token_id);

        if (dump_layers) {
            char filename[64];
            snprintf(filename, sizeof(filename), "/tmp/cpp_layer_0.bin");
            FILE* f = fopen(filename, "wb");
            fwrite(hidden, sizeof(float), HIDDEN_DIM, f);
            fclose(f);
        }

        if (g_use_metal_fused) {
            forward_all_layers_fused(hidden, (int)t);
        } else {
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                forward_layer_with_cache(hidden, layer, (int)t);
            }
        }

        if (dump_layers && t == tokens.size() - 1) {
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                char filename[64];
                snprintf(filename, sizeof(filename), "/tmp/cpp_layer_%d.bin", layer + 1);
                FILE* f = fopen(filename, "wb");
                fwrite(hidden, sizeof(float), HIDDEN_DIM, f);
                fclose(f);
            }
        }
    }
    g_seq_len = (int)tokens.size();
    return 0;
}

void generate(float* hidden, float* logits, int prompt_len, int n_tokens, int top_k) {
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) return;

    // Bootstrap: logits from prefill-processed hidden → predicts first gen token
    get_logits(logits, hidden);

    for (int gen = 0; gen < n_tokens; gen++) {
        @autoreleasepool {
        int pos = prompt_len + gen;
        if (pos >= MAX_SEQ_LEN) break;

        // logits[] contains prediction from prev iteration (or bootstrap)
        int next_token = sample_token(logits, top_k);
        printf("%d\n", next_token);
        fflush(stdout);

        process_embedding(hidden, next_token);

        if (g_use_metal_fused) {
            if (gen + 1 < n_tokens) {
                // Single CB: forward layers + logits for next token
                forward_and_logits_fused(hidden, logits, pos);
            } else {
                // Last token: just forward, no logits needed
                forward_all_layers_fused(hidden, pos);
            }
        } else {
            forward_all_layers(hidden, pos);
            // For non-fused, compute logits here for next iteration
            if (gen + 1 < n_tokens) {
                if (g_fpga_q8) get_logits_q8(logits, hidden);
                else get_logits(logits, hidden);
            }
        }
        }
    }
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.tmac> [--generate N] [--dump-layers] [--fpga] [--fpga-int16] [--fpga-q8] [--fpga-q4k] [--metal] [--perf] [--dump-tiles N]\n", argv[0]);
        printf("  Prompt tokens read from stdin, generated tokens printed to stdout\n");
        printf("  --fpga:       Same as --fpga-int16 (recommended path)\n");
        printf("  --fpga-int16: Pre-dequant all tensors to INT16 on CPU, INT16×INT16 FPGA path\n");
        printf("  --fpga-q8:    Q8_0 FPGA path for Q8_0 tensors (token embeddings), INT16 fallback\n");
        printf("  --fpga-q4k:   INT16 FPGA path for Q4_K tensors (FFN layers), INT16 fallback for others\n");
        printf("  --metal:      Use Metal GPU acceleration with per-layer sync\n");
        printf("  --metal-fused: Use Metal GPU with single-command-buffer fused path (implies --metal)\n");
        printf("  --perf:       Enable pipeline profiling (Chrome trace JSON + bottleneck analysis)\n");
        printf("  --gpu-capture: Capture GPU trace to /tmp/capture.gputrace\n");
        printf("  --dump-tiles N: Dump first N Q8 tiles for Verilog cosimulation\n");
        return 1;
    }

    g_ddr = (uint8_t*)malloc(DDR_SIZE);
    if (!g_ddr) { printf("[ERROR] Cannot allocate DDR\n"); return 1; }
    printf("[OK] Allocated %d MB DDR\n", (int)(DDR_SIZE / (1024 * 1024))); fflush(stdout);
    fpga_sim::g_timing.reset();
    if (load_tmac(argv[1]) != 0) { printf("[ERROR] Failed to load TMAC\n"); return 1; }

    float* hidden = new float[HIDDEN_DIM];
    float* logits = new float[VOCAB_SIZE];
    int generate_n = 0;
    bool dump_layers = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_n = atoi(argv[++i]);
        }
        if (strcmp(argv[i], "--dump-layers") == 0) dump_layers = true;
        if (strcmp(argv[i], "--fpga") == 0) { g_use_fpga = true; }
        if (strcmp(argv[i], "--fpga-int16") == 0) { g_use_fpga = true; }
        if (strcmp(argv[i], "--fpga-q8") == 0) { g_use_fpga = true; g_fpga_q8 = true; }
        if (strcmp(argv[i], "--fpga-q4k") == 0) { g_use_fpga = true; g_fpga_q4k = true; }
        if (strcmp(argv[i], "--perf") == 0) g_perf_enabled = true;
        if (strcmp(argv[i], "--perf-granular") == 0) { g_perf_enabled = true; g_perf_granular = true; }
        if (strcmp(argv[i], "--gpu-capture") == 0) g_gpu_capture = true;
        if (strcmp(argv[i], "--dump-tiles") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n > 0) {
                g_dump_file = fopen("/tmp/cosim_tiles.bin", "wb");
                if (g_dump_file) {
                    uint32_t header[4] = {(uint32_t)n, 0, 0, 0};
                    fwrite(header, 4, 4, g_dump_file);
                    g_dump_tiles_remaining = n;
                    printf("[COSIM] Dumping first %d Q8 tiles to /tmp/cosim_tiles.bin\n", n);
                }
            }
        }
        if (strcmp(argv[i], "--metal") == 0) {
            g_use_metal = true;
            if (!metal_backend::init(g_mtl_ctx)) {
                printf("[ERROR] Metal init failed\n");
                return 1;
            }
            g_mtl_ctx_ptr = &g_mtl_ctx;
        }
        if (strcmp(argv[i], "--metal-fused") == 0) {
            g_use_metal = true;
            g_use_metal_fused = true;
            if (!metal_backend::init(g_mtl_ctx)) {
                printf("[ERROR] Metal init failed\n");
                return 1;
            }
            g_mtl_ctx_ptr = &g_mtl_ctx;
        }
    }

    std::vector<int> tokens;
    int token;
    while (scanf("%d", &token) == 1) tokens.push_back(token);
    if (tokens.empty()) { printf("[ERROR] No prompt tokens\n"); return 1; }

    {
        if (g_perf_enabled) metal_backend::metal_trace_begin();
        PROFILE_SCOPE("process_prompt");
        if (process_prompt(hidden, tokens, dump_layers) != 0) return 1;
    }

    if (generate_n > 0) {
        if (g_gpu_capture) {
            printf("[GPU CAPTURE] Attach Xcode: Debug -> Attach to Process -> tmac_gguf\n");
            printf("  Then press Enter in this terminal...\n");
            fflush(stdout);
            for (;;) {
                FILE* tty = fopen("/dev/tty", "r");
                if (tty) {
                    int c = getc(tty); fclose(tty);
                    if (c != EOF) break;
                }
                sleep(1);
            }
            for (int tries = 0; tries < 30; tries++) {
                if (metal_backend::capture_start(g_mtl_ctx)) break;
                printf("  Waiting for Xcode to attach...\n");
                sleep(1);
            }
        }
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            generate(hidden, logits, (int)tokens.size(), generate_n, 40);
            auto t1 = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("\n[BENCH] Generated %d tokens in %.1f ms — %.1f ms/tok, %.0f tok/s\n",
                   generate_n, total_ms, total_ms / generate_n, generate_n / (total_ms / 1000.0));
        }
        if (g_gpu_capture) {
            metal_backend::capture_stop();
            printf("[GPU CAPTURE] Done. Check Xcode for the captured frame.\n");
        }
    } else {
        if (g_fpga_q8) {
            get_logits_q8(logits, hidden);
        } else {
            get_logits(logits, hidden);
        }
        printf("[");
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < VOCAB_SIZE; i++) top.push_back({logits[i], i});
        std::partial_sort(top.begin(), top.begin() + 10, top.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < 10; i++) {
            printf("{\"id\":%d,\"logit\":%.3f}", top[i].second, top[i].first);
            if (i < 9) printf(",");
        }
        printf("]\n");
    }

    if (g_use_metal_fused) {
        printf("[INFO] Fused Metal path (1 CB per forward pass)\n");
    } else if (g_use_metal) {
        printf("[INFO] Batched Metal path (4 CBs per layer)\n");
    }
    if (g_perf_enabled && g_use_metal) {
        printf("[INFO] Metal full-forward: rows=%d cols=%d total=%lld\n",
               HIDDEN_DIM, HIDDEN_DIM, (long long)(NUM_LAYERS * 2 * HIDDEN_DIM));
    }
    if (g_use_fpga) {
        fpga_sim::g_timing.total_fpga_cycles += fpga_sim::accel().total_cycles();
        fpga_sim::g_timing.report();
    }
    if (g_perf_enabled) {
        if (g_use_metal) {
            metal_backend::metal_trace_end();
            metal_backend::metal_trace_report();
        }
        print_trace_summary();
        dump_chrome_trace("/tmp/pipeline_trace.json");
    }
    free(g_ddr);
    free(g_tensors);
    delete[] hidden;
    delete[] logits;
    if (g_use_metal) metal_backend::shutdown(g_mtl_ctx);
    return 0;
}