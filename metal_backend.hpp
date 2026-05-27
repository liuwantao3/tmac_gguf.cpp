#pragma once

#ifdef __APPLE__
#include <Metal/Metal.h>
#else
typedef void* id;
#define nil NULL
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <unordered_map>

namespace metal_backend {

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipe_fp32 = nil;
    id<MTLComputePipelineState> pipe_q8_0 = nil;
    id<MTLComputePipelineState> pipe_q5_0 = nil;
    id<MTLComputePipelineState> pipe_q4_k = nil;
    id<MTLComputePipelineState> pipe_q6_k = nil;
    id<MTLComputePipelineState> pipe_elem_add = nil;
    id<MTLComputePipelineState> pipe_elem_silu = nil;
    id<MTLComputePipelineState> pipe_elem_write = nil;
    id<MTLComputePipelineState> pipe_rope = nil;
    id<MTLComputePipelineState> pipe_attn = nil;
    id<MTLComputePipelineState> pipe_rmsnorm = nil;
    id<MTLComputePipelineState> pipe_fused_qkv_q5 = nil; // V=Q5_0
    id<MTLComputePipelineState> pipe_fused_qkv_q8 = nil; // V=Q8_0
    id<MTLComputePipelineState> pipe_fused_ffn_gate_up = nil;
    bool initialized = false;
};

// ── Batch state + cached resources (global, single-threaded usage) ──

// Weight buffer cache: maps data pointer → MTLBuffer (reused across dispatches)
static std::unordered_map<const uint8_t*, id<MTLBuffer>> g_w_cache;

// General buffer cache: maps any pointer → MTLBuffer (for input/output/intermediate data)
static std::unordered_map<const void*, id<MTLBuffer>> g_buf_cache;

inline id<MTLBuffer> wrap_buffer(Context& ctx, const void* ptr, size_t bytes) {
    if (!ptr) return nil;
    auto it = g_buf_cache.find(ptr);
    if (it != g_buf_cache.end()) {
        if ([it->second length] >= bytes) return it->second;
        it->second = [ctx.device newBufferWithBytesNoCopy:(void*)ptr
                      length:bytes options:MTLStorageModeShared deallocator:nil];
        return it->second;
    }
    id<MTLBuffer> buf = [ctx.device newBufferWithBytesNoCopy:(void*)ptr
                         length:bytes options:MTLStorageModeShared deallocator:nil];
    g_buf_cache[ptr] = buf;
    return buf;
}

// Pre-allocated params buffer pool: each dispatch writes params at a unique offset
static id<MTLBuffer> g_params_buf = nil;
static int g_params_offset = 0;
static constexpr int PARAMS_POOL_SIZE = 16384; // enough for all dispatches in one fused forward

static id<MTLCommandBuffer> g_batch_cb = nil;
static id<MTLComputeCommandEncoder> g_batch_enc = nil;
static id<MTLComputePipelineState> g_batch_pipe = nil;

// ── Multi-CB support (commit without wait, then wait-all at end) ──
static constexpr int MAX_CBS = 32;
static id<MTLCommandBuffer> g_committed_cbs[MAX_CBS] = {};
static int g_num_committed_cbs = 0;

// ── GPU timestamp tracing (optional, via --perf) ──
static bool g_trace_enabled = false;
static const char* g_trace_name = nullptr;
static id<MTLCounterSampleBuffer> g_trace_buf = nil;
static constexpr int MAX_TRACE_SAMPLES = 512;
static const char* g_trace_names[MAX_TRACE_SAMPLES];
static int g_trace_idx = 0;

inline void metal_trace_init(Context& ctx) {
    for (id<MTLCounterSet> cs in ctx.device.counterSets) {
        if ([[cs name] caseInsensitiveCompare:@"timestamp"] == NSOrderedSame) {
            MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
            desc.counterSet = cs;
            desc.storageMode = MTLStorageModeShared;
            desc.sampleCount = MAX_TRACE_SAMPLES;
            NSError* err;
            g_trace_buf = [ctx.device newCounterSampleBufferWithDescriptor:desc error:&err];
            if (!g_trace_buf) printf("[WARN] CounterBuffer: %s\n", [[err description] UTF8String]);
            return;
        }
    }
    // Fallback: try first counter set
    if (ctx.device.counterSets.count > 0) {
        MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = ctx.device.counterSets[0];
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = MAX_TRACE_SAMPLES;
        NSError* err;
        g_trace_buf = [ctx.device newCounterSampleBufferWithDescriptor:desc error:&err];
        if (!g_trace_buf) printf("[WARN] CounterBuffer: %s\n", [[err description] UTF8String]);
        return;
    }
    printf("[WARN] No timestamp counter set available\n");
}

inline void metal_trace_begin() {
    g_trace_enabled = true;
    g_trace_idx = 0;
}

inline void metal_trace_end() { g_trace_enabled = false; }

inline void metal_trace_sample() {
    // GPU counter sampling not supported on M1 compute encoders
}

inline void metal_trace_report() {
    if (!g_trace_buf || g_trace_idx < 2) { printf("[TRACE] No GPU timing data\n"); return; }
    NSData* res = [g_trace_buf resolveCounterRange:NSMakeRange(0, g_trace_idx)];
    const uint64_t* data = (const uint64_t*)[res bytes];
    printf("\n[GPU PER-KERNEL TIMING]\n");
    double total_ns = 0;
    for (int i = 0; i < g_trace_idx - 1; i++) {
        double ns = (double)(data[i+1] - data[i]);
        total_ns += ns;
        printf("  %-25s %8.1f us\n", g_trace_names[i], ns / 1000.0);
    }
    printf("  %-25s %8.1f us  (%.2f ms)\n", "TOTAL", total_ns, total_ns / 1000000.0);
}

inline void metal_batch_begin(Context& ctx, bool reset_offset = true) {
    g_batch_cb = [ctx.queue commandBuffer];
    g_batch_enc = nil;
    g_batch_pipe = nil;
    if (reset_offset) g_params_offset = 0;
    if (!g_params_buf)
        g_params_buf = [ctx.device newBufferWithLength:PARAMS_POOL_SIZE
                         options:MTLStorageModeShared];
}

// Commit current CB without waiting (appends to committed list for batch_wait_all)
inline void metal_batch_commit() {
    if (g_batch_enc) { [g_batch_enc endEncoding]; g_batch_enc = nil; g_batch_pipe = nil; }
    if (g_batch_cb) {
        [g_batch_cb commit];
        if (g_num_committed_cbs < MAX_CBS) {
            g_committed_cbs[g_num_committed_cbs++] = g_batch_cb;
        }
        g_batch_cb = nil;
    }
}

// Wait for all committed CBs and clear the list
inline void metal_batch_wait_all() {
    for (int i = 0; i < g_num_committed_cbs; i++) {
        [g_committed_cbs[i] waitUntilCompleted];
        g_committed_cbs[i] = nil;
    }
    g_num_committed_cbs = 0;
}

inline void metal_batch_end() {
    if (g_batch_enc) { [g_batch_enc endEncoding]; g_batch_enc = nil; g_batch_pipe = nil; }
    if (g_batch_cb) {
        [g_batch_cb commit];
        [g_batch_cb waitUntilCompleted];
        g_batch_cb = nil;
    }
    // Also flush any previously committed CBs (safety)
    metal_batch_wait_all();
}

inline void metal_batch_ensure_encoder(Context& ctx, id<MTLComputePipelineState> pipe) {
    if (pipe == g_batch_pipe && g_batch_enc) return;
    if (g_batch_enc) { [g_batch_enc endEncoding]; g_batch_enc = nil; }
    if (!g_batch_cb) g_batch_cb = [ctx.queue commandBuffer];
    g_batch_enc = [g_batch_cb computeCommandEncoder];
    [g_batch_enc setComputePipelineState:pipe];
    g_batch_pipe = pipe;
}

// Dispatch a single matmul: into the current batch if one is active,
// otherwise creates a standalone command buffer, commits, and waits.
inline void metal_batch_dispatch(
    Context& ctx, id<MTLComputePipelineState> pipe,
    int rows, int cols,
    const uint8_t* w_data, size_t w_bytes,
    const float* x_data, float* y_data, bool is_simd)
{
    bool batch_was_active = (g_batch_cb != nil || g_batch_enc != nil);
    metal_batch_ensure_encoder(ctx, pipe);

    // Cached weight buffer: reuse MTLBuffer per tensor pointer
    auto wit = g_w_cache.find(w_data);
    id<MTLBuffer> bufW;
    if (wit != g_w_cache.end()) {
        bufW = wit->second;
    } else {
        bufW = [ctx.device newBufferWithBytesNoCopy:(void*)w_data
                length:w_bytes options:MTLStorageModeShared deallocator:nil];
        g_w_cache[w_data] = bufW;
    }

    id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:(void*)x_data
                         length:(size_t)cols * 4 options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:(void*)y_data
                         length:(size_t)rows * 4 options:MTLStorageModeShared deallocator:nil];

    // Params from pre-allocated pool (write at unique offset)
    if (!g_params_buf)
        g_params_buf = [ctx.device newBufferWithLength:PARAMS_POOL_SIZE
                         options:MTLStorageModeShared];
    int slot = batch_was_active ? g_params_offset : 0;
    if (batch_was_active) g_params_offset += 8;
    int* params = (int*)((uint8_t*)[g_params_buf contents] + slot);
    params[0] = rows;
    params[1] = cols;

    [g_batch_enc setBuffer:bufW offset:0 atIndex:0];
    [g_batch_enc setBuffer:bufX offset:0 atIndex:1];
    [g_batch_enc setBuffer:bufY offset:0 atIndex:2];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:3];

    metal_trace_sample();
    if (is_simd) {
        int total = ((rows + 3) / 4) * 64;
        [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
        int tg = std::min(rows, 64);
        [g_batch_enc dispatchThreads:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    // Standalone call: commit+wait here (batch-active calls rely on caller to end)
    if (!batch_was_active) metal_batch_end();
}

constexpr const char* kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

// ── FP32 ──────────────────────────────────────────────────
kernel void mul_mat_fp32(device const float* A [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    if (gid >= (uint)rows) return;
    float sum = 0;
    for (int j = 0; j < cols; j++)
        sum += A[(uint)gid * cols + j] * x[j];
    y[gid] = sum;
}

// ── Q8_0 (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Block: [half scale][int8×32] = 34 bytes per 32 values
kernel void mul_mat_q8_0(device const uint8_t* W [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = cols / 32;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 34;
            float d = (float)*(device const half*)(W + base);
            float q = (float)*(device const int8_t*)(W + base + 2 + lane);
            sumf[r] += q * d * xv;
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}

// ── Q5_0 (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 1 element per 32-element block
kernel void mul_mat_q5_0(device const uint8_t* W [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = cols / 32;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    float sumf[2] = {0.f, 0.f};
    int lane_half = lane < 16 ? lane : lane - 16;
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 22;
            float d = (float)*(device const half*)(W + base);
            uint qh = (uint)W[base + 2] | ((uint)W[base + 3] << 8)
                     | ((uint)W[base + 4] << 16) | ((uint)W[base + 5] << 24);
            uint qs_byte = W[base + 6 + lane_half];
            uint ql = lane < 16 ? (qs_byte & 0xF) : (qs_byte >> 4);
            int q = (int)((((qh >> lane) & 1) << 4) | ql) - 16;
            sumf[r] += (float)q * d * xv;
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}

// ── Q4_K (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 8 elements per 256-element block
kernel void mul_mat_q4_k(device const uint8_t* W [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = (cols + 255) / 256;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    uint pos = lane * 8;
    uint sub = pos / 32;
    uint nib = pos % 32;
    uint byte_base = (sub / 2) * 32 + nib;
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        ulong x_ofs = (ulong)ib * 256 + pos;
        float xv[8];
        int k_lim = 8;
        if (ib == nb - 1) {
            int rem = cols - ib * 256 - (int)pos;
            k_lim = rem < 8 ? max(0, rem) : 8;
        }
        for (int k = 0; k < k_lim; k++) xv[k] = x[x_ofs + k];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 144;
            float d    = (float)*(device const half*)(W + base);
            float dmin = (float)*(device const half*)(W + base + 2);
            device const uint8_t* scales = W + base + 4;
            device const uint8_t* qs = W + base + 16;
            uint sc, m;
            if (sub < 4) {
                sc = scales[sub] & 63;  m = scales[sub + 4] & 63;
            } else {
                sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
                m  = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
            }
            float d_sc = d * (float)sc;
            float dmin_m = dmin * (float)m;
            device const uint8_t* qb = qs + byte_base;
            for (int k = 0; k < k_lim; k++) {
                uint8_t byte = qb[k];
                uint q4 = (sub & 1) ? (byte >> 4) : (byte & 0xF);
                sumf[r] += (d_sc * (float)q4 - dmin_m) * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}

// ── Q6_K (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 8 elements per 256-element block
kernel void mul_mat_q6_k(device const uint8_t* W [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = (cols + 255) / 256;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    uint pos = lane * 8;
    uint hf = pos / 128;
    uint pos_in_half = pos % 128;
    uint sub = pos_in_half / 32;
    uint l_beg = pos_in_half % 32;
    uint is = l_beg / 16;
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        ulong x_ofs = (ulong)ib * 256 + pos;
        float xv[8];
        int k_lim = 8;
        if (ib == nb - 1) {
            int rem = cols - ib * 256 - (int)pos;
            k_lim = rem < 8 ? max(0, rem) : 8;
        }
        for (int k = 0; k < k_lim; k++) xv[k] = x[x_ofs + k];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 210;
            float d = (float)*(device const half*)(W + base + 208);
            device const uint8_t* ql = W + base + hf * 64 + l_beg + (sub & 1) * 32;
            device const uint8_t* qh = W + base + 128 + hf * 32 + l_beg;
            int scale = (int)*(device const int8_t*)(W + base + 192 + hf * 8 + is + sub * 2);
            for (int k = 0; k < k_lim; k++) {
                uint ql_byte = ql[k];
                uint ql_nib = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
                uint qh_bits = (qh[k] >> (sub * 2)) & 0x3;
                int q6 = (int)((qh_bits << 4) | ql_nib) - 32;
                sumf[r] += (float)scale * (float)q6 * d * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}
// ── Fused QKV: Q5_0 for Q+K, Q5_0 or Q8_0 for V (function-constant v_type) ──
// v_type=0: V also Q5_0, v_type=1: V is Q8_0
constant int v_type [[function_constant(1)]];
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
    int q_rows = params[0], k_rows = params[1], v_rows = params[2];
    int cols = params[3];
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int total_rows = q_rows + k_rows + v_rows;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= total_rows) return;
    int nr = min(2, total_rows - first_row);
    int nb = cols / 32;
    float sumf[2] = {0.f, 0.f};
    int lane_half = lane < 16 ? lane : lane - 16;
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            int global_row = first_row + r;
            device const uint8_t* W;
            int local_row;
            if (global_row < q_rows) {
                W = W_q; local_row = global_row;
            } else if (global_row < q_rows + k_rows) {
                W = W_k; local_row = global_row - q_rows;
            } else {
                W = W_v; local_row = global_row - q_rows - k_rows;
            }
            if (global_row < q_rows + k_rows || v_type == 0) {
                ulong base = ((ulong)local_row * nb + ib) * 22;
                float d = (float)*(device const half*)(W + base);
                uint qh = (uint)W[base+2]|((uint)W[base+3]<<8)|((uint)W[base+4]<<16)|((uint)W[base+5]<<24);
                uint qs_byte = W[base + 6 + lane_half];
                uint ql = lane < 16 ? (qs_byte & 0xF) : (qs_byte >> 4);
                int q = (int)((((qh >> lane) & 1) << 4) | ql) - 16;
                sumf[r] += (float)q * d * xv;
            } else {
                ulong base = ((ulong)local_row * nb + ib) * 34;
                float d = (float)*(device const half*)(W + base);
                float q = (float)*(device const int8_t*)(W + base + 2 + lane);
                sumf[r] += q * d * xv;
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) {
            int global_row = first_row + r;
            if (global_row < q_rows) y_q[global_row] = total;
            else if (global_row < q_rows + k_rows) y_k[global_row - q_rows] = total;
            else y_v[global_row - q_rows - k_rows] = total;
        }
    }
}
// ── Fused FFN gate+up (both Q6_K, same dims) ──
kernel void kernel_fused_ffn_gate_up(
    device const uint8_t* W_gate [[buffer(0)]],
    device const uint8_t* W_up [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int rows_per = params[0]; int cols = params[1];
    int total_rows = rows_per * 2;
    int nb = (cols + 255) / 256;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= total_rows) return;
    int nr = min(2, total_rows - first_row);
    uint pos = lane * 8;
    uint hf = pos / 128;
    uint pos_in_half = pos % 128;
    uint sub = pos_in_half / 32;
    uint l_beg = pos_in_half % 32;
    uint is = l_beg / 16;
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        ulong x_ofs = (ulong)ib * 256 + pos;
        float xv[8];
        int k_lim = 8;
        if (ib == nb - 1) {
            int rem = cols - ib * 256 - (int)pos;
            k_lim = rem < 8 ? max(0, rem) : 8;
        }
        for (int k = 0; k < k_lim; k++) xv[k] = x[x_ofs + k];
        for (int r = 0; r < nr; r++) {
            int global_row = first_row + r;
            device const uint8_t* W = global_row < rows_per ? W_gate : W_up;
            int local_row = global_row < rows_per ? global_row : global_row - rows_per;
            ulong base = ((ulong)local_row * nb + ib) * 210;
            float d = (float)*(device const half*)(W + base + 208);
            device const uint8_t* ql = W + base + hf * 64 + l_beg + (sub & 1) * 32;
            device const uint8_t* qh = W + base + 128 + hf * 32 + l_beg;
            int scale = (int)*(device const int8_t*)(W + base + 192 + hf * 8 + is + sub * 2);
            for (int k = 0; k < k_lim; k++) {
                uint ql_byte = ql[k];
                uint ql_nib = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
                uint qh_bits = (qh[k] >> (sub * 2)) & 0x3;
                int q6 = (int)((qh_bits << 4) | ql_nib) - 32;
                sumf[r] += (float)scale * (float)q6 * d * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}
//
// ── Fused-pipeline kernels ──
//

// Element-wise: specialized via function constant kernel_op
// 0 = add, 1 = silu_x_up, 2 = cache_write
constant int kernel_op [[function_constant(0)]];
kernel void kernel_elem(device float* data [[buffer(0)]],
                         device const float* aux [[buffer(1)]],
                         constant int* params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    if (gid >= (uint)dim) return;
    if (kernel_op == 0) {
        data[gid] += aux[gid];
    } else if (kernel_op == 1) {
        float x = data[gid];
        data[gid] = (x / (1 + exp(-x))) * data[gid + dim];
    } else if (kernel_op == 2) {
        data[gid] = aux[gid];
    }
}

// Rope: in-place rotation
kernel void kernel_rope(device float* data [[buffer(0)]],
                         constant int* params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    int n_heads = params[0];
    int head_dim = params[1];
    int pos = params[2];
    float base = as_type<float>(params[3]);
    int pairs = head_dim / 2;
    int head = gid / pairs;
    int pair = gid % pairs;
    if (head >= n_heads) return;
    int idx = head * head_dim + pair * 2;
    float theta = 1.0 / pow(base, (float)(pair * 2) / head_dim);
    float angle = (float)pos * theta;
    float c = cos(angle), s = sin(angle);
    float x0 = data[idx], x1 = data[idx + 1];
    data[idx] = x0 * c - x1 * s;
    data[idx + 1] = x0 * s + x1 * c;
}

// RMSNorm: in-place, 32-thread single-SG dispatch
kernel void kernel_rmsnorm(device float* data [[buffer(0)]],
                            device const float* weight [[buffer(1)]],
                            constant int* params [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    float my_sum = 0;
    for (int i = gid; i < dim; i += 32)
        my_sum += data[i] * data[i];
    float total = simd_sum(my_sum);
    float rms = sqrt(total / (float)dim + 1e-6f);
    for (int i = gid; i < dim; i += 32)
        data[i] = (data[i] / rms) * weight[i];
}

// Attention: flash-attention single-token GQA, 32-thread TG (1 simdgroup).
// Processes KV in tiles of 8 with online softmax — single pass, no threadgroup memory, no barriers.
kernel void kernel_attn(device const float* Q [[buffer(0)]],
                         device const float* K_cache [[buffer(1)]],
                         device const float* V_cache [[buffer(2)]],
                         device float* output [[buffer(3)]],
                         constant int* params [[buffer(4)]],
                         uint head [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    int n_head = params[0];
    int n_kv_head = params[1];
    int head_dim = params[2];
    int past_len = params[3];
    if (head >= (uint)n_head) return;
    int kv_head = head / (n_head / n_kv_head);
    int d0 = lane * 2, d1 = lane * 2 + 1;
    if (d0 >= head_dim) return;

    float my_q0 = Q[(uint)head * head_dim + d0];
    float my_q1 = Q[(uint)head * head_dim + d1];
    float scale = 1.0 / sqrt((float)head_dim);

    // Flash attention: online softmax, single pass, TILE=8
    float O0 = 0, O1 = 0, m = -INFINITY, d = 0;
    int n_pos = past_len + 1;

    for (int tile_start = 0; tile_start < n_pos; tile_start += 8) {
        int tile_end = min(tile_start + 8, n_pos);
        int tile_sz = tile_end - tile_start;

        // Compute scores for this tile
        float s[8];
        for (int t = 0; t < tile_sz; t++) {
            int p = tile_start + t;
            ulong cache_ofs = ((ulong)p * n_kv_head + kv_head) * head_dim;
            float k0 = K_cache[cache_ofs + d0];
            float k1 = K_cache[cache_ofs + d1];
            s[t] = simd_sum(my_q0 * k0 + my_q1 * k1) * scale;
        }

        // Online softmax: find tile max, rescale previous accum, add new contributions
        float m_new = m;
        for (int t = 0; t < tile_sz; t++) m_new = max(m_new, s[t]);
        float old_scale = exp(m - m_new);
        O0 *= old_scale; O1 *= old_scale; d *= old_scale;

        for (int t = 0; t < tile_sz; t++) {
            float e = exp(s[t] - m_new);
            int p = tile_start + t;
            ulong cache_ofs = ((ulong)p * n_kv_head + kv_head) * head_dim;
            O0 += e * V_cache[cache_ofs + d0];
            O1 += e * V_cache[cache_ofs + d1];
            d += e;
        }
        m = m_new;
    }

    output[(uint)head * head_dim + d0] = O0 / d;
    output[(uint)head * head_dim + d1] = O1 / d;
}
)";

inline bool init(Context& ctx) {
    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) { printf("[METAL] No Metal device\n"); return false; }
    ctx.queue = [ctx.device newCommandQueue];
    if (!ctx.queue) { printf("[METAL] No command queue\n"); return false; }

    NSError* err = nil;
    ctx.library = [ctx.device newLibraryWithSource:@(kernel_source) options:nil error:&err];
    if (err) { printf("[METAL] Library: %s\n", [[err localizedDescription] UTF8String]); return false; }

    auto get_pipe = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [ctx.library newFunctionWithName:@(name)];
        if (!fn) { printf("[METAL] No function: %s\n", name); return nil; }
        id<MTLComputePipelineState> p = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (err) { printf("[METAL] Pipeline %s: %s\n", name, [[err localizedDescription] UTF8String]); return nil; }
        return p;
    };

    ctx.pipe_fp32 = get_pipe("mul_mat_fp32");
    ctx.pipe_q8_0 = get_pipe("mul_mat_q8_0");
    ctx.pipe_q5_0 = get_pipe("mul_mat_q5_0");
    ctx.pipe_q4_k = get_pipe("mul_mat_q4_k");
    ctx.pipe_q6_k = get_pipe("mul_mat_q6_k");
    if (!ctx.pipe_q5_0 || !ctx.pipe_q4_k || !ctx.pipe_q6_k) return false;

    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_pipe = [&](int op_val) -> id<MTLComputePipelineState> {
            [cv setConstantValue:&op_val type:MTLDataTypeInt atIndex:0];
            id<MTLFunction> fn = [ctx.library newFunctionWithName:@"kernel_elem"
                                   constantValues:cv error:&err];
            if (err) return nil;
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_elem_add  = make_pipe(0);
        ctx.pipe_elem_silu = make_pipe(1);
        ctx.pipe_elem_write = make_pipe(2);
        if (!ctx.pipe_elem_add || !ctx.pipe_elem_silu || !ctx.pipe_elem_write) return false;
    }
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_qkv = [&](int v) -> id<MTLComputePipelineState> {
            [cv setConstantValue:&v type:MTLDataTypeInt atIndex:1];
            id<MTLFunction> fn = [ctx.library newFunctionWithName:@"kernel_fused_qkv"
                                   constantValues:cv error:&err];
            if (err) { printf("[METAL] fused_qkv fn: %s\n", [[err description] UTF8String]); return nil; }
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_fused_qkv_q5 = make_qkv(0);
        ctx.pipe_fused_qkv_q8 = make_qkv(1);
        if (!ctx.pipe_fused_qkv_q5 || !ctx.pipe_fused_qkv_q8) return false;
    }
    ctx.pipe_fused_ffn_gate_up = get_pipe("kernel_fused_ffn_gate_up");
    if (!ctx.pipe_fused_ffn_gate_up) return false;
    ctx.pipe_rope = get_pipe("kernel_rope");
    ctx.pipe_attn = get_pipe("kernel_attn");
    ctx.pipe_rmsnorm = get_pipe("kernel_rmsnorm");
    if (!ctx.pipe_rope || !ctx.pipe_attn || !ctx.pipe_rmsnorm) return false;

    ctx.initialized = true;
    printf("[METAL] Device: %s\n", [[ctx.device name] UTF8String]);
    printf("[METAL] Max threads/group: %zu\n", ctx.device.maxThreadsPerThreadgroup.width);
    metal_trace_init(ctx);
    return true;
}

inline void shutdown(Context& ctx) { ctx.initialized = false; }

// ── GPU trace capture (writes .gputrace for Metal Debugger/Instruments) ──
inline bool capture_start(Context& ctx) {
    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
    if (![mgr supportsDestination:MTLCaptureDestinationDeveloperTools]) return false;
    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = ctx.device;
    desc.destination = MTLCaptureDestinationDeveloperTools;
    NSError* err;
    return [mgr startCaptureWithDescriptor:desc error:&err];
}

inline void capture_stop() {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

// ── Dispatch helpers (batch-based, no per-call commit/wait) ──

inline void matmul_q8_0(Context& ctx, int rows, int cols,
                        const uint8_t* w, const float* x, float* y) {
    size_t wb = ((size_t)rows * cols + 31) / 32 * 34;
    metal_batch_dispatch(ctx, ctx.pipe_q8_0, rows, cols, w, wb, x, y, true);
}

inline void matmul_q5_0(Context& ctx, int rows, int cols,
                        const uint8_t* w, const float* x, float* y) {
    size_t wb = ((size_t)rows * cols + 31) / 32 * 22;
    metal_batch_dispatch(ctx, ctx.pipe_q5_0, rows, cols, w, wb, x, y, true);
}

inline void matmul_q4_k(Context& ctx, int rows, int cols,
                        const uint8_t* w, const float* x, float* y) {
    size_t wb = ((size_t)rows * cols + 255) / 256 * 144;
    metal_batch_dispatch(ctx, ctx.pipe_q4_k, rows, cols, w, wb, x, y, true);
}

inline void matmul_q6_k(Context& ctx, int rows, int cols,
                        const uint8_t* w, const float* x, float* y) {
    size_t wb = ((size_t)rows * cols + 255) / 256 * 210;
    metal_batch_dispatch(ctx, ctx.pipe_q6_k, rows, cols, w, wb, x, y, true);
}

// ── Fused-kernel dispatch helpers ──

inline void elem_op(Context& ctx, int op, float* data, const float* aux, int dim) {
    id<MTLComputePipelineState> pipe;
    if (op == 0) pipe = ctx.pipe_elem_add;
    else if (op == 1) pipe = ctx.pipe_elem_silu;
    else pipe = ctx.pipe_elem_write;
    metal_batch_ensure_encoder(ctx, pipe);
    id<MTLBuffer> buf_d = wrap_buffer(ctx, data, (size_t)(op == 1 ? 2 : 1) * dim * 4);
    id<MTLBuffer> buf_a;
    if (op == 1 || !aux) {
        buf_a = buf_d; // dummy — kernel doesn't read aux for op=1
    } else {
        buf_a = wrap_buffer(ctx, aux, (size_t)dim * 4);
    }
    int slot = g_params_offset; g_params_offset += 4;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = dim;
    [g_batch_enc setBuffer:buf_d offset:0 atIndex:0];
    [g_batch_enc setBuffer:buf_a offset:0 atIndex:1];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:2];
    metal_trace_sample();
    int total = ((dim + 63) / 64) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

inline void rope_op(Context& ctx, float* data, int n_heads, int head_dim, int pos, float base) {
    metal_batch_ensure_encoder(ctx, ctx.pipe_rope);
    id<MTLBuffer> buf_d = wrap_buffer(ctx, data, (size_t)n_heads * head_dim * 4);
    int slot = g_params_offset; g_params_offset += 16;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = n_heads; pp[1] = head_dim; pp[2] = pos; pp[3] = 0;
    memcpy(&pp[3], &base, 4);
    [g_batch_enc setBuffer:buf_d offset:0 atIndex:0];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:2];
    metal_trace_sample();
    int total = ((n_heads * head_dim / 2 + 63) / 64) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

inline void rmsnorm_op(Context& ctx, float* data, const float* weight, int dim) {
    metal_batch_ensure_encoder(ctx, ctx.pipe_rmsnorm);
    id<MTLBuffer> buf_d = wrap_buffer(ctx, data, (size_t)dim * 4);
    id<MTLBuffer> buf_w = wrap_buffer(ctx, weight, (size_t)dim * 4);
    int slot = g_params_offset; g_params_offset += 4;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = dim;
    [g_batch_enc setBuffer:buf_d offset:0 atIndex:0];
    [g_batch_enc setBuffer:buf_w offset:0 atIndex:1];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:2];
    metal_trace_sample();
    [g_batch_enc dispatchThreads:MTLSizeMake(32, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

inline void attention_op(Context& ctx,
                         const float* Q, const float* K_cache, const float* V_cache,
                         float* output, int n_head, int n_kv_head, int head_dim, int past_len) {
    metal_batch_ensure_encoder(ctx, ctx.pipe_attn);
    id<MTLBuffer> buf_Q = wrap_buffer(ctx, Q, (size_t)n_head * head_dim * 4);
    id<MTLBuffer> buf_K = wrap_buffer(ctx, K_cache, (size_t)256 * n_kv_head * head_dim * 4);
    id<MTLBuffer> buf_V = wrap_buffer(ctx, V_cache, (size_t)256 * n_kv_head * head_dim * 4);
    id<MTLBuffer> buf_O = wrap_buffer(ctx, output, (size_t)n_head * head_dim * 4);
    int slot = g_params_offset; g_params_offset += 16;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = n_head; pp[1] = n_kv_head; pp[2] = head_dim; pp[3] = past_len;
    [g_batch_enc setBuffer:buf_Q offset:0 atIndex:0];
    [g_batch_enc setBuffer:buf_K offset:0 atIndex:1];
    [g_batch_enc setBuffer:buf_V offset:0 atIndex:2];
    [g_batch_enc setBuffer:buf_O offset:0 atIndex:3];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:4];
    metal_trace_sample();
    [g_batch_enc dispatchThreadgroups:MTLSizeMake(n_head, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

// ── Fused QKV dispatch (replaces 3 matmuls) ──
inline void fused_qkv_op(Context& ctx,
                          const uint8_t* W_q, const uint8_t* W_k, const uint8_t* W_v,
                          const float* x,
                          float* y_q, float* y_k, float* y_v,
                          int q_rows, int k_rows, int v_rows, int cols, bool v_is_q8) {
    id<MTLComputePipelineState> pipe = v_is_q8 ? ctx.pipe_fused_qkv_q8 : ctx.pipe_fused_qkv_q5;
    metal_batch_ensure_encoder(ctx, pipe);
    id<MTLBuffer> bufWq = wrap_buffer(ctx, W_q, (size_t)q_rows * (cols / 32) * 22);
    id<MTLBuffer> bufWk = wrap_buffer(ctx, W_k, (size_t)k_rows * (cols / 32) * 22);
    size_t v_bytes = v_is_q8 ? (size_t)v_rows * (cols / 32) * 34
                             : (size_t)v_rows * (cols / 32) * 22;
    id<MTLBuffer> bufWv = wrap_buffer(ctx, W_v, v_bytes);
    id<MTLBuffer> bufX  = wrap_buffer(ctx, x, (size_t)cols * 4);
    id<MTLBuffer> bufYq = wrap_buffer(ctx, y_q, (size_t)q_rows * 4);
    id<MTLBuffer> bufYk = wrap_buffer(ctx, y_k, (size_t)k_rows * 4);
    id<MTLBuffer> bufYv = wrap_buffer(ctx, y_v, (size_t)v_rows * 4);
    int slot = g_params_offset; g_params_offset += 16;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = q_rows; pp[1] = k_rows; pp[2] = v_rows; pp[3] = cols;
    [g_batch_enc setBuffer:bufWq offset:0 atIndex:0];
    [g_batch_enc setBuffer:bufWk offset:0 atIndex:1];
    [g_batch_enc setBuffer:bufWv offset:0 atIndex:2];
    [g_batch_enc setBuffer:bufX  offset:0 atIndex:3];
    [g_batch_enc setBuffer:bufYq offset:0 atIndex:4];
    [g_batch_enc setBuffer:bufYk offset:0 atIndex:5];
    [g_batch_enc setBuffer:bufYv offset:0 atIndex:6];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:7];
    metal_trace_sample();
    int total = ((q_rows + k_rows + v_rows + 3) / 4) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

inline void fused_ffn_gate_up_op(Context& ctx,
                                  const uint8_t* W_gate, const uint8_t* W_up,
                                  const float* x, float* y,
                                  int rows_per, int cols) {
    id<MTLComputePipelineState> pipe = ctx.pipe_fused_ffn_gate_up;
    metal_batch_ensure_encoder(ctx, pipe);
    size_t wb = ((size_t)rows_per * cols + 255) / 256 * 210;
    id<MTLBuffer> bufWg = wrap_buffer(ctx, W_gate, wb);
    id<MTLBuffer> bufWu = wrap_buffer(ctx, W_up, wb);
    id<MTLBuffer> bufX  = wrap_buffer(ctx, x, (size_t)cols * 4);
    id<MTLBuffer> bufY  = wrap_buffer(ctx, y, (size_t)rows_per * 2 * 4);
    int slot = g_params_offset; g_params_offset += 16;
    int* pp = (int*)((uint8_t*)[g_params_buf contents] + slot);
    pp[0] = rows_per; pp[1] = cols;
    [g_batch_enc setBuffer:bufWg offset:0 atIndex:0];
    [g_batch_enc setBuffer:bufWu offset:0 atIndex:1];
    [g_batch_enc setBuffer:bufX  offset:0 atIndex:2];
    [g_batch_enc setBuffer:bufY  offset:0 atIndex:3];
    [g_batch_enc setBuffer:g_params_buf offset:slot atIndex:4];
    metal_trace_sample();
    int total = ((rows_per * 2 + 3) / 4) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

} // namespace metal_backend
