#pragma once

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <simd/simd.h>
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
    id<MTLComputePipelineState> pipe_q8_0_simd = nil;
    id<MTLComputePipelineState> pipe_q5_0 = nil;
    id<MTLComputePipelineState> pipe_q4_k = nil;
    id<MTLComputePipelineState> pipe_q6_k = nil;
    id<MTLComputePipelineState> pipe_elem_add = nil;
    id<MTLComputePipelineState> pipe_elem_silu = nil;
    id<MTLComputePipelineState> pipe_elem_write = nil;
    id<MTLComputePipelineState> pipe_rope = nil;
    id<MTLComputePipelineState> pipe_attn = nil;
    id<MTLComputePipelineState> pipe_flash_attn = nil;
    id<MTLComputePipelineState> pipe_rmsnorm = nil;
    id<MTLComputePipelineState> pipe_fused_qkv_q5 = nil; // V=Q5_0
    id<MTLComputePipelineState> pipe_fused_qkv_q8 = nil; // V=Q8_0
    id<MTLComputePipelineState> pipe_fused_ffn_gate_up_q5 = nil; // Q5_0 version

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
static constexpr int MAX_CBS = 512;
static id<MTLCommandBuffer> g_committed_cbs[MAX_CBS] = {};
static const char* g_committed_labels[MAX_CBS] = {};
static int g_num_committed_cbs = 0;

// ── Per-CB GPU timing (via GPUStartTime/GPUEndTime, set via --perf) ──
static bool g_trace_enabled = false;
static const char* g_trace_name = nullptr;
static const char* g_cb_labels[MAX_CBS];
static double g_cb_ms[MAX_CBS];
static int g_cb_count = 0;

inline void metal_trace_init(Context&) {}
inline void metal_trace_begin() { g_trace_enabled = true; g_cb_count = 0; }
inline void metal_trace_end() { g_trace_enabled = false; }
inline void metal_trace_sample() {}
inline void metal_trace_label(const char* label) { g_trace_name = label; }

static inline void metal_trace_record(id<MTLCommandBuffer> cb) {
    if (!g_trace_enabled || g_cb_count >= MAX_CBS) return;
    g_cb_labels[g_cb_count] = g_trace_name;
    g_cb_ms[g_cb_count] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    g_cb_count++;
}

inline void metal_trace_report() {
    if (g_cb_count == 0) { printf("[TRACE] No GPU timing data\n"); return; }
    printf("\n[GPU PER-CB TIMING]\n");
    double total_ms = 0;
    for (int i = 0; i < g_cb_count; i++) {
        total_ms += g_cb_ms[i];
        printf("  %-25s %8.1f us\n", g_cb_labels[i] ? g_cb_labels[i] : "(null)", g_cb_ms[i] * 1000.0);
    }
    printf("  %-25s %8.1f us  (%.2f ms)\n", "TOTAL", total_ms * 1000.0, total_ms);
    printf("  (%d command buffers traced)\n", g_cb_count);
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
        if (g_num_committed_cbs < MAX_CBS) {
            g_committed_cbs[g_num_committed_cbs] = g_batch_cb;
            g_committed_labels[g_num_committed_cbs] = g_trace_name;
            g_num_committed_cbs++;
        }
        [g_batch_cb commit];
        g_batch_cb = nil;
    }
}

// Commit current CB and start a new one (for granular profiling)
inline void metal_batch_checkpoint(void* ctx, const char* name) {
    Context* c = (Context*)ctx;
    if (g_batch_enc) { [g_batch_enc endEncoding]; g_batch_enc = nil; g_batch_pipe = nil; }
    if (g_batch_cb) {
        [g_batch_cb commit];
        [g_batch_cb waitUntilCompleted];
        if (g_trace_enabled && g_cb_count < MAX_CBS) {
            const char* label = name ? name : g_trace_name;
            g_cb_labels[g_cb_count] = label;
            g_cb_ms[g_cb_count] = (g_batch_cb.GPUEndTime - g_batch_cb.GPUStartTime) * 1000.0;
            g_cb_count++;
        }
        g_batch_cb = nil;
    }
}

// Wait for all committed CBs, record GPU timings, and clear the list
inline void metal_batch_wait_all() {
    // All CBs are serialized on the same queue — waiting on the last one
    // guarantees all are complete. Skip the loop of per-CB waits.
    if (g_num_committed_cbs > 0) {
        [g_committed_cbs[g_num_committed_cbs - 1] waitUntilCompleted];
    }
    if (g_trace_enabled) {
        for (int i = 0; i < g_num_committed_cbs && g_cb_count < MAX_CBS; i++) {
            g_cb_labels[g_cb_count] = g_committed_labels[i];
            g_cb_ms[g_cb_count] = (g_committed_cbs[i].GPUEndTime - g_committed_cbs[i].GPUStartTime) * 1000.0;
            g_cb_count++;
        }
    }
    for (int i = 0; i < g_num_committed_cbs; i++) {
        g_committed_cbs[i] = nil;
        g_committed_labels[i] = nullptr;
    }
    g_num_committed_cbs = 0;
}

inline void metal_batch_end() {
    if (g_batch_enc) { [g_batch_enc endEncoding]; g_batch_enc = nil; g_batch_pipe = nil; }
    if (g_batch_cb) {
        [g_batch_cb commit];
        [g_batch_cb waitUntilCompleted];
        if (g_trace_enabled && g_cb_count < MAX_CBS) {
            g_cb_labels[g_cb_count] = g_trace_name;
            g_cb_ms[g_cb_count] = (g_batch_cb.GPUEndTime - g_batch_cb.GPUStartTime) * 1000.0;
            g_cb_count++;
        }
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
// Dispatch for simdgroup kernels (different threadgrid semantics)
inline void metal_batch_dispatch_simd(
    Context& ctx, id<MTLComputePipelineState> pipe,
    int tg_x, int tg_y, int tg_z,
    int rows, int cols,
    const uint8_t* w_data, size_t w_bytes,
    const float* x_data, float* y_data)
{
    bool batch_was_active = (g_batch_cb != nil || g_batch_enc != nil);
    metal_batch_ensure_encoder(ctx, pipe);

    auto wit = g_w_cache.find(w_data);
    id<MTLBuffer> bufW;
    if (wit != g_w_cache.end()) {
        bufW = wit->second;
    } else {
        bufW = [ctx.device newBufferWithBytesNoCopy:(void*)w_data
                length:w_bytes options:MTLStorageModeShared deallocator:nil];
        g_w_cache[w_data] = bufW;
    }

    // For simdgroup, use the passed dimensions
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:(void*)x_data
                         length:(size_t)cols * 4 options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:(void*)y_data
                         length:(size_t)rows * 4 options:MTLStorageModeShared deallocator:nil];

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

    [g_batch_enc dispatchThreads:MTLSizeMake(tg_x, tg_y, tg_z)
             threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
}

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
        [g_batch_enc pushDebugGroup:@(g_trace_name)];
        [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [g_batch_enc popDebugGroup];
    } else {
        int tg = std::min(rows, 64);
        [g_batch_enc pushDebugGroup:@(g_trace_name)];
        [g_batch_enc dispatchThreads:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [g_batch_enc popDebugGroup];
    }
    // Standalone call: commit+wait here (batch-active calls rely on caller to end)
    if (!batch_was_active) metal_batch_end();
}

// Kernels are loaded from separate .metal files in the kernels/ directory.
// DO NOT add kernel source here — edit the corresponding file in kernels/.

// ── Capture stop ───────────────────────────────────────────────

inline void capture_stop() {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

// ── Dispatch helpers (batch-based, no per-call commit/wait) ──

inline void matmul_q8_0(Context& ctx, int rows, int cols,
                        const uint8_t* w, const float* x, float* y) {
    // Prefer simd kernel if available
    if (ctx.pipe_q8_0_simd) {
        int nb = cols / 32;
        int tg_x = (nb + 15) / 16;
        int tg_y = (rows + 7) / 8;
        size_t wb = ((size_t)rows * cols + 31) / 32 * 34;
        metal_batch_dispatch_simd(ctx, ctx.pipe_q8_0_simd, tg_x, tg_y, 1, rows, cols, w, wb, x, y);
    } else {
        size_t wb = ((size_t)rows * cols + 31) / 32 * 34;
        metal_batch_dispatch(ctx, ctx.pipe_q8_0, rows, cols, w, wb, x, y, true);
    }
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
    [g_batch_enc pushDebugGroup:@(g_trace_name)];
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_batch_enc popDebugGroup];
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
    [g_batch_enc pushDebugGroup:@(g_trace_name)];
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_batch_enc popDebugGroup];
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
    static bool use_flash_attn = false; // DISABLED: tmp[8] overflow
    if (use_flash_attn && ctx.pipe_flash_attn && head_dim == 64) {
        metal_batch_ensure_encoder(ctx, ctx.pipe_flash_attn);
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
        int n_tg = (n_head + 7) / 8;
        [g_batch_enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
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
    [g_batch_enc pushDebugGroup:@(g_trace_name)];
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_batch_enc popDebugGroup];
}

inline void fused_ffn_gate_up_op(Context& ctx,
                                  const uint8_t* W_gate, const uint8_t* W_up,
                                  const float* x, float* y,
                                  int rows_per, int cols,
                                  id<MTLComputePipelineState> pipe,
                                  size_t wb_bytes) {
    metal_batch_ensure_encoder(ctx, pipe);
    id<MTLBuffer> bufWg = wrap_buffer(ctx, W_gate, wb_bytes);
    id<MTLBuffer> bufWu = wrap_buffer(ctx, W_up, wb_bytes);
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
    [g_batch_enc pushDebugGroup:@(g_trace_name)];
    int total = ((rows_per * 2 + 3) / 4) * 64;
    [g_batch_enc dispatchThreads:MTLSizeMake(total, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [g_batch_enc popDebugGroup];
}

// Init/shutdown — implemented in metal_backend.cpp (loads kernels from .metal files)
bool init(Context& ctx);
void shutdown(Context& ctx);
bool capture_start(Context& ctx);
} // namespace metal_backend

// ── GPU trace capture ──────────────────────────────────────────
