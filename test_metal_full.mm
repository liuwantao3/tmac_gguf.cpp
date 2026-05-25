#include "metal_backend.hpp"
#include <Metal/Metal.h>
#include <iostream>
#include <chrono>

// Config from tmac_gguf.cpp - Qwen2-0.5B
constexpr int HIDDEN_DIM = 896;
constexpr int INTER_DIM = 4864;
constexpr int VOCAB_SIZE = 151936;
constexpr int NUM_LAYERS = 24;

// Current: one thread per output dimension
void run_inference_simple() {
    metal_backend::Context ctx;
    if (!metal_backend::init(ctx)) return;

    printf("\n=== Qwen2-0.5B Forward Pass (Simplified) ===\n\n");

    // Example: attn_q weight = [HIDDEN_DIM, HIDDEN_DIM] = [896, 896]
    const int rows = HIDDEN_DIM;
    const int cols = HIDDEN_DIM;
    
    std::vector<float> W(rows * cols);
    std::vector<float> x(cols);
    std::vector<float> y(rows);
    
    for (int i = 0; i < rows * cols; i++) W[i] = (i % 100) * 0.01f;
    for (int i = 0; i < cols; i++) x[i] = i * 0.02f;

    // CPU baseline
    std::vector<float> y_cpu(rows);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) sum += W[r * cols + c] * x[c];
        y_cpu[r] = sum;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU
    memset(y.data(), 0, y.size() * sizeof(float));
    id<MTLBuffer> bufW = [ctx.device newBufferWithBytes:W.data() length:W.size()*sizeof(float) options:MTLStorageModeShared];
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:x.data() length:x.size()*sizeof(float) options:MTLStorageModeShared];
    id<MTLBuffer> bufY = [ctx.device newBufferWithBytes:y.data() length:y.size()*sizeof(float) options:MTLStorageModeShared];
    int params[2] = {rows, cols};
    id<MTLBuffer> bufParams = [ctx.device newBufferWithBytes:params length:sizeof(params) options:MTLStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipeline];
    [enc setBuffer:bufW offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    [enc setBuffer:bufParams offset:0 atIndex:3];

    auto g0 = std::chrono::high_resolution_clock::now();
    [enc dispatchThreads:{rows, 1, 1} threadsPerThreadgroup:{std::min((NSUInteger)rows, (NSUInteger)256), 1, 1}];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    auto g1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    printf("Layer: attn_q   [%4d × %4d]  CPU: %6.2f ms  GPU: %6.2f ms  Speedup: %.2fx\n",
           rows, cols, cpu_ms, gpu_ms, cpu_ms/gpu_ms);

    // Verify
    memcpy(y.data(), [bufY contents], y.size() * sizeof(float));
    float max_diff = 0;
    for (int i = 0; i < 10; i++) max_diff = fmaxf(max_diff, fabsf(y_cpu[i] - y[i]));
    printf("Verifying first 10 outputs: max diff = %.6f %s\n\n", max_diff, max_diff < 1e-3f ? "✓" : "✗");

    // FFN: intermediate dimension [INTER_DIM, HIDDEN_DIM] = [4864, 896]
    const int ffn_rows = INTER_DIM;
    const int ffn_cols = HIDDEN_DIM;
    W.resize(ffn_rows * ffn_cols);
    x.resize(ffn_cols);
    y_cpu.resize(ffn_rows);
    y.resize(ffn_rows);
    
    for (int i = 0; i < ffn_rows * ffn_cols; i++) W[i] = (i % 100) * 0.01f;
    for (int i = 0; i < ffn_cols; i++) x[i] = i * 0.02f;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < ffn_rows; r++) {
        float sum = 0;
        for (int c = 0; c < ffn_cols; c++) sum += W[r * ffn_cols + c] * x[c];
        y_cpu[r] = sum;
    }
    t1 = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    memset(y.data(), 0, y.size() * sizeof(float));
    bufW = [ctx.device newBufferWithBytes:W.data() length:W.size()*sizeof(float) options:MTLStorageModeShared];
    bufX = [ctx.device newBufferWithBytes:x.data() length:x.size()*sizeof(float) options:MTLStorageModeShared];
    bufY = [ctx.device newBufferWithBytes:y.data() length:y.size()*sizeof(float) options:MTLStorageModeShared];
    int ffn_params[2] = {ffn_rows, ffn_cols};
    bufParams = [ctx.device newBufferWithBytes:ffn_params length:sizeof(ffn_params) options:MTLStorageModeShared];

    cmdBuf = [ctx.queue commandBuffer];
    enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipeline];
    [enc setBuffer:bufW offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    [enc setBuffer:bufParams offset:0 atIndex:3];

    g0 = std::chrono::high_resolution_clock::now();
    [enc dispatchThreads:{ffn_rows, 1, 1} threadsPerThreadgroup:{256, 1, 1}];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    g1 = std::chrono::high_resolution_clock::now();
    gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    printf("Layer: ffn_up    [%4d × %4d]  CPU: %6.2f ms  GPU: %6.2f ms  Speedup: %.2fx\n",
           ffn_rows, ffn_cols, cpu_ms, gpu_ms, cpu_ms/gpu_ms);

    metal_backend::shutdown(ctx);
}

int main() {
    printf("=== Metal Backend for Qwen2-0.5B ===\n");
    printf("Hidden dim: %d, FFN dim: %d, Vocab: %d, Layers: %d\n\n",
           HIDDEN_DIM, INTER_DIM, VOCAB_SIZE, NUM_LAYERS);
    
    run_inference_simple();

    printf("Summary:\n");
    printf("- Each layer needs multiple matmul operations (Q, K, V, output proj, FFN)\n");
    printf("- Q,K,V: [896×896], Output proj: [896×896]\n");
    printf("- FFN: gate=[4864×896], up=[4864×896], down=[896×4864]\n");
    printf("- 24 layers × ~10 matmuls = ~240 matmuls per token\n");
    
    return 0;
}