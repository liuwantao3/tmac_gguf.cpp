#include "metal_backend.hpp"
#include <chrono>
#include <cmath>

void run_matmul(metal_backend::Context& ctx,
                int rows, int cols,
                id<MTLBuffer> bufA,
                id<MTLBuffer> bufX,
                id<MTLBuffer> bufY,
                id<MTLBuffer> bufParams) {
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    [enc setBuffer:bufParams offset:0 atIndex:3];
    MTLSize grid = {(NSUInteger)rows, 1, 1};
    MTLSize tg   = {(NSUInteger)std::min(rows, 256), 1, 1};
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

int main() {
    metal_backend::Context ctx;
    if (!metal_backend::init(ctx)) return 1;

    const int rows = 896, cols = 896;
    const int n_bytes = rows * cols * sizeof(float);

    // PRODUCTION PATTERN: allocate MTLBuffer, work directly with pointers
    // ──────────────────────────────────────────────────────────────────
    id<MTLBuffer> bufW = [ctx.device newBufferWithLength:n_bytes
                                                  options:MTLStorageModeShared];
    id<MTLBuffer> bufX = [ctx.device newBufferWithLength:cols * sizeof(float)
                                                  options:MTLStorageModeShared];
    id<MTLBuffer> bufY = [ctx.device newBufferWithLength:rows * sizeof(float)
                                                  options:MTLStorageModeShared];
    int params[2] = {rows, cols};
    id<MTLBuffer> bufParams = [ctx.device newBufferWithBytes:params
                                                      length:sizeof(params)
                                                     options:MTLStorageModeShared];

    // Initialize data via CPU on the shared buffer directly
    float* A = (float*)[bufW contents];
    float* x = (float*)[bufX contents];
    for (int i = 0; i < rows * cols; i++) A[i] = (i % 100) * 0.01f;
    for (int i = 0; i < cols; i++) x[i] = i * 0.02f;

    // CPU baseline (using pointers, no vector)
    float y_ref[rows];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) sum += A[r * cols + c] * x[c];
        y_ref[r] = sum;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // GPU — writes directly to bufY contents
    float* y_gpu = (float*)[bufY contents];
    memset(y_gpu, 0, rows * sizeof(float));

    auto g0 = std::chrono::high_resolution_clock::now();
    run_matmul(ctx, rows, cols, bufW, bufX, bufY, bufParams);
    auto g1 = std::chrono::high_resolution_clock::now();

    // Verify — read bufY contents directly (no memcpy)
    float max_diff = 0;
    for (int i = 0; i < rows; i++)
        max_diff = fmaxf(max_diff, fabsf(y_ref[i] - ((float*)[bufY contents])[i]));

    printf("[896×896] CPU: %.3fms  GPU: %.3fms  Speedup: %.2fx  Diff: %.6f %s\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(),
           std::chrono::duration<double, std::milli>(g1 - g0).count(),
           std::chrono::duration<double, std::milli>(t1 - t0).count() /
               std::chrono::duration<double, std::milli>(g1 - g0).count(),
           max_diff, max_diff < 1e-3f ? "✓" : "✗");

    metal_backend::shutdown(ctx);
    return 0;
}