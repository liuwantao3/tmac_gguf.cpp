// Test Q8_0 matmul with V projection dimensions (128x896)
#include "metal_backend.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static float half_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1f;
    int mant = h & 0x3ff;
    if (exp == 0) return ldexp((float)mant, -24);
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    return ldexp((float)(mant | 0x400), exp - 25) * (sign ? -1 : 1);
}
static uint16_t float_to_half(float f) {
    int sign = f < 0 ? 1 : 0; f = fabsf(f);
    int exp; float mant_frac = frexpf(f, &exp); exp += 14;
    if (exp <= 0) return (sign << 15);
    if (exp >= 31) return (sign << 15) | (31 << 10);
    int mant = (int)((mant_frac - 0.5f) * 2048.0f + 0.5f);
    if (mant >= 1024) { mant = 0; exp++; }
    if (exp >= 31) return (sign << 15) | (31 << 10);
    return (sign << 15) | (exp << 10) | mant;
}

metal_backend::Context ctx;

int main(int argc, char** argv) {
    int use_simd = (argc > 1 && strcmp(argv[1], "scalar") == 0) ? 0 : 1;
    
    if (!metal_backend::init(ctx)) { printf("FAIL: init\n"); return 1; }
    
    id<MTLComputePipelineState> pipe;
    if (use_simd) {
        NSString* src = [NSString stringWithContentsOfFile:@"kernels/mul_mat_q8_0_simd.metal"
                           encoding:NSUTF8StringEncoding error:nil];
        id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:nil];
        pipe = [ctx.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mul_mat_q8_0_simd"] error:nil];
        if (!pipe) { printf("FAIL: SIMD pipe\n"); return 1; }
        printf("=== SIMD kernel ===\n");
    } else {
        pipe = ctx.pipe_q8_0;
        printf("=== Scalar kernel (pipe_q8_0) ===\n");
    }
    
    int rows = 128, cols = 896, nb = cols/32;
    size_t wb = ((size_t)rows * cols + 31) / 32 * 34;
    uint8_t* W = (uint8_t*)calloc(1, wb);
    for (size_t i = 0; i < wb; i++) W[i] = (uint8_t)(i & 0xFF);
    
    float* x = (float*)calloc(cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = (float)(i % 7 - 3);
    
    float* y = (float*)calloc(rows, sizeof(float));
    float* y_ref = (float*)calloc(rows, sizeof(float));
    
    // CPU reference
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            int b = c / 32, l = c % 32;
            size_t off = ((size_t)r * nb + b) * 34;
            float d = half_to_float(*(uint16_t*)(W + off));
            float q = (float)*(int8_t*)(W + off + 2 + l);
            sum += q * d * x[c];
        }
        y_ref[r] = sum;
    }
    
    // Dispatch: same as what matmul_q8_0 passes
    int tg_y = (rows + 3) / 4; // 4 rows per group for SIMD; scalar uses 1D
    int tg_x = 1; // all K-blocks in one group
    
    id<MTLBuffer> bufW = [ctx.device newBufferWithBytesNoCopy:W length:wb
                           options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:x length:cols*4
                           options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:y length:rows*4
                           options:MTLStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:bufW offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    int params[3] = {rows, cols, 0};
    [enc setBytes:params length:12 atIndex:3];
    
    if (use_simd) {
        [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(((rows+3)/4)*64, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    float max_diff = 0;
    for (int r = 0; r < rows; r++) {
        float diff = fabsf(y[r] - y_ref[r]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  rows=%d cols=%d max_diff=%.6f\n", rows, cols, max_diff);
    if (max_diff > 0.001f) {
        printf("  first 8: gpu=");
        for (int i = 0; i < 8; i++) printf(" %.2f", y[i]);
        printf(" cpu=");
        for (int i = 0; i < 8; i++) printf(" %.2f", y_ref[i]);
        printf("\n");
    }
    printf("  %s\n", max_diff < 0.001f ? "PASS" : "FAIL");
    
    free(W); free(x); free(y); free(y_ref);
    metal_backend::shutdown(ctx);
    return 0;
}
