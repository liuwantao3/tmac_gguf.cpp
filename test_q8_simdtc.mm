// Standalone Q8_0 simdgroup tensor-core kernel test
// Compile: clang++ -std=c++17 -x objective-c++ -O3 test_q8_simdtc.mm metal_backend.cpp \
//           -framework Foundation -framework Metal -fobjc-arc -o test_q8_simdtc

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
    int sign = f < 0 ? 1 : 0;
    f = fabsf(f);
    int exp;
    float mant_frac = frexpf(f, &exp);
    exp += 14;
    if (exp <= 0) return (sign << 15);
    if (exp >= 31) return (sign << 15) | (31 << 10);
    int mant = (int)((mant_frac - 0.5f) * 2048.0f + 0.5f);
    if (mant >= 1024) { mant = 0; exp++; }
    if (exp >= 31) return (sign << 15) | (31 << 10);
    return (sign << 15) | (exp << 10) | mant;
}

metal_backend::Context ctx;

int test(int rows, int cols) {
    printf("\n--- rows=%d cols=%d ---\n", rows, cols);
    int nb = cols / 32;
    size_t w_bytes = (size_t)rows * nb * 34;

    uint8_t* W = (uint8_t*)calloc(1, w_bytes);
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            size_t block_off = ((size_t)r * nb + b) * 34;
            *(uint16_t*)(W + block_off) = float_to_half(1.0f);
            for (int l = 0; l < 32; l++) {
                int val = (r * 100 + b * 10 + l) % 127 - 64;
                *(int8_t*)(W + block_off + 2 + l) = (int8_t)val;
            }
        }
    }

    float* x = (float*)calloc(cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = (float)(i % 7 - 3);

    // CPU reference
    float* y_ref = (float*)calloc(rows, sizeof(float));
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

    // GPU: simdgroup tensor core kernel
    // Load kernel from file
    NSString* nsPath = @"kernels/mul_mat_q8_0_simdtc.metal";
    NSError* err = nil;
    NSString* source = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&err];
    if (!source) { printf("FAIL: read kernel: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:nil error:&err];
    if (!lib) { printf("FAIL: compile kernel: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_mat_q8_0_simd"];
    if (!fn) { printf("FAIL: no function\n"); return 1; }
    id<MTLComputePipelineState> pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (err) { printf("FAIL: pipeline: %s\n", [[err description] UTF8String]); return 1; }

    float* y = (float*)calloc(rows, sizeof(float));

    id<MTLBuffer> bufW = [ctx.device newBufferWithBytesNoCopy:W
                           length:w_bytes options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:x
                           length:cols*4 options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:y
                           length:rows*4 options:MTLStorageModeShared deallocator:nil];

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:bufW offset:0 atIndex:0];
    [enc setBuffer:bufX offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    int params[3] = {rows, cols, 0};
    [enc setBytes:params length:12 atIndex:3];
    // Dispatch: 8 rows per threadgroup
    int tg_y = (rows + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(1, tg_y, 1)
     threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    float max_diff = 0;
    int bad = 0;
    for (int i = 0; i < rows; i++) {
        float diff = fabsf(y[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.001f && bad < 5) {
            printf("  row %d: gpu=%.6f cpu=%.6f diff=%.6f\n", i, y[i], y_ref[i], diff);
            bad++;
        }
    }
    printf("  max_diff = %.6f  %s\n", max_diff, max_diff < 0.001f ? "PASS" : "FAIL");

    free(W); free(x); free(y); free(y_ref);
    return (max_diff < 0.001f) ? 0 : 1;
}

int main() {
    if (!metal_backend::init(ctx)) {
        printf("FAIL: metal init\n");
        return 1;
    }

    int failures = 0;
    failures += test(8, 64);     // single threadgroup, single block per row
    failures += test(8, 896);   // single TG, many blocks per row
    failures += test(128, 64);  // many TGs, single block per row
    failures += test(128, 896); // V-projection dims
    failures += test(896, 896); // Q-projection dims

    metal_backend::shutdown(ctx);
    printf("\n=== %s ===\n", failures == 0 ? "ALL PASS" : "SOME FAILED");
    return failures;
}
