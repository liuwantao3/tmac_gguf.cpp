// Standalone Q8_0 SIMD kernel test
// Compile: clang++ -std=c++17 -x objective-c++ -O3 test_q8_simd.mm metal_backend.cpp \
//           -framework Foundation -framework Metal -fobjc-arc -o test_q8_simd

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
    if (exp == 0) {
        return ldexp((float)mant, -24);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
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

int main() {
    if (!metal_backend::init(ctx)) {
        printf("FAIL: metal init\n");
        return 1;
    }

    // Load the SIMD kernel from file
    NSString* nsPath = @"kernels/mul_mat_q8_0_simd.metal";
    NSError* err = nil;
    NSString* source = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&err];
    if (!source) { printf("FAIL: read kernel: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:nil error:&err];
    if (!lib) { printf("FAIL: compile kernel: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_mat_q8_0_simd"];
    if (!fn) { printf("FAIL: no function\n"); return 1; }
    id<MTLComputePipelineState> pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (err) { printf("FAIL: pipeline: %s\n", [[err description] UTF8String]); return 1; }
    printf("[OK] SIMD pipeline created\n");
    printf("[OK] Thread execution width: %zu\n", pipe.threadExecutionWidth);
    NSUInteger tg_size = [pipe maxTotalThreadsPerThreadgroup];
    printf("[OK] Max threads/threadgroup: %zu\n", tg_size);

    constexpr int ROWS = 8;
    constexpr int COLS = 64;
    int nb = COLS / 32;
    size_t w_bytes = (size_t)ROWS * nb * 34;
    uint8_t* W = (uint8_t*)calloc(1, w_bytes);
    
    // Fill with known values: row r, block b, lane l
    for (int r = 0; r < ROWS; r++) {
        for (int b = 0; b < nb; b++) {
            size_t block_off = ((size_t)r * nb + b) * 34;
            *(uint16_t*)(W + block_off) = float_to_half(1.0f);
            for (int l = 0; l < 32; l++) {
                int val = (r * 100 + b * 10 + l) % 127 - 64;
                *(int8_t*)(W + block_off + 2 + l) = (int8_t)val;
            }
        }
    }

    float* x = (float*)calloc(COLS, sizeof(float));
    for (int i = 0; i < COLS; i++) x[i] = (float)(i % 7 - 3);

    float* y = (float*)calloc(ROWS, sizeof(float));
    float* y_dbg = (float*)calloc(16, sizeof(float)); // debug: 8 sb + 8 sa

    printf("\n=== CPU reference ===\n");
    for (int r = 0; r < ROWS; r++) {
        float sum = 0;
        for (int c = 0; c < COLS; c++) {
            int b = c / 32, l = c % 32;
            size_t off = ((size_t)r * nb + b) * 34;
            float d = half_to_float(*(uint16_t*)(W + off));
            float q = (float)*(int8_t*)(W + off + 2 + l);
            sum += q * d * x[c];
        }
        printf("  cpu y[%d] = %.6f\n", r, sum);
    }

    printf("\n=== DEBUG: dump sa/sb for ib=0 ===\n");
    {
        id<MTLBuffer> bufW = [ctx.device newBufferWithBytesNoCopy:W
                               length:w_bytes options:MTLStorageModeShared deallocator:nil];
        id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:x
                               length:COLS*4 options:MTLStorageModeShared deallocator:nil];
        id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:y_dbg
                               length:16*4 options:MTLStorageModeShared deallocator:nil];

        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bufW offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        int params[3] = {ROWS, COLS, 1}; // debug_mode=1
        [enc setBytes:params length:12 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, (ROWS+3)/4, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        printf("  sb[0..7] (x from ib=0, lanes 0..7):");
        for (int i = 0; i < 8; i++) printf(" %.4f", y_dbg[i]);
        printf("\n  expected x[0..7]:");
        for (int i = 0; i < 8; i++) printf(" %.4f", x[i]);
        printf("\n");

        printf("  sa[0..7] (W row=0, block=0, lanes 0..7):");
        for (int i = 0; i < 8; i++) printf(" %.4f", y_dbg[8+i]);
        printf("\n");
        printf("  expected (W[0][0]*1.0 for lanes 0..7):");
        for (int l = 0; l < 8; l++) {
            size_t off = (size_t)0 * nb * 34 + 0 * 34;
            float q = (float)*(int8_t*)(W + off + 2 + l);
            printf(" %.4f", q * 1.0f);
        }
        printf("\n");
    }

    printf("\n=== REAL COMPUTATION ===\n");
    {
        id<MTLBuffer> bufW = [ctx.device newBufferWithBytesNoCopy:W
                               length:w_bytes options:MTLStorageModeShared deallocator:nil];
        id<MTLBuffer> bufX = [ctx.device newBufferWithBytesNoCopy:x
                               length:COLS*4 options:MTLStorageModeShared deallocator:nil];
        id<MTLBuffer> bufY = [ctx.device newBufferWithBytesNoCopy:y
                               length:ROWS*4 options:MTLStorageModeShared deallocator:nil];

        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:bufW offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        int params[3] = {ROWS, COLS, 0}; // debug_mode=0
        [enc setBytes:params length:12 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, (ROWS+3)/4, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        printf("  GPU y:");
        for (int i = 0; i < ROWS; i++) printf(" %.6f", y[i]);
        printf("\n");

        float max_diff = 0;
        for (int i = 0; i < ROWS; i++) {
            float expected = 0;
            for (int c = 0; c < COLS; c++) {
                int b = c / 32, l = c % 32;
                size_t off = ((size_t)i * nb + b) * 34;
                float d = half_to_float(*(uint16_t*)(W + off));
                float q = (float)*(int8_t*)(W + off + 2 + l);
                expected += q * d * x[c];
            }
            float diff = fabsf(y[i] - expected);
            printf("  row %d: gpu=%.6f cpu=%.6f diff=%.6f%s\n", i, y[i], expected, diff,
                   diff < 0.001f ? " OK" : " MISMATCH");
            if (diff > max_diff) max_diff = diff;
        }
        printf("  max_diff = %.6f  %s\n", max_diff, max_diff < 0.001f ? "PASS" : "FAIL");
    }

    free(W);
    free(x);
    free(y);
    free(y_dbg);
    metal_backend::shutdown(ctx);
    return 0;
}
