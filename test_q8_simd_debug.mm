// Debug: test each dispatch parameter individually
#include "metal_backend.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static float half_to_float(uint16_t h) {
    int s = (h>>15)&1, e = (h>>10)&0x1f, m = h&0x3ff;
    if (e==0) return ldexpf((float)m, -24);
    if (e==31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return ldexpf((float)(m|0x400), e-25) * (s ? -1 : 1);
}

metal_backend::Context ctx;

void test(int rows, int cols) {
    printf("\n--- rows=%d cols=%d ---\n", rows, cols);
    int nb = cols/32;
    
    uint8_t* W = (uint8_t*)calloc(1, ((size_t)rows*nb*34));
    for (size_t i = 0; i < (size_t)rows*nb*34; i++) W[i] = (uint8_t)(i & 0xFF);
    
    float* x = (float*)calloc(cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = (float)(i % 7 - 3);
    float* y = (float*)calloc(rows, sizeof(float));
    
    // CPU reference
    float* y_ref = (float*)calloc(rows, sizeof(float));
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            int b = c/32, l = c%32;
            size_t off = ((size_t)r*nb + b)*34;
            float d = half_to_float(*(uint16_t*)(W+off));
            float q = (float)*(int8_t*)(W+off+2+l);
            sum += q*d*x[c];
        }
        y_ref[r] = sum;
    }
    
    // GPU SIMD dispatch
    int tg_y = (rows + 3) / 4;
    id<MTLBuffer> bW = [ctx.device newBufferWithBytesNoCopy:W length:(size_t)rows*nb*34
                         options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bX = [ctx.device newBufferWithBytesNoCopy:x length:cols*4
                         options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bY = [ctx.device newBufferWithBytesNoCopy:y length:rows*4
                         options:MTLStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipe_q8_0_simd];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bX offset:0 atIndex:1];
    [enc setBuffer:bY offset:0 atIndex:2];
    int params[3] = {rows, cols, 0};
    [enc setBytes:params length:12 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, tg_y, 1)
     threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    float max_diff = 0; int bad_rows = 0;
    for (int r = 0; r < rows; r++) {
        float d = fabsf(y[r] - y_ref[r]);
        if (d > max_diff) max_diff = d;
        if (d > 0.001f && bad_rows < 5) {
            printf("  row %d: gpu=%.2f cpu=%.2f diff=%.2f\n", r, y[r], y_ref[r], d);
            bad_rows++;
        }
    }
    printf("  max_diff=%.6f  rows_checked=%d  %s\n", max_diff, rows,
           max_diff < 0.001f ? "PASS" : "FAIL");
    
    free(W); free(x); free(y); free(y_ref);
}

int main() {
    if (!metal_backend::init(ctx)) { printf("FAIL: init\n"); return 1; }
    
    // Load SIMD kernel
    NSString* src = [NSString stringWithContentsOfFile:@"kernels/mul_mat_q8_0_simd.metal"
                       encoding:NSUTF8StringEncoding error:nil];
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:nil];
    ctx.pipe_q8_0_simd = [ctx.device newComputePipelineStateWithFunction:
                          [lib newFunctionWithName:@"mul_mat_q8_0_simd"] error:nil];
    if (!ctx.pipe_q8_0_simd) { printf("FAIL: SIMD pipe\n"); return 1; }
    
    printf("SIMD kernel loaded\n");
    
    test(8, 64);     // small test
    test(8, 896);    // 8 rows, full K
    test(128, 64);   // many rows, small K
    test(128, 896);  // V projection
    test(896, 896);  // Q projection
    
    metal_backend::shutdown(ctx);
    return 0;
}
