#include "metal_backend.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static uint16_t float_to_half(float f) {
    int sign = f<0?1:0; f=fabsf(f);
    int exp; float m=frexpf(f,&exp); exp+=14;
    if (exp<=0) return sign<<15;
    if (exp>=31) return (sign<<15)|(31<<10);
    int mant=(int)((m-0.5f)*2048.0f+0.5f);
    if (mant>=1024){mant=0;exp++;}
    if (exp>=31) return (sign<<15)|(31<<10);
    return (sign<<15)|(exp<<10)|mant;
}

metal_backend::Context ctx;

int main(int argc, char** argv) {
    int use_simd = 1;
    if (argc > 1 && strcmp(argv[1], "scalar") == 0) use_simd = 0;
    
    if (!metal_backend::init(ctx)) return 1;

    id<MTLComputePipelineState> pipe;
    if (use_simd) {
        NSString* src = [NSString stringWithContentsOfFile:@"kernels/mul_mat_q8_0_simd.metal"
                           encoding:NSUTF8StringEncoding error:nil];
        id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:nil];
        pipe = [ctx.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mul_mat_q8_0_simd"] error:nil];
        printf("SIMD kernel\n");
    } else {
        pipe = ctx.pipe_q8_0;
        printf("Scalar kernel\n");
    }

    int rows=128, cols=896, nb=cols/32;
    size_t wb = (size_t)rows * nb * 34;
    uint8_t* W = (uint8_t*)calloc(1, wb);
    
    // Fill with clean weights: row r, block b, lane l = (r*100 + b*10 + l)
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            size_t off = ((size_t)r * nb + b) * 34;
            *(uint16_t*)(W+off) = float_to_half(1.0f);
            for (int l = 0; l < 32; l++) {
                int val = (r * 100 + b * 10 + l) % 127 - 64;
                *(int8_t*)(W+off+2+l) = (int8_t)val;
            }
        }
    }
    
    float* x = (float*)calloc(cols, sizeof(float));
    for (int i = 0; i < cols; i++) x[i] = (float)(i % 7 - 3);
    float* y = (float*)calloc(rows, sizeof(float));
    
    // CPU ref
    float* y_ref = (float*)calloc(rows, sizeof(float));
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            int b=c/32, l=c%32;
            size_t off = ((size_t)r * nb + b) * 34;
            float d = (float)*(int16_t*)(W+off);
            // convert from half manually
            uint16_t h = *(uint16_t*)(W+off);
            int e=(h>>10)&0x1f, m=h&0x3ff;
            if (e==0) d=ldexpf((float)m,-24);
            else if (e==31) d=m?NAN:INFINITY;
            else d=ldexpf((float)(m|0x400),e-25)*(h>>15?-1:1);
            float q = (float)*(int8_t*)(W+off+2+l);
            sum += q * d * x[c];
        }
        y_ref[r] = sum;
    }
    
    id<MTLBuffer> bW = [ctx.device newBufferWithBytesNoCopy:W length:wb
                         options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bX = [ctx.device newBufferWithBytesNoCopy:x length:cols*4
                         options:MTLStorageModeShared deallocator:nil];
    id<MTLBuffer> bY = [ctx.device newBufferWithBytesNoCopy:y length:rows*4
                         options:MTLStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bX offset:0 atIndex:1];
    [enc setBuffer:bY offset:0 atIndex:2];
    int params[3] = {rows, cols, 0};
    [enc setBytes:params length:12 atIndex:3];
    
    if (use_simd) {
        [enc dispatchThreadgroups:MTLSizeMake(1, (rows+3)/4, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(((rows+3)/4)*64, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    float max_diff = 0; int bad = 0;
    for (int r = 0; r < rows; r++) {
        float d = fabsf(y[r] - y_ref[r]);
        if (d > max_diff) max_diff = d;
        if (d > 0.001f && bad < 5) {
            printf("  row %d: gpu=%.6f cpu=%.6f diff=%.6f\n", r, y[r], y_ref[r], d);
            bad++;
        }
    }
    printf("  max_diff=%.6f %s\n", max_diff, max_diff<0.001f ? "PASS" : "FAIL");
    
    free(W); free(x); free(y); free(y_ref);
    metal_backend::shutdown(ctx);
    return 0;
}
