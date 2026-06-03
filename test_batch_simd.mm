// Test metal_batch_dispatch_simd within a batch with other ops
#include "metal_backend.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

static uint16_t float_to_half(float f) {
    int s=f<0?1:0;f=fabsf(f);int e;float m=frexpf(f,&e);e+=14;
    if(e<=0)return s<<15;if(e>=31)return(s<<15)|(31<<10);
    int mant=(int)((m-0.5f)*2048.0f+0.5f);
    if(mant>=1024){mant=0;e++;}if(e>=31)return(s<<15)|(31<<10);
    return(s<<15)|(e<<10)|mant;
}

metal_backend::Context ctx;

int main() {
    if (!metal_backend::init(ctx)) return 1;
    NSString* src = [NSString stringWithContentsOfFile:@"kernels/mul_mat_q8_0_simd.metal"
                       encoding:NSUTF8StringEncoding error:nil];
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:nil];
    ctx.pipe_q8_0_simd = [ctx.device newComputePipelineStateWithFunction:
                          [lib newFunctionWithName:@"mul_mat_q8_0_simd"] error:nil];

    int rows=128, cols=896, nb=cols/32;
    size_t wb = (size_t)rows*nb*34;
    uint8_t* W = (uint8_t*)calloc(1, wb);
    for (int r=0;r<rows;r++) for(int b=0;b<nb;b++) {
        size_t off=((size_t)r*nb+b)*34;
        *(uint16_t*)(W+off)=float_to_half(1.0f);
        for(int l=0;l<32;l++) *(int8_t*)(W+off+2+l)=(int8_t)((r*100+b*10+l)%127-64);
    }
    float* buf = (float*)calloc(cols+rows, sizeof(float));
    float* x = buf;
    float* y = buf + cols;
    for(int i=0;i<cols;i++) x[i]=(float)(i%7-3);
    float* y_ref=(float*)calloc(rows,sizeof(float));
    for(int r=0;r<rows;r++){float s=0;for(int c=0;c<cols;c++){int b=c/32,l=c%32;size_t o=((size_t)r*nb+b)*34;uint16_t h=*(uint16_t*)(W+o);int e=(h>>10)&0x1f,m=h&0x3ff;float d=e==0?ldexpf((float)m,-24):e==31?m?NAN:INFINITY:ldexpf((float)(m|0x400),e-25)*(h>>15?-1:1);float q=(float)*(int8_t*)(W+o+2+l);s+=q*d*x[c];}y_ref[r]=s;}
    
    // BATCH: elem_op + SIMD matmul + elem_op (like the fused path)
    float scratch[896];
    float output[128];
    
    metal_backend::metal_batch_begin(ctx);
    
    // elem_op: copy x to scratch
    metal_backend::elem_op(ctx, 2, scratch, x, cols);
    
    // SIMD matmul
    int tg_x = 1;
    int tg_y = (rows + 3) / 4;
    size_t wb2 = ((size_t)rows * cols + 31) / 32 * 34;
    metal_backend::metal_batch_dispatch_simd(ctx, ctx.pipe_q8_0_simd,
        tg_x, tg_y, 1, rows, cols, W, wb2, scratch, output);
    
    // elem_op: add (just for testing)
    metal_backend::elem_op(ctx, 0, output, y_ref, rows);
    
    metal_backend::metal_batch_end();
    
    // Now output should be SIMD_result + y_ref = 2 * y_ref
    float md=0;
    for(int r=0;r<rows;r++){float d=fabsf(output[r]-2*y_ref[r]);if(d>md)md=d;}
    printf("max_diff=%.6f (expected 0, got output+ref vs 2*ref)\n",md);
    printf("%s\n",md<0.001f?"PASS":"FAIL");
    
    free(W);free(buf);free(y_ref);
    metal_backend::shutdown(ctx);
    return 0;
}
