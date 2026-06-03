#include <metal_stdlib>
using namespace metal;

// Simple FP32 matvec: one thread per output row
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