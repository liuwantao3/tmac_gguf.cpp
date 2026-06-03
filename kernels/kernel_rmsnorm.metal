#include <metal_stdlib>
using namespace metal;

// RMSNorm: in-place, 32-thread single-SG dispatch
kernel void kernel_rmsnorm(device float* data [[buffer(0)]],
                             device const float* weight [[buffer(1)]],
                             constant int* params [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    float my_sum = 0;
    for (int i = gid; i < dim; i += 32)
        my_sum += data[i] * data[i];
    float total = simd_sum(my_sum);
    float rms = sqrt(total / (float)dim + 1e-6f);
    for (int i = gid; i < dim; i += 32)
        data[i] = (data[i] / rms) * weight[i];
}