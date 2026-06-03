#include <metal_stdlib>
using namespace metal;

// Element-wise ops: specialized via function constant kernel_op
// kernel_op: 0=add, 1=silu_x_up, 2=cache_write
constant int kernel_op [[function_constant(0)]];

kernel void kernel_elem(device float* data [[buffer(0)]],
                         device const float* aux [[buffer(1)]],
                         constant int* params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    if (gid >= (uint)dim) return;
    if (kernel_op == 0) {
        data[gid] += aux[gid];
    } else if (kernel_op == 1) {
        float x = data[gid];
        data[gid] = (x / (1 + exp(-x))) * data[gid + dim];
    } else if (kernel_op == 2) {
        data[gid] = aux[gid];
    }
}