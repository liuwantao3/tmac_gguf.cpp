#include <metal_stdlib>
using namespace metal;

// RoPE: in-place rotation on position embeddings
kernel void kernel_rope(device float* data [[buffer(0)]],
                         constant int* params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    int n_heads = params[0];
    int head_dim = params[1];
    int pos = params[2];
    float base = as_type<float>(params[3]);
    int pairs = head_dim / 2;
    int head = gid / pairs;
    int pair = gid % pairs;
    if (head >= n_heads) return;
    int idx = head * head_dim + pair * 2;
    float theta = 1.0 / pow(base, (float)(pair * 2) / head_dim);
    float angle = (float)pos * theta;
    float c = cos(angle), s = sin(angle);
    float x0 = data[idx], x1 = data[idx + 1];
    data[idx] = x0 * c - x1 * s;
    data[idx + 1] = x0 * s + x1 * c;
}