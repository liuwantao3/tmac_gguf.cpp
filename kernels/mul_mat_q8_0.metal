#include <metal_stdlib>
using namespace metal;

// Q8_0 naive matvec: per-lane simd_sum reduction
// Block: [half scale][int8 x 32] = 34 bytes per 32 values
kernel void mul_mat_q8_0(
    device const uint8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = cols / 32;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 34;
            float d = (float)*(device const half*)(W + base);
            float q = (float)*(device const int8_t*)(W + base + 2 + lane);
            sumf[r] += q * d * xv;
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}