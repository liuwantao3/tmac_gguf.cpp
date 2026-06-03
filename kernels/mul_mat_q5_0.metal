#include <metal_stdlib>
using namespace metal;

// Q5_0 matvec: branchless nibble extraction, simd_sum reduction
// Block: [half scale][uint4 x 32] + qh bits = 22 bytes per 32 values
kernel void mul_mat_q5_0(
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
            ulong base = ((ulong)(first_row + r) * nb + ib) * 22;
            float d = (float)*(device const half*)(W + base);
            uint qh = *(device const uint32_t*)(W + base + 2);
            uint qs_byte = W[base + 6 + (lane & 15)];
            uint ql = (qs_byte >> ((lane >> 4) * 4)) & 0xF;
            int q = (int)((((qh >> lane) & 1) << 4) | ql) - 16;
            sumf[r] += (float)q * d * xv;
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}