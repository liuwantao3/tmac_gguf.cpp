#include <metal_stdlib>
using namespace metal;

// Fused FFN gate+up Q5_0: both W_gate and W_up are Q5_0, same dimensions
kernel void kernel_fused_ffn_gate_up_q5(
    device const uint8_t* W_gate [[buffer(0)]],
    device const uint8_t* W_up [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int rows_per = params[0]; int cols = params[1];
    int total_rows = rows_per * 2;
    int nb = cols / 32;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= total_rows) return;
    int nr = min(2, total_rows - first_row);
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            int global_row = first_row + r;
            device const uint8_t* W = global_row < rows_per ? W_gate : W_up;
            int local_row = global_row < rows_per ? global_row : global_row - rows_per;
            ulong base = ((ulong)local_row * nb + ib) * 22;
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