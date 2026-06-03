#include <metal_stdlib>
using namespace metal;

// Q6_K matvec: 256-element block, per-lane simd_sum reduction
kernel void mul_mat_q6_k(
    device const uint8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    int nb = (cols + 255) / 256;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    uint pos = lane * 8;
    uint hf = pos / 128;
    uint pos_in_half = pos % 128;
    uint sub = pos_in_half / 32;
    uint l_beg = pos_in_half % 32;
    uint is = l_beg / 16;
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        ulong x_ofs = (ulong)ib * 256 + pos;
        float xv[8];
        int k_lim = 8;
        if (ib == nb - 1) {
            int rem = cols - ib * 256 - (int)pos;
            k_lim = rem < 8 ? max(0, rem) : 8;
        }
        for (int k = 0; k < k_lim; k++) xv[k] = x[x_ofs + k];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 210;
            float d = (float)*(device const half*)(W + base + 208);
            device const uint8_t* ql = W + base + hf * 64 + l_beg + (sub & 1) * 32;
            device const uint8_t* qh = W + base + 128 + hf * 32 + l_beg;
            int scale = (int)*(device const int8_t*)(W + base + 192 + hf * 8 + is + sub * 2);
            for (int k = 0; k < k_lim; k++) {
                uint ql_byte = ql[k];
                uint ql_nib = (ql_byte >> (((sub >> 1) & 1) * 4)) & 0xF;
                uint qh_bits = (qh[k] >> (sub * 2)) & 0x3;
                int q6 = (int)((qh_bits << 4) | ql_nib) - 32;
                sumf[r] += (float)scale * (float)q6 * d * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}