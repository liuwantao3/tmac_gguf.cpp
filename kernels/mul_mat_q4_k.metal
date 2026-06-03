#include <metal_stdlib>
using namespace metal;

// Q4_K matvec: 256-element block, per-lane simd_sum reduction
kernel void mul_mat_q4_k(
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
    uint sub = pos / 32;
    uint nib = pos % 32;
    uint byte_base = (sub / 2) * 32 + nib;
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
            ulong base = ((ulong)(first_row + r) * nb + ib) * 144;
            float d    = (float)*(device const half*)(W + base);
            float dmin = (float)*(device const half*)(W + base + 2);
            device const uint8_t* scales = W + base + 4;
            device const uint8_t* qs = W + base + 16;
            uint a = scales[sub];
            uint b = scales[sub + 4];
            uint c = scales[(sub - 4) & 7];
            uint sc = sub < 4 ? (a & 63) : ((b & 0xF) | ((c >> 6) << 4));
            uint m  = sub < 4 ? (b & 63) : ((b >> 4) | ((a >> 6) << 4));
            float d_sc = d * (float)sc;
            float dmin_m = dmin * (float)m;
            device const uint8_t* qb = qs + byte_base;
            for (int k = 0; k < k_lim; k++) {
                uint8_t byte = qb[k];
                uint q4 = (byte >> ((sub & 1) * 4)) & 0xF;
                sumf[r] += (d_sc * (float)q4 - dmin_m) * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}