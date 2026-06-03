#include <metal_stdlib>
using namespace metal;

// Q8_0 matvec: uses simd_sum reduction, 4 rows per threadgroup.
// Thread layout: 32 threads per simdgroup × 2 simdgroups = 64 threads per group.
kernel void mul_mat_q8_0_simd(
    device const uint8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int* params [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    int rows = params[0], cols = params[1];
    int nb = cols / 32;
    // Each threadgroup covers 4 rows (2 per simdgroup)
    int tg_base = tgpig.y * 4;
    int first_row = tg_base + sgitg * 2;
    if (first_row >= rows) return;
    int nr = min(2, rows - first_row);
    float sumf[2]; sumf[0] = 0.0f; sumf[1] = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + tiisg];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 34;
            float d = (float)*(device const half*)(W + base);
            float q = (float)*(device const int8_t*)(W + base + 2 + tiisg);
            sumf[r] += q * d * xv;
        }
    }
    float total = simd_sum(sumf[0]);
    if (tiisg == 0) y[first_row + 0] = total;
    if (nr >= 2) {
        total = simd_sum(sumf[1]);
        if (tiisg == 0) y[first_row + 1] = total;
    }
}