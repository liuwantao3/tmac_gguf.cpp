#include <metal_stdlib>
using namespace metal;

// Q8_0 simdgroup matvec: hardware-accelerated 8x8 matmul unit
// Block: [half scale][int8 x 32] = 34 bytes per 32 values
kernel void mul_mat_q8_0_simd(
    device const uint8_t* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int* params [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    int rows = params[0], cols = params[1];
    int nb = cols / 32;
    int r0 = tgpig.y * 8;
    if (r0 >= rows) return;
    int nr = (rows - r0) > 8 ? 8 : (rows - r0);
    int group_k = tgpig.x * 16;

    threadgroup float sa[16 * 64];
    threadgroup float sb[16 * 32];

    simdgroup_float8x8 acc;
    simdgroup_float8x8 ma, mb;
    ma = mb = make_filled_simdgroup_matrix<float, 8>(0.f);
    acc = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (int ib = group_k; ib < nb && ib < group_k + 16; ib++) {
        // Load A into threadgroup
        for (int i = 0; i < 16; i++) {
            int row_offset = (r0 + i/2) * nb + ib;
            if (row_offset < rows * nb) {
                device const uchar* bp = W + (ulong)row_offset * 34;
                half d = *(device const half*)bp;
                for (int j = 0; j < 8; j++) {
                    sa[i*64 + (i%2)*4 + j] = (float)d * (float)*(bp + 2 + j);
                }
            }
        }
        // Load B into threadgroup
        for (int i = 0; i < 8; i++) {
            sb[i*32] = (ib * 32 + i < cols) ? x[ib * 32 + i] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // matmul: load from threadgroup memory
        threadgroup const float* lda = sa;
        threadgroup const float* ldb = sb;
        simdgroup_load(ma, lda, 8, 0, false);
        simdgroup_load(mb, ldb, 8, 0, false);
        simdgroup_multiply_accumulate(ma, mb, ma, acc);
    }

    // Store accumulator
    threadgroup float* tmp = sa + (sgitg & 1) * 32;
    simdgroup_store(acc, tmp, 8, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        for (int i = 0; i < nr; i++) y[r0 + i] = tmp[i];
    }
}