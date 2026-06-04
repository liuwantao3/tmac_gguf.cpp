#include <metal_stdlib>
using namespace metal;

// Q8_0 matvec using simdgroup_float8x8 (Apple GPU tensor core unit)
// Block: [half scale][int8 x 32] = 34 bytes per 32 values
// Layout: 8 rows per threadgroup, both simdgroups load W into threadgroup
//   sgitg=0 does all 8 tiles of 8x8 simdgroup matmul per iteration
// sa[8][64]: dequantized W, row-major
// sb[64]: broadcast buffer for x (8 copies of each x value, 8 rows x 8 cols)
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
    int debug_mode = params[2];
    int nb = cols / 32;
    int r0 = tgpig.y * 8;
    if (r0 >= rows) return;
    int nr = min(8, rows - r0);

    threadgroup float sa[8 * 64];
    threadgroup float sb[64];

    simdgroup_float8x8 acc;
    simdgroup_float8x8 ma, mb;
    acc = make_filled_simdgroup_matrix<float,8>(0.f);
    ma = make_filled_simdgroup_matrix<float,8>(0.f);
    mb = make_filled_simdgroup_matrix<float,8>(0.f);

    for (int ib = 0; ib < nb; ib += 2) {
        // Load A into threadgroup: both simdgroups load one Q8_0 block each
        for (int row = 0; row < 8; row++) {
            int row_offset = (r0 + row) * nb + ib + sgitg;
            sa[row*64 + sgitg*32 + tiisg] = 0.f;
            if (row_offset < rows * nb) {
                device const uchar* bp = W + (ulong)row_offset * 34;
                half d = *(device const half*)bp;
                uchar rq = *(bp + 2 + tiisg);
                float q = rq > 127 ? (float)(rq - 256) : (float)rq;
                sa[row*64 + sgitg*32 + tiisg] = (float)d * q;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // DEBUG: dump first 8 x values and first 8 dequantized W values
        if (debug_mode == 1 && ib == 0) {
            if (tiisg < 8) {
                y[tiisg] = x[tiisg];
                y[8 + tiisg] = sa[tiisg];
            }
            return;
        }

        // Matmul: only simdgroup 0 does the 8x8 tensor core ops
        if (sgitg == 0) {
            for (int tile = 0; tile < 8; tile++) {
                // Build broadcast: sb[row*8 + 0..7] = x_val[row] repeated 8x
                int row_b = tiisg / 4;
                int cp_b = tiisg % 4;
                int x_idx = ib * 32 + tile * 8 + row_b;
                float xv = x_idx < cols ? x[x_idx] : 0.f;
                sb[row_b * 8 + cp_b * 2 + 0] = xv;
                sb[row_b * 8 + cp_b * 2 + 1] = xv;

                // Load 8x8 tile from dequantized W (stride=64, row-major)
                simdgroup_load(ma, sa + tile * 8, 64, ulong2(0, 0), false);
                // Load 8x8 broadcast x (stride=8, row-major)
                simdgroup_load(mb, sb, 8, ulong2(0, 0), false);
                // acc += W * x  →  acc[r][c] = Σ_k W[r][k] * x[k]  (all cols same)
                simdgroup_multiply_accumulate(acc, ma, mb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulator: acc[r][c] = dot row r with x (same for all c)
    threadgroup float tmp[64];
    if (sgitg == 0) {
        simdgroup_store(acc, tmp, 8, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read column 0 of each row
    if (sgitg == 0 && tiisg < nr) {
        y[r0 + tiisg] = tmp[tiisg * 8];
    }
}
