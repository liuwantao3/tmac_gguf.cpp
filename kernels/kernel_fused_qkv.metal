#include <metal_stdlib>
using namespace metal;

// Fused QKV: Q5_0 for Q+K, Q5_0 or Q8_0 for V (function constant v_type)
// v_type=0: V is Q5_0, v_type=1: V is Q8_0
constant int v_type [[function_constant(1)]];

kernel void kernel_fused_qkv(
    device const uint8_t* W_q [[buffer(0)]],
    device const uint8_t* W_k [[buffer(1)]],
    device const uint8_t* W_v [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y_q [[buffer(4)]],
    device float* y_k [[buffer(5)]],
    device float* y_v [[buffer(6)]],
    constant int* params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    int q_rows = params[0], k_rows = params[1], v_rows = params[2];
    int cols = params[3];
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int total_rows = q_rows + k_rows + v_rows;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= total_rows) return;
    int nr = min(2, total_rows - first_row);
    int nb = cols / 32;
    float sumf[2] = {0.f, 0.f};
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            int global_row = first_row + r;
            device const uint8_t* W;
            int local_row;
            if (global_row < q_rows) {
                W = W_q; local_row = global_row;
            } else if (global_row < q_rows + k_rows) {
                W = W_k; local_row = global_row - q_rows;
            } else {
                W = W_v; local_row = global_row - q_rows - k_rows;
            }
            if (global_row < q_rows + k_rows || v_type == 0) {
                ulong base = ((ulong)local_row * nb + ib) * 22;
                float d = (float)*(device const half*)(W + base);
                int lane_half = lane < 16 ? lane : lane - 16;
                uint qh = (uint)W[base+2]|((uint)W[base+3]<<8)|((uint)W[base+4]<<16)|((uint)W[base+5]<<24);
                uint qs_byte = W[base + 6 + lane_half];
                uint ql = lane < 16 ? (qs_byte & 0xF) : (qs_byte >> 4);
                int q = (int)((((qh >> lane) & 1) << 4) | ql) - 16;
                sumf[r] += (float)q * d * xv;
            } else {
                ulong base = ((ulong)local_row * nb + ib) * 34;
                float d = (float)*(device const half*)(W + base);
                float q = (float)*(device const int8_t*)(W + base + 2 + lane);
                sumf[r] += q * d * xv;
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) {
            int global_row = first_row + r;
            if (global_row < q_rows) y_q[global_row] = total;
            else if (global_row < q_rows + k_rows) y_k[global_row - q_rows] = total;
            else y_v[global_row - q_rows - k_rows] = total;
        }
    }
}