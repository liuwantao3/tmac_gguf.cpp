#include <metal_stdlib>
using namespace metal;

// ── FP32 ──────────────────────────────────────────────────
kernel void mul_mat_fp32(device const float* A [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant int* params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    int rows = params[0], cols = params[1];
    if (gid >= (uint)rows) return;
    float sum = 0;
    for (int j = 0; j < cols; j++)
        sum += A[(uint)gid * cols + j] * x[j];
    y[gid] = sum;
}

// ── Q8_0 (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Block: [half scale][int8×32] = 34 bytes per 32 values
kernel void mul_mat_q8_0(device const uint8_t* W [[buffer(0)]],
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

// ── Q5_0 (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 1 element per 32-element block
kernel void mul_mat_q5_0(device const uint8_t* W [[buffer(0)]],
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
    int lane_half = lane < 16 ? lane : lane - 16;
    for (int ib = 0; ib < nb; ib++) {
        float xv = x[(uint)ib * 32 + lane];
        for (int r = 0; r < nr; r++) {
            ulong base = ((ulong)(first_row + r) * nb + ib) * 22;
            float d = (float)*(device const half*)(W + base);
            uint qh = (uint)W[base + 2] | ((uint)W[base + 3] << 8)
                     | ((uint)W[base + 4] << 16) | ((uint)W[base + 5] << 24);
            uint qs_byte = W[base + 6 + lane_half];
            uint ql = lane < 16 ? (qs_byte & 0xF) : (qs_byte >> 4);
            int q = (int)((((qh >> lane) & 1) << 4) | ql) - 16;
            sumf[r] += (float)q * d * xv;
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}

// ── Q4_K (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 8 elements per 256-element block
kernel void mul_mat_q4_k(device const uint8_t* W [[buffer(0)]],
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
            uint sc, m;
            if (sub < 4) {
                sc = scales[sub] & 63;  m = scales[sub + 4] & 63;
            } else {
                sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
                m  = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
            }
            float d_sc = d * (float)sc;
            float dmin_m = dmin * (float)m;
            device const uint8_t* qb = qs + byte_base;
            for (int k = 0; k < k_lim; k++) {
                uint8_t byte = qb[k];
                uint q4 = (sub & 1) ? (byte >> 4) : (byte & 0xF);
                sumf[r] += (d_sc * (float)q4 - dmin_m) * xv[k];
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        float total = simd_sum(sumf[r]);
        if (lane == 0) y[first_row + r] = total;
    }
}

// ── Q6_K (SIMD, 64-thread threadgroup, 2 SIMD groups × 2 rows) ──
// Each lane handles 8 elements per 256-element block
kernel void mul_mat_q6_k(device const uint8_t* W [[buffer(0)]],
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
                uint ql_nib = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
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
// ── Fused QKV: Q5_0 for Q+K, Q5_0 or Q8_0 for V (function-constant v_type) ──
// v_type=0: V also Q5_0, v_type=1: V is Q8_0
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
    int lane_half = lane < 16 ? lane : lane - 16;
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
// ── Fused FFN gate+up (both Q6_K, same dims) ──
kernel void kernel_fused_ffn_gate_up(
    device const uint8_t* W_gate [[buffer(0)]],
    device const uint8_t* W_up [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int rows_per = params[0]; int cols = params[1];
    int total_rows = rows_per * 2;
    int nb = (cols + 255) / 256;
    int lane = gid % 32;
    int sg = (gid / 32) % 2;
    int tgpig = gid / 64;
    int first_row = tgpig * 4 + sg * 2;
    if (first_row >= total_rows) return;
    int nr = min(2, total_rows - first_row);
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
            int global_row = first_row + r;
            device const uint8_t* W = global_row < rows_per ? W_gate : W_up;
            int local_row = global_row < rows_per ? global_row : global_row - rows_per;
            ulong base = ((ulong)local_row * nb + ib) * 210;
            float d = (float)*(device const half*)(W + base + 208);
            device const uint8_t* ql = W + base + hf * 64 + l_beg + (sub & 1) * 32;
            device const uint8_t* qh = W + base + 128 + hf * 32 + l_beg;
            int scale = (int)*(device const int8_t*)(W + base + 192 + hf * 8 + is + sub * 2);
            for (int k = 0; k < k_lim; k++) {
                uint ql_byte = ql[k];
                uint ql_nib = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
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
//
// ── Fused-pipeline kernels ──
//

// Element-wise: specialized via function constant kernel_op
// 0 = add, 1 = silu_x_up, 2 = cache_write
constant int kernel_op [[function_constant(0)]];
kernel void kernel_elem(device float* data [[buffer(0)]],
                         device const float* aux [[buffer(1)]],
                         constant int* params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    if (gid >= (uint)dim) return;
    if (kernel_op == 0) {
        data[gid] += aux[gid];
    } else if (kernel_op == 1) {
        float x = data[gid];
        data[gid] = (x / (1 + exp(-x))) * data[gid + dim];
    } else if (kernel_op == 2) {
        data[gid] = aux[gid];
    }
}

// Rope: in-place rotation
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

// RMSNorm: in-place, 32-thread single-SG dispatch
kernel void kernel_rmsnorm(device float* data [[buffer(0)]],
                            device const float* weight [[buffer(1)]],
                            constant int* params [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    int dim = params[0];
    float my_sum = 0;
    for (int i = gid; i < dim; i += 32)
        my_sum += data[i] * data[i];
    float total = simd_sum(my_sum);
    float rms = sqrt(total / (float)dim + 1e-6f);
    for (int i = gid; i < dim; i += 32)
        data[i] = (data[i] / rms) * weight[i];
}

// Attention: flash-attention single-token GQA, 32-thread TG (1 simdgroup).
// Processes KV in tiles of 8 with online softmax — single pass, no threadgroup memory, no barriers.
kernel void kernel_attn(device const float* Q [[buffer(0)]],
                         device const float* K_cache [[buffer(1)]],
                         device const float* V_cache [[buffer(2)]],
                         device float* output [[buffer(3)]],
                         constant int* params [[buffer(4)]],
                         uint head [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    int n_head = params[0];
    int n_kv_head = params[1];
    int head_dim = params[2];
    int past_len = params[3];
    if (head >= (uint)n_head) return;
    int kv_head = head / (n_head / n_kv_head);
    int d0 = lane * 2, d1 = lane * 2 + 1;
    if (d0 >= head_dim) return;

    float my_q0 = Q[(uint)head * head_dim + d0];
    float my_q1 = Q[(uint)head * head_dim + d1];
    float scale = 1.0 / sqrt((float)head_dim);

    // Flash attention: online softmax, single pass, TILE=8
    float O0 = 0, O1 = 0, m = -INFINITY, d = 0;
    int n_pos = past_len + 1;

    for (int tile_start = 0; tile_start < n_pos; tile_start += 8) {
        int tile_end = min(tile_start + 8, n_pos);
        int tile_sz = tile_end - tile_start;

        // Compute scores for this tile
        float s[8];
        for (int t = 0; t < tile_sz; t++) {
            int p = tile_start + t;
            ulong cache_ofs = ((ulong)p * n_kv_head + kv_head) * head_dim;
            float k0 = K_cache[cache_ofs + d0];
            float k1 = K_cache[cache_ofs + d1];
            s[t] = simd_sum(my_q0 * k0 + my_q1 * k1) * scale;
        }

        // Online softmax: find tile max, rescale previous accum, add new contributions
        float m_new = m;
        for (int t = 0; t < tile_sz; t++) m_new = max(m_new, s[t]);
        float old_scale = exp(m - m_new);
        O0 *= old_scale; O1 *= old_scale; d *= old_scale;

        for (int t = 0; t < tile_sz; t++) {
            float e = exp(s[t] - m_new);
            int p = tile_start + t;
            ulong cache_ofs = ((ulong)p * n_kv_head + kv_head) * head_dim;
            O0 += e * V_cache[cache_ofs + d0];
            O1 += e * V_cache[cache_ofs + d1];
            d += e;
        }
        m = m_new;
    }

    output[(uint)head * head_dim + d0] = O0 / d;
    output[(uint)head * head_dim + d1] = O1 / d;
}
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