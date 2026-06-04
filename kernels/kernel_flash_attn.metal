#include <metal_stdlib>
using namespace metal;

// Flash Attention with online softmax — GQA-correct
// simd_sum-based dot product (not simdgroup), covers all 64 dims
// 64 threads/TG (2 simdgroups), 4 query heads per simdgroup
// Each thread handles 2 dims: d0=tiisg*2, d1=tiisg*2+1
// Processes ALL KV positions (not limited to 4 per tile)
kernel void kernel_flash_attn(
        device const float* Q [[buffer(0)]],
        device const float* K_cache [[buffer(1)]],
        device const float* V_cache [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant int* params [[buffer(4)]],
        uint tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 64;
    constexpr short DV = 64;
    constexpr short Q_PER_TG = 8;
    constexpr short C = 16;

    int n_head = params[0];
    int n_kv_head = params[1];
    int head_dim = params[2];
    int past_len = params[3];
    float scale = 1.0f / sqrt((float)head_dim);

    int n_tg = (n_head + Q_PER_TG - 1) / Q_PER_TG;
    if (tgpig >= (uint)n_tg) return;

    int q_base = tgpig * Q_PER_TG;
    int q_start = q_base + sgitg * (Q_PER_TG / 2);
    int nq_sg = Q_PER_TG / 2;
    if (q_start + nq_sg > n_head) nq_sg = n_head - q_start;
    if (nq_sg <= 0) return;

    int d0 = tiisg * 2;
    int d1 = tiisg * 2 + 1;
    if (d0 >= DK) return;

    float O_lo[4], O_hi[4];
    float m_val[4], d_val[4];
    for (short h = 0; h < 4; h++) {
        O_lo[h] = 0.0f; O_hi[h] = 0.0f;
        m_val[h] = -1e9f; d_val[h] = 0.0f;
    }

    int n_pos = past_len + 1;
    for (int tile_start = 0; tile_start < n_pos; tile_start += C) {
        int tile_end = min(tile_start + C, n_pos);
        int tile_sz = tile_end - tile_start;

        for (short hh = 0; hh < nq_sg; hh++) {
            int q_idx = q_start + hh;
            int kv_head = q_idx * n_kv_head / n_head;

            float q0 = Q[q_idx * DK + d0];
            float q1 = Q[q_idx * DK + d1];
            float row_max = m_val[hh];
            float row_sum = d_val[hh];

            // Process ALL positions in this tile
            float s[16];
            for (int t = 0; t < tile_sz; t++) {
                int p = tile_start + t;
                ulong base = ((ulong)p * n_kv_head + kv_head) * DK;
                float k0 = K_cache[base + d0];
                float k1 = K_cache[base + d1];
                float partial = q0 * k0 + q1 * k1;
                s[t] = simd_sum(partial) * scale;
            }

            // Online softmax
            float new_max = row_max;
            for (int t = 0; t < tile_sz; t++)
                new_max = max(new_max, s[t]);

            if (row_max > new_max) {
                float old_scale = exp(row_max - new_max);
                O_lo[hh] *= old_scale;
                O_hi[hh] *= old_scale;
                row_sum *= old_scale;
            }

            for (int t = 0; t < tile_sz; t++) {
                float exp_val = exp(s[t] - new_max);
                row_sum += exp_val;
                int p = tile_start + t;
                ulong base = ((ulong)p * n_kv_head + kv_head) * DV;
                O_lo[hh] += exp_val * V_cache[base + d0];
                O_hi[hh] += exp_val * V_cache[base + d1];
            }

            m_val[hh] = new_max;
            d_val[hh] = row_sum;
        }
    }

    for (short hh = 0; hh < nq_sg; hh++) {
        int q_idx = q_start + hh;
        float inv_d = 1.0f / (d_val[hh] + 1e-8f);
        output[q_idx * DV + d0] = O_lo[hh] * inv_d;
        output[q_idx * DV + d1] = O_hi[hh] * inv_d;
    }
}
