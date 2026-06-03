#include <metal_stdlib>
using namespace metal;

// Flash Attention with Tensor Cores - dk64 variant
// Load K/V from device memory, use simdgroup for QxK and online softmax
kernel void kernel_flash_attn(
        device const float* Q [[buffer(0)]],
        device const float* K_cache [[buffer(1)]],
        device const float* V_cache [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant int* params [[buffer(4)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 64;
    constexpr short DV = 64;
    constexpr short Q_PER_TG = 8;
    constexpr short NSG = 2;
    constexpr short NQ = Q_PER_TG / NSG;
    constexpr short C = 16;
    constexpr short DK8 = DK / 8;

    int n_head = params[0];
    int n_kv_head = params[1];
    int head_dim = params[2];
    int past_len = params[3];
    float scale = 1.0f / sqrt((float)head_dim);

    int tg_idx = tgpig;
    int n_tg = (n_head + Q_PER_TG - 1) / Q_PER_TG;
    if (tg_idx >= n_tg) return;

    int kv_head_start = tg_idx * n_kv_head / n_tg;
    if (kv_head_start >= n_kv_head) return;

    threadgroup float sq[Q_PER_TG * DK];
    threadgroup float so[Q_PER_TG * DV];
    threadgroup float smax[Q_PER_TG];
    threadgroup float ssum[Q_PER_TG];

    int q_base = tg_idx * Q_PER_TG;

    for (short j = 0; j < Q_PER_TG; j++) {
        int q_idx = q_base + j;
        if (q_idx < n_head) {
            if (tiisg < DK) {
                sq[j * DK + tiisg] = Q[q_idx * DK + tiisg];
            }
        } else if (tiisg < DK) {
            sq[j * DK + tiisg] = 0.0f;
        }
        if (tiisg < DV) {
            so[j * DV + tiisg] = 0.0f;
        }
    }
    if (tiisg < Q_PER_TG) {
        smax[tiisg] = -1e9f;
        ssum[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int iter = 0; iter < past_len + 1; iter += C) {
        int cur_C = (iter + C > past_len + 1) ? (past_len + 1 - iter) : C;
        if (cur_C <= 0) break;

        for (short j = 0; j < Q_PER_TG; j++) {
            int q_idx = q_base + j;
            if (q_idx >= n_head) continue;

            float row_max = smax[j];
            float row_sum = ssum[j];
            float scores[4];

            for (short c = 0; c < 4 && c < cur_C; c++) {
                int p = iter + c;
                if (p >= past_len + 1) {
                    scores[c] = -1e9f;
                    continue;
                }

                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_float8x8 mq, mk;

                device const float* k_ptr = K_cache + ((ulong)p * n_kv_head + kv_head_start) * DK;

                for (short ii = 0; ii < DK8; ii++) {
                    simdgroup_load(mq, sq + j * DK + ii * 8, 8);
                    simdgroup_load(mk, k_ptr + ii * 8, DK);
                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                }

                threadgroup float tmp[8];
                simdgroup_store(mqk, tmp, 8);
                float sum = 0;
                for (int i = 0; i < 8; i++) sum += tmp[i];
                scores[c] = sum * scale;
            }

            float new_max = row_max;
            for (short c = 0; c < 4 && c < cur_C; c++) {
                new_max = max(new_max, scores[c]);
            }

            if (row_max > new_max) {
                float scale_old = exp(row_max - new_max);
                if (tiisg < DV) so[j * DV + tiisg] *= scale_old;
                row_sum *= scale_old;
            }

            for (short c = 0; c < 4 && c < cur_C; c++) {
                int p = iter + c;
                if (p >= past_len + 1) continue;

                float exp_val = exp(scores[c] - new_max);
                row_sum += exp_val;

                device const float* v_ptr = V_cache + ((ulong)p * n_kv_head + kv_head_start) * DV;
                if (tiisg < DV) {
                    so[j * DV + tiisg] += exp_val * v_ptr[tiisg];
                }
            }

            if (tiisg == j) {
                smax[j] = new_max;
                ssum[j] = row_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short j = 0; j < Q_PER_TG; j++) {
        int q_idx = q_base + j;
        if (q_idx >= n_head) continue;
        float lnorm = 1.0f / (ssum[j] + 1e-8f);
        if (tiisg < DV) {
            output[q_idx * DV + tiisg] = so[j * DV + tiisg] * lnorm;
        }
    }
}