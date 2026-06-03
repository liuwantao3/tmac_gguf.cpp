#include <metal_stdlib>
using namespace metal;

// Attention: flash-attention single-token GQA, 32-thread TG (1 simdgroup).
// Single pass, no threadgroup memory, no barriers.
// Processes KV in tiles of 8 with online softmax.
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

    float O0 = 0, O1 = 0, m = -INFINITY, d = 0;
    int n_pos = past_len + 1;

    for (int tile_start = 0; tile_start < n_pos; tile_start += 8) {
        int tile_end = min(tile_start + 8, n_pos);
        int tile_sz = tile_end - tile_start;

        float s[8];
        for (int t = 0; t < tile_sz; t++) {
            int p = tile_start + t;
            ulong cache_ofs = ((ulong)p * n_kv_head + kv_head) * head_dim;
            float k0 = K_cache[cache_ofs + d0];
            float k1 = K_cache[cache_ofs + d1];
            s[t] = simd_sum(my_q0 * k0 + my_q1 * k1) * scale;
        }

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