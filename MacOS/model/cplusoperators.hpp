#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cic_edge {

    template <typename T>
    using AlignedVector = std::vector<T, std::allocator<T>>;

    // ==========================================================
    // 熵編碼量化算子 (z)
    // ==========================================================
    inline void quantize_z_a53(
        const float* __restrict__ z, const float* __restrict__ medians, 
        int32_t* __restrict__ symbols, float* __restrict__ z_hat, int C, int H, int W) 
    {
        int HW = H * W;
        for (int c = 0; c < C; ++c) {
            float med = medians[c];
            int c_offset = c * HW;
            for (int hw = 0; hw < HW; ++hw) {
                int idx = c_offset + hw;
                int32_t sym = static_cast<int32_t>(std::round(z[idx] - med));
                symbols[idx] = sym;
                z_hat[idx] = static_cast<float>(sym) + med;
            }
        }
    }

    // ==========================================================
    // 熵編碼量化算子 (y) - 二元搜尋維持不變
    // ==========================================================
    inline void quantize_and_index_y_a53(
        const float* __restrict__ y, const float* __restrict__ scales, const float* __restrict__ means,
        const float* __restrict__ scale_table, int num_scales, 
        int32_t* __restrict__ symbols, int32_t* __restrict__ indexes, int C, int H, int W)
    {
        int total = C * H * W;
        for (int i = 0; i < total; ++i) {
            symbols[i] = static_cast<int32_t>(std::round(y[i] - means[i]));
            float target_scale = scales[i];
            
            if (target_scale <= scale_table[0]) { indexes[i] = 0; continue; }
            if (target_scale >= scale_table[num_scales - 1]) { indexes[i] = num_scales - 1; continue; }

            const float* it = std::lower_bound(scale_table, scale_table + num_scales, target_scale);
            float diff_right = *it - target_scale;
            float diff_left = target_scale - *(it - 1);
            
            indexes[i] = (diff_right < diff_left) ? std::distance(scale_table, it) : std::distance(scale_table, it - 1);
        }
    }

} // namespace cic_edge