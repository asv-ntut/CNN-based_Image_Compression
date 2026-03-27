#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

// 引入最底層的 Rans64 算子 (確保 rans64.h 在同一個目錄)
#include "rans/rans64.h"

namespace tic_edge {

// ==========================================================
// 基礎 ANS 狀態更新函式
// ==========================================================
static inline void Rans64EncPutDynamic(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq) {
    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t) x;
        x >>= 32;
    }
    *r = ((x / freq) << 16) + (x % freq) + start;
}

// ==========================================================
// 🌟 2-Way Interleaved 編碼器
// ==========================================================
class PureDynamicRansEncoder2Way {
public:
    PureDynamicRansEncoder2Way() = default;

    std::vector<uint8_t> encode_with_indexes(
        const int32_t* sym_ptr, 
        const int32_t* idx_ptr,
        size_t N, 
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes, // 保留相容性
        const std::vector<int32_t>& offsets) 
    {
        if (N == 0) return std::vector<uint8_t>();

        std::vector<uint32_t> output(N + 1024, 0xCC);
        uint32_t* ptr = output.data() + output.size();

        Rans64State rans0, rans1;
        Rans64EncInit(&rans0);
        Rans64EncInit(&rans1);

        size_t i = N;
        if (i % 2 != 0) {
            int32_t idx = idx_ptr[i - 1];
            int32_t val = sym_ptr[i - 1] - offsets[idx];
            const auto& cdf = cdfs[idx];
            Rans64EncPutDynamic(&rans0, &ptr, cdf[val], cdf[val + 1] - cdf[val]);
            i--;
        }

        for (; i > 0; i -= 2) {
            int32_t idx1 = idx_ptr[i - 1];
            int32_t idx0 = idx_ptr[i - 2];
            
            int32_t val1 = sym_ptr[i - 1] - offsets[idx1];
            int32_t val0 = sym_ptr[i - 2] - offsets[idx0];
            
            const auto& cdf1 = cdfs[idx1];
            const auto& cdf0 = cdfs[idx0];

            Rans64EncPutDynamic(&rans1, &ptr, cdf1[val1], cdf1[val1 + 1] - cdf1[val1]);
            Rans64EncPutDynamic(&rans0, &ptr, cdf0[val0], cdf0[val0 + 1] - cdf0[val0]);
        }

        Rans64EncFlush(&rans1, &ptr);
        Rans64EncFlush(&rans0, &ptr);

        const int nbytes = std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
        return std::vector<uint8_t>(byte_ptr, byte_ptr + nbytes);
    }
};

// ==========================================================
// 🌟 2-Way Interleaved 解碼器
// ==========================================================
class PureDynamicRansDecoder2Way {
public:
    PureDynamicRansDecoder2Way() = default;

    void decode_with_indexes(
        const std::vector<uint8_t>& encoded_data,
        const int32_t* idx_ptr,
        size_t N,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& /* cdfs_sizes */, // 註解掉變數名稱消除 unused 警告
        const std::vector<int32_t>& offsets,
        int32_t* decoded_sym_ptr) 
    {
        if (N == 0 || encoded_data.empty()) return;

        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(encoded_data.data());

        Rans64State rans0, rans1;
        Rans64DecInit(&rans0, (uint32_t**)&ptr);
        Rans64DecInit(&rans1, (uint32_t**)&ptr);

        size_t i = 0;
        for (; i + 1 < N; i += 2) {
            int32_t idx0 = idx_ptr[i];
            int32_t idx1 = idx_ptr[i + 1];

            const auto& cdf0 = cdfs[idx0];
            const auto& cdf1 = cdfs[idx1];

            uint32_t cum0 = Rans64DecGet(&rans0, 16);
            uint32_t cum1 = Rans64DecGet(&rans1, 16);

            int32_t s0 = 0, s1 = 0;
            // 加入強制轉型 (uint32_t) 消除 signed/unsigned 比較警告
            while ((uint32_t)cdf0[s0 + 1] <= cum0) s0++;
            while ((uint32_t)cdf1[s1 + 1] <= cum1) s1++;

            decoded_sym_ptr[i] = s0 + offsets[idx0];
            decoded_sym_ptr[i + 1] = s1 + offsets[idx1];

            Rans64DecAdvance(&rans0, (uint32_t**)&ptr, cdf0[s0], cdf0[s0+1]-cdf0[s0], 16);
            Rans64DecAdvance(&rans1, (uint32_t**)&ptr, cdf1[s1], cdf1[s1+1]-cdf1[s1], 16);
        }

        if (i < N) {
            int32_t idx = idx_ptr[i];
            const auto& cdf = cdfs[idx];
            uint32_t cum = Rans64DecGet(&rans0, 16);
            int32_t s = 0;
            while ((uint32_t)cdf[s + 1] <= cum) s++;
            decoded_sym_ptr[i] = s + offsets[idx];
            Rans64DecAdvance(&rans0, (uint32_t**)&ptr, cdf[s], cdf[s+1]-cdf[s], 16);
        }
    }
};

} // namespace tic_edge