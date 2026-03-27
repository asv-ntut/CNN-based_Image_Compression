#include "dynamic_rans_2way.hpp"
#include <algorithm>
#include <stdexcept>

// 基礎 ANS 狀態更新函式
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

// 2-Way Interleaved 編碼器
py::bytes DynamicRansEncoder2Way::encode_with_indexes(
    py::array_t<int32_t> symbols, 
    py::array_t<int32_t> indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets) 
{
    // 🚀 [核心優化] 零拷貝提取底層指標
    py::buffer_info sym_buf = symbols.request();
    py::buffer_info idx_buf = indexes.request();

    if (sym_buf.size != idx_buf.size) {
        throw std::runtime_error("Symbols and indexes must have the same size");
    }

    size_t N = sym_buf.size;
    if (N == 0) return py::bytes("");

    // 取得真正的 C 陣列指標
    const int32_t* sym_ptr = static_cast<int32_t*>(sym_buf.ptr);
    const int32_t* idx_ptr = static_cast<int32_t*>(idx_buf.ptr);

    std::vector<uint32_t> output(N + 1024, 0xCC);
    uint32_t* ptr = output.data() + output.size();

    Rans64State rans0, rans1;
    Rans64EncInit(&rans0);
    Rans64EncInit(&rans1);

    size_t i = N;
    if (i % 2 != 0) {
        int32_t idx = idx_ptr[i - 1]; // 直接用指標存取
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
    return std::string(reinterpret_cast<char*>(ptr), nbytes);
}

// 2-Way Interleaved 解碼器
py::array_t<int32_t> DynamicRansDecoder2Way::decode_with_indexes(
    const std::string& encoded, 
    py::array_t<int32_t> indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdfs_sizes,
    const std::vector<int32_t>& offsets) 
{
    // 🚀 [核心優化] 零拷貝提取索引指標
    py::buffer_info idx_buf = indexes.request();
    size_t N = idx_buf.size;
    
    // 直接建立一個空的 Numpy 陣列作為回傳值
    auto result = py::array_t<int32_t>(N);
    py::buffer_info res_buf = result.request();
    int32_t* out_ptr = static_cast<int32_t*>(res_buf.ptr);
    const int32_t* idx_ptr = static_cast<int32_t*>(idx_buf.ptr);

    if (N == 0) return result;
    
    Rans64State rans0, rans1;
    uint32_t* ptr = (uint32_t*)encoded.data();
    
    Rans64DecInit(&rans0, &ptr);
    Rans64DecInit(&rans1, &ptr);

    size_t i = 0;
    for (; i < (N & ~1); i += 2) {
        int32_t idx0 = idx_ptr[i];
        int32_t idx1 = idx_ptr[i + 1];
        const auto& cdf0 = cdfs[idx0];
        const auto& cdf1 = cdfs[idx1];

        uint32_t cum_freq0 = rans0 & ((1u << 16) - 1);
        uint32_t cum_freq1 = rans1 & ((1u << 16) - 1);

        auto it0 = std::upper_bound(cdf0.begin(), cdf0.begin() + cdfs_sizes[idx0], cum_freq0);
        auto it1 = std::upper_bound(cdf1.begin(), cdf1.begin() + cdfs_sizes[idx1], cum_freq1);
        
        uint32_t s0 = std::distance(cdf0.begin(), it0) - 1;
        uint32_t s1 = std::distance(cdf1.begin(), it1) - 1;

        out_ptr[i + 0] = s0 + offsets[idx0]; // 直接寫入 Numpy 記憶體
        out_ptr[i + 1] = s1 + offsets[idx1];

        uint64_t x0 = rans0;
        x0 = (cdf0[s0 + 1] - cdf0[s0]) * (x0 >> 16) + cum_freq0 - cdf0[s0];
        if (x0 < RANS64_L) { x0 = (x0 << 32) | *ptr++; }
        rans0 = x0;

        uint64_t x1 = rans1;
        x1 = (cdf1[s1 + 1] - cdf1[s1]) * (x1 >> 16) + cum_freq1 - cdf1[s1];
        if (x1 < RANS64_L) { x1 = (x1 << 32) | *ptr++; }
        rans1 = x1;
    }

    if (i < N) {
        int32_t idx0 = idx_ptr[i];
        const auto& cdf0 = cdfs[idx0];
        uint32_t cum_freq0 = rans0 & ((1u << 16) - 1);
        auto it0 = std::upper_bound(cdf0.begin(), cdf0.begin() + cdfs_sizes[idx0], cum_freq0);
        uint32_t s0 = std::distance(cdf0.begin(), it0) - 1;
        out_ptr[i] = s0 + offsets[idx0];
    }

    return result; // 直接把 Numpy 陣列丟回 Python，0 轉換時間！
}