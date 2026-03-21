#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // 🚨 新增：引入 numpy 支援
#include <vector>
#include <string>
#include "rans64.h"

namespace py = pybind11;

class DynamicRansEncoder2Way {
public:
    DynamicRansEncoder2Way() = default;

    py::bytes encode_with_indexes(
        py::array_t<int32_t> symbols,    
        py::array_t<int32_t> indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
};

class DynamicRansDecoder2Way {
public:
    DynamicRansDecoder2Way() = default;

    py::array_t<int32_t> decode_with_indexes(
        const std::string& encoded, 
        py::array_t<int32_t> indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
};