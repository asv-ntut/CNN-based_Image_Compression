#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <atomic>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <map>
#include <cmath>
#include <zlib.h> 

// ONNX Runtime & OpenCV
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Core optimization libraries
#include "./model/cplusoperators.hpp"
#include "./model/cdfs/fixed_cdfs.hpp"
#include "./model/pure_rans_2way.hpp"

using namespace cic_edge;
namespace fs = std::filesystem;

// ==========================================================
// 1. Binary Packet Header
// ==========================================================
#pragma pack(push, 1)
struct CICPacketHeader {
    char magic[3];     // 'CIC'
    uint8_t img_id;
    uint8_t row;
    uint8_t col;
    uint16_t z_h;
    uint16_t z_w;
    uint32_t y_len;
    uint32_t z_len;
};
#pragma pack(pop)

// ==========================================================
// 2. 工具函式
// ==========================================================
void extract_patch_to_buffer(const cv::Mat& img_float, float* buffer_ptr, int r, int c, int ps) {
    int s_h = r * ps, s_w = c * ps;
    int v_h = std::min(ps, img_float.rows - s_h);
    int v_w = std::min(ps, img_float.cols - s_w);
    std::fill(buffer_ptr, buffer_ptr + (3 * ps * ps), 0.0f);

    if (v_h > 0 && v_w > 0) {
        cv::Mat crop = img_float(cv::Rect(s_w, s_h, v_w, v_h));
        for (int i = 0; i < v_h; ++i) {
            const float* row = crop.ptr<float>(i);
            for (int j = 0; j < v_w; ++j) {
                buffer_ptr[0*ps*ps + i*ps + j] = row[j*3+0];
                buffer_ptr[1*ps*ps + i*ps + j] = row[j*3+1];
                buffer_ptr[2*ps*ps + i*ps + j] = row[j*3+2];
            }
        }
    }
}

// ==========================================================
// 3. Shared resource structure
// ==========================================================
struct SharedORTResource {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CIC_Final"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> enc_sess;
    std::unique_ptr<Ort::Session> hyp_sess;

    const char* enc_in[1] = {"input_image"};
    const char* enc_out[2] = {"y", "z"};
    const char* hyp_in[1] = {"z_hat"};
    const char* hyp_out[2] = {"scales", "means"};

    SharedORTResource(const std::string& ep, const std::string& hp) {
        options.SetIntraOpNumThreads(4); 
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        enc_sess = std::make_unique<Ort::Session>(env, ep.c_str(), options);
        hyp_sess = std::make_unique<Ort::Session>(env, hp.c_str(), options);
    }
};

// ==========================================================
// 4. Main function
// ==========================================================
int main(int argc, char** argv) {
    try {
        const std::string DEFAULT_ORIGINAL_TIF = "../Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif";
        const std::string DEFAULT_BIN_DIR      = "./compressed_bins";
        const std::string MODEL_ENCODER        = "./model/onnx/encoder.onnx";
        const std::string MODEL_HYPER_DECODER  = "./model/onnx/hyper_decoder.onnx";

        std::string tif = (argc > 1) ? argv[1] : DEFAULT_ORIGINAL_TIF;
        std::string base_bin_dir = (argc > 2) ? argv[2] : DEFAULT_BIN_DIR;
        int ENC_BATCH = (argc > 3) ? std::stoi(argv[3]) : 4;
        int HYP_BATCH = ENC_BATCH * 4;

        std::string file_stem = fs::path(tif).stem().string();
        std::string final_output_dir = (fs::path(base_bin_dir) / file_stem).string();
        if (!fs::exists(final_output_dir)) fs::create_directories(final_output_dir);

        std::map<std::string, double> stats;
        std::vector<std::string> steps = {"load", "extract", "prep", "encoder", "entropy_z", "hyper", "entropy_y", "save"};
        for (const auto& s : steps) stats[s] = 0.0;

        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imread(tif, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        if (img.empty()) throw std::runtime_error("Cannot read image: " + tif);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat img_f; img.convertTo(img_f, CV_32FC3, 1.0/255.0);
        stats["load"] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();

        size_t orig_file_size = fs::file_size(tif);
        int ps = 256;
        int nr = (img_f.rows + ps - 1) / ps, nc = (img_f.cols + ps - 1) / ps;
        int total_patches = nr * nc;

        SharedORTResource res(MODEL_ENCODER, MODEL_HYPER_DECODER);
        PureDynamicRansEncoder2Way rans;
        auto m_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<AlignedVector<float>> all_y_raw(total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_z_hat(total_patches, AlignedVector<float>(32*4*4));
        std::vector<AlignedVector<int32_t>> all_z_sym(total_patches, AlignedVector<int32_t>(32*4*4));
        std::vector<AlignedVector<float>> all_scales(total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_means(total_patches, AlignedVector<float>(192*16*16));

        t_start = std::chrono::high_resolution_clock::now();
        std::vector<AlignedVector<float>> patches_raw(total_patches, AlignedVector<float>(3*ps*ps));
        for (int i = 0; i < total_patches; ++i) {
            extract_patch_to_buffer(img_f, patches_raw[i].data(), i / nc, i % nc, ps);
        }
        stats["extract"] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();

        for (int i = 0; i < total_patches; i += ENC_BATCH) {
            int cur_batch = std::min(ENC_BATCH, total_patches - i);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            AlignedVector<float> stacked_in(cur_batch * 3 * ps * ps);
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(patches_raw[i+b].begin(), patches_raw[i+b].end(), stacked_in.begin() + b*(3*ps*ps));
            }
            stats["prep"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            std::vector<int64_t> enc_shape = {cur_batch, 3, ps, ps};
            Ort::Value in_t = Ort::Value::CreateTensor<float>(m_info, stacked_in.data(), stacked_in.size(), enc_shape.data(), enc_shape.size());
            auto outs = res.enc_sess->Run(Ort::RunOptions{nullptr}, res.enc_in, &in_t, 1, res.enc_out, 2);
            stats["encoder"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            float* y_ptr = outs[0].GetTensorMutableData<float>();
            float* z_ptr = outs[1].GetTensorMutableData<float>();
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(y_ptr + b*(192*16*16), y_ptr + (b+1)*(192*16*16), all_y_raw[i+b].begin());
                quantize_z_a53(z_ptr + b*(32*4*4), FIXED_EB_MEDIANS.data(), all_z_sym[i+b].data(), all_z_hat[i+b].data(), 32, 4, 4);
            }
            stats["entropy_z"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "\r[1/3] Encoder: " << i + cur_batch << " / " << total_patches << std::flush;
        }

        for (int i = 0; i < total_patches; i += HYP_BATCH) {
            auto t0 = std::chrono::high_resolution_clock::now();
            int cur_batch = std::min(HYP_BATCH, total_patches - i);
            AlignedVector<float> stacked_z(cur_batch * 32 * 4 * 4);
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(all_z_hat[i+b].begin(), all_z_hat[i+b].end(), stacked_z.begin() + b*(32*4*4));
            }
            std::vector<int64_t> hyp_shape = {cur_batch, 32, 4, 4};
            Ort::Value z_t = Ort::Value::CreateTensor<float>(m_info, stacked_z.data(), stacked_z.size(), hyp_shape.data(), hyp_shape.size());
            auto h_outs = res.hyp_sess->Run(Ort::RunOptions{nullptr}, res.hyp_in, &z_t, 1, res.hyp_out, 2);

            float* s_ptr = h_outs[0].GetTensorMutableData<float>();
            float* m_ptr = h_outs[1].GetTensorMutableData<float>();
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(s_ptr + b*(192*16*16), s_ptr + (b+1)*(192*16*16), all_scales[i+b].begin());
                std::copy(m_ptr + b*(192*16*16), m_ptr + (b+1)*(192*16*16), all_means[i+b].begin());
            }
            stats["hyper"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "\r[2/3] Hyper Decoder: " << i + cur_batch << " / " << total_patches << std::flush;
        }

        size_t total_compressed_bytes = 0;
        for (int i = 0; i < total_patches; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            AlignedVector<int32_t> y_sym(192*16*16), y_idx(192*16*16);
            quantize_and_index_y_a53(all_y_raw[i].data(), all_scales[i].data(), all_means[i].data(), 
                                     FIXED_GC_SCALE_TABLE.data(), FIXED_GC_SCALE_TABLE.size(), 
                                     y_sym.data(), y_idx.data(), 192, 16, 16);
            auto y_bits = rans.encode_with_indexes(y_sym.data(), y_idx.data(), 192*16*16, FIXED_GC_CDF, FIXED_GC_LENGTH, FIXED_GC_OFFSET);
            stats["entropy_y"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            AlignedVector<int32_t> z_idx_map(32*4*4);
            for (size_t k = 0; k < 32*4*4; ++k) z_idx_map[k] = (k / (4*4)) % 32;
            auto z_bits = rans.encode_with_indexes(all_z_sym[i].data(), z_idx_map.data(), 32*4*4, FIXED_EB_CDF, FIXED_EB_LENGTH, FIXED_EB_OFFSET);
            stats["entropy_z"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            CICPacketHeader header = { {'C', 'I', 'C'}, 1, (uint8_t)(i / nc), (uint8_t)(i % nc), 4, 4, (uint32_t)y_bits.size(), (uint32_t)z_bits.size() };
            
            std::vector<uint8_t> full_payload;
            full_payload.reserve(sizeof(header) + y_bits.size() + z_bits.size() + 4);
            full_payload.insert(full_payload.end(), (uint8_t*)&header, (uint8_t*)&header + sizeof(header));
            full_payload.insert(full_payload.end(), y_bits.begin(), y_bits.end());
            full_payload.insert(full_payload.end(), z_bits.begin(), z_bits.end());
            
            uint32_t crc = crc32(0L, full_payload.data(), full_payload.size());
            full_payload.insert(full_payload.end(), (uint8_t*)&crc, (uint8_t*)&crc + 4);

            std::string fpath = final_output_dir + "/patch_" + std::to_string(i) + ".bin";
            std::ofstream f(fpath, std::ios::binary);
            f.write((char*)full_payload.data(), full_payload.size());
            f.close();
            
            total_compressed_bytes += full_payload.size();
            stats["save"] += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "\r[3/3] Entropy & Save: " << i + 1 << " / " << total_patches << std::flush;
        }

        double total_pixels = (double)img.cols * img.rows;
        double avg_bpp = ((double)total_compressed_bytes * 8.0) / total_pixels;
        double total_time = 0.0;
        for (const auto& s : steps) total_time += stats[s];

        std::cout << "\n\n============================================================\n";
        std::cout << " Performance Report: " << file_stem << "\n";
        std::cout << "============================================================\n";
        for (const auto& s : steps) {
            std::cout << std::left << std::setw(15) << s << " | ";
            if (stats[s] > 0) std::cout << std::fixed << std::setprecision(4) << stats[s] << " s";
            else std::cout << "NaN";
            std::cout << "\n";
        }
        std::cout << "------------------------------------------------------------\n";
        std::cout << std::left << std::setw(25) << "Total Time"            << ": " << std::fixed << std::setprecision(4) << total_time << " s\n";
        std::cout << std::left << std::setw(25) << "Original File Size"    << ": " << std::fixed << std::setprecision(2) << (double)orig_file_size / (1024.0 * 1024.0) << " MB\n";
        std::cout << std::left << std::setw(25) << "Compressed File Size"  << ": " << (double)total_compressed_bytes / (1024.0 * 1024.0) << " MB\n";
        std::cout << std::left << std::setw(25) << "Compression Ratio"     << ": " << (double)orig_file_size / total_compressed_bytes << "\n";
        std::cout << std::left << std::setw(25) << "Average BPP"           << ": " << std::fixed << std::setprecision(4) << avg_bpp << "\n";
        std::cout << "============================================================\n";

    } catch (const std::exception& e) { std::cerr << "Execution Error: " << e.what() << "\n"; }
    return 0;
}