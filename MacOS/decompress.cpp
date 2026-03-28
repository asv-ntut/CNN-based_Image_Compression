#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <cstdlib>
#include <zlib.h> // 加入 zlib 用於 CRC32 校驗

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
// 1. Binary Packet Header (對齊新版 CIC 格式)
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
// 2. IO 與影像寫入工具
// ==========================================================
std::vector<uint8_t> read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filepath);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void write_patch_to_image_8u(cv::Mat& img_8u, const float* buffer_ptr, int r, int c, int ps) {
    int s_h = r * ps, s_w = c * ps;
    int v_h = std::min(ps, img_8u.rows - s_h);
    int v_w = std::min(ps, img_8u.cols - s_w);
    if (v_h > 0 && v_w > 0) {
        cv::Mat crop = img_8u(cv::Rect(s_w, s_h, v_w, v_h));
        for (int i = 0; i < v_h; ++i) {
            uint8_t* row = crop.ptr<uint8_t>(i);
            for (int j = 0; j < v_w; ++j) {
                row[j*3+0] = static_cast<uint8_t>(std::round(std::clamp(buffer_ptr[0*ps*ps + i*ps + j] * 255.0f, 0.0f, 255.0f)));
                row[j*3+1] = static_cast<uint8_t>(std::round(std::clamp(buffer_ptr[1*ps*ps + i*ps + j] * 255.0f, 0.0f, 255.0f)));
                row[j*3+2] = static_cast<uint8_t>(std::round(std::clamp(buffer_ptr[2*ps*ps + i*ps + j] * 255.0f, 0.0f, 255.0f)));
            }
        }
    }
}

// ==========================================================
// 3. Shared Decoder Resource
// ==========================================================
struct SharedORTDecoder {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CIC_Board_Decoder"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> dec_sess;
    std::unique_ptr<Ort::Session> hyp_sess;
    const char* dec_in[1] = {"y_hat"};
    const char* dec_out[1] = {"x_hat"};
    const char* hyp_in[1] = {"z_hat"};
    const char* hyp_out[2] = {"scales", "means"};

    SharedORTDecoder(const std::string& dp, const std::string& hp) {
        options.SetIntraOpNumThreads(4); // 適配 A53 四核心
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        dec_sess = std::make_unique<Ort::Session>(env, dp.c_str(), options);
        hyp_sess = std::make_unique<Ort::Session>(env, hp.c_str(), options);
    }
};

// ==========================================================
// 4. Main
// ==========================================================
int main(int argc, char** argv) {
    try {
        const std::string DEFAULT_ORIGINAL_TIF = "../Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif"; 
        const std::string DEFAULT_BIN_DIR      = "./compressed_bins";
        const std::string MODEL_DECODER        = "./model/onnx/decoder.onnx";
        const std::string MODEL_HYPER_DECODER  = "./model/onnx/hyper_decoder.onnx";

        std::string original_tif = (argc > 1) ? argv[1] : DEFAULT_ORIGINAL_TIF;
        std::string base_bin_dir = (argc > 2) ? argv[2] : DEFAULT_BIN_DIR;

        const char* home_dir = std::getenv("HOME");
        if (!home_dir) throw std::runtime_error("Cannot find HOME environment variable.");
        fs::path output_dir = "./output";
        if (!fs::exists(output_dir)) fs::create_directories(output_dir);

        std::string file_stem = fs::path(original_tif).stem().string();
        std::string input_patch_dir = (fs::path(base_bin_dir) / file_stem).string();
        std::string result_img_path = (output_dir / (file_stem + "_reconstructed.png")).string();

        std::cout << "[INFO] Loading image dimensions..." << std::endl;
        cv::Mat original_img = cv::imread(original_tif, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        if (original_img.empty()) throw std::runtime_error("Failed to read: " + original_tif);
        
        int rows = original_img.rows;
        int cols = original_img.cols;
        original_img.release(); // 防止 OOM
        
        int ps = 256;
        int nr = (rows + ps - 1) / ps, nc = (cols + ps - 1) / ps;
        int total_patches = nr * nc;

        SharedORTDecoder res(MODEL_DECODER, MODEL_HYPER_DECODER);
        PureDynamicRansDecoder2Way rans_dec;
        auto m_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        cv::Mat rec_img_8u = cv::Mat::zeros(rows, cols, CV_8UC3); 

        const int DEC_BATCH = 4; // A53 Batch 4
        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "[INFO] Starting decode loop for " << total_patches << " patches..." << std::endl;

        for (int i = 0; i < total_patches; i += DEC_BATCH) {
            int cur_batch = std::min(DEC_BATCH, total_patches - i);
            AlignedVector<float> b_z_hat(cur_batch * 32 * 4 * 4);
            AlignedVector<float> b_y_hat(cur_batch * 192 * 16 * 16);
            std::vector<std::vector<uint8_t>> b_z_bits(cur_batch), b_y_bits(cur_batch);

            // 讀取並解析單一 .bin 封包
            for (int b = 0; b < cur_batch; ++b) {
                int patch_idx = i + b;
                std::string filepath = input_patch_dir + "/patch_" + std::to_string(patch_idx) + ".bin";
                std::vector<uint8_t> packet = read_binary_file(filepath);

                if (packet.size() < sizeof(CICPacketHeader) + 4) {
                    throw std::runtime_error("Packet too short: " + filepath);
                }

                // 校驗 CRC32
                uint32_t stored_crc;
                std::memcpy(&stored_crc, packet.data() + packet.size() - 4, 4);
                uint32_t calc_crc = crc32(0L, packet.data(), packet.size() - 4);
                
                if (stored_crc != calc_crc) {
                    throw std::runtime_error("CRC checksum failed for patch " + std::to_string(patch_idx) + ". File is corrupted!");
                }

                // 解析 Header
                CICPacketHeader header;
                std::memcpy(&header, packet.data(), sizeof(CICPacketHeader));
                if (header.magic[0] != 'C' || header.magic[1] != 'I' || header.magic[2] != 'C') {
                    throw std::runtime_error("Invalid Magic Number in patch " + std::to_string(patch_idx));
                }

                // 提取 Y 與 Z 的位元流
                size_t y_start = sizeof(CICPacketHeader);
                size_t z_start = y_start + header.y_len;

                b_y_bits[b].assign(packet.begin() + y_start, packet.begin() + z_start);
                b_z_bits[b].assign(packet.begin() + z_start, packet.begin() + z_start + header.z_len);
            }

            for (int b = 0; b < cur_batch; ++b) {
                AlignedVector<int32_t> z_idx_map(32*4*4), z_sym(32*4*4);
                for (size_t k = 0; k < 32*4*4; ++k) z_idx_map[k] = (k/(4*4))%32;
                rans_dec.decode_with_indexes(b_z_bits[b], z_idx_map.data(), 32*4*4, FIXED_EB_CDF, FIXED_EB_LENGTH, FIXED_EB_OFFSET, z_sym.data());
                dequantize_z_a53(z_sym.data(), FIXED_EB_MEDIANS.data(), b_z_hat.data() + b*(32*4*4), 32, 4, 4);
            }

            std::vector<int64_t> h_shape = {cur_batch, 32, 4, 4};
            Ort::Value z_t = Ort::Value::CreateTensor<float>(m_info, b_z_hat.data(), b_z_hat.size(), h_shape.data(), h_shape.size());
            auto h_outs = res.hyp_sess->Run(Ort::RunOptions{nullptr}, res.hyp_in, &z_t, 1, res.hyp_out, 2);
            float *s_p = h_outs[0].GetTensorMutableData<float>(), *m_p = h_outs[1].GetTensorMutableData<float>();

            for (int b = 0; b < cur_batch; ++b) {
                int patch_idx = i + b;
                AlignedVector<int32_t> y_idx(192*16*16), y_sym(192*16*16);
                compute_y_indexes_a53(s_p + b*(192*16*16), m_p + b*(192*16*16), FIXED_GC_SCALE_TABLE.data(), FIXED_GC_SCALE_TABLE.size(), y_idx.data(), 192, 16, 16);
                // ==========================================================
                // 🌟 [步驟 A]：在此處插入驗證碼
                // ==========================================================
                long long idx_sum = 0;
                for(int k=0; k < 192*16*16; ++k) {
                    idx_sum += y_idx[k];
                }
                // 印出該 Patch 的編號與 Index 總和，方便與 A53 的結果對照
                std::cout << "[CHECK] Patch " << std::setw(3) << patch_idx 
                        << " | Index Sum: " << std::setw(10) << idx_sum << std::endl;
                // ==========================================================
                rans_dec.decode_with_indexes(b_y_bits[b], y_idx.data(), 192*16*16, FIXED_GC_CDF, FIXED_GC_LENGTH, FIXED_GC_OFFSET, y_sym.data());
                dequantize_y_a53(y_sym.data(), m_p + b*(192*16*16), b_y_hat.data() + b*(192*16*16), 192, 16, 16);
            }

            std::vector<int64_t> d_shape = {cur_batch, 192, 16, 16};
            Ort::Value y_t = Ort::Value::CreateTensor<float>(m_info, b_y_hat.data(), b_y_hat.size(), d_shape.data(), d_shape.size());
            auto d_outs = res.dec_sess->Run(Ort::RunOptions{nullptr}, res.dec_in, &y_t, 1, res.dec_out, 1);
            float* x_ptr = d_outs[0].GetTensorMutableData<float>();
            
            for (int b = 0; b < cur_batch; ++b) {
                int patch_idx = i + b;
                write_patch_to_image_8u(rec_img_8u, x_ptr + b*(3*ps*ps), patch_idx/nc, patch_idx%nc, ps);
            }
            std::cout << "\r[Progress] " << i + cur_batch << " / " << total_patches << std::flush;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();

        res.dec_sess.reset(); 
        res.hyp_sess.reset();

        std::cout << "\n[INFO] Decoding finished. Saving image..." << std::endl;
        cv::Mat rec_bgr;
        cv::cvtColor(rec_img_8u, rec_bgr, cv::COLOR_RGB2BGR);
        
        if(cv::imwrite(result_img_path, rec_bgr)) {
            std::cout << "=========================================\n";
            std::cout << " CIC Board Decoder Report\n";
            std::cout << "=========================================\n";
            std::cout << std::left << std::setw(25) << "Result Image" << ": " << result_img_path << "\n";
            std::cout << std::left << std::setw(25) << "Total Time"    << ": " << std::fixed << std::setprecision(4) << elapsed << " s\n";
            std::cout << "=========================================\n";
        } else {
            std::cerr << "\n[ERROR] Failed to write image to disk.\n";
        }

    } catch (const std::exception& e) { std::cerr << "\nError: " << e.what() << "\n"; }
    return 0;
}