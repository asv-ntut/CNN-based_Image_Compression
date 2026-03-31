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
#include <string>
#include <sstream>
#include <cstdlib>
#include <zlib.h> 

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "./model/cplusoperators.hpp"
#include "./model/cdfs/fixed_cdfs.hpp"
#include "./model/pure_rans_2way.hpp"

using namespace cic_edge;
namespace fs = std::filesystem;

#pragma pack(push, 1)
struct CICPacketHeader {
    char magic[3];
    uint8_t img_id;
    uint8_t row;
    uint8_t col;
    uint16_t z_h;
    uint16_t z_w;
    uint32_t y_len;
    uint32_t z_len;
};
#pragma pack(pop)

// ---------------------------------------------------------
// OBC UART Communication Wrapper
// ---------------------------------------------------------
class OBCInterface {
private:
    int uart_fd;
    std::string device_path;
    std::string rx_buffer;

public:
    OBCInterface(const std::string& dev) : device_path(dev), uart_fd(-1), rx_buffer("") {}

    bool init() {
        uart_fd = open(device_path.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (uart_fd == -1) {
            return false;
        }

        struct termios options;
        tcgetattr(uart_fd, &options);
        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);

        // RAW mode setup
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        options.c_iflag &= ~(IXON | IXOFF | IXANY);

        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = 5; 

        tcsetattr(uart_fd, TCSANOW, &options);
        fcntl(uart_fd, F_SETFL, O_NDELAY); 
        return true;
    }

    void close_port() {
        if (uart_fd >= 0) {
            close(uart_fd);
            uart_fd = -1;
        }
    }

    void send_msg(const std::string& msg) {
        if (uart_fd >= 0) {
            write(uart_fd, msg.c_str(), msg.length());
        }
    }

    std::string read_line() {
        char c;
        while (read(uart_fd, &c, 1) > 0) {
            if (c == '\n') {
                std::string complete_cmd = rx_buffer;
                rx_buffer.clear();
                return complete_cmd;
            } else if (c != '\r') {
                rx_buffer += c;
            }
        }
        return "";
    }
};

// ---------------------------------------------------------
// Core Compression Logic
// ---------------------------------------------------------
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

struct SharedORTResource {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CIC_Payload"};
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

struct CompressionResult {
    bool success;
    int total_patches;
    int success_patches;
    size_t total_bytes;
    std::string output_dir;
    std::string error_msg;
};

CompressionResult process_image(const std::string& tif_path, const std::string& base_bin_dir) {
    CompressionResult result = {false, 0, 0, 0, "", ""};
    
    try {
        const std::string MODEL_ENCODER = "./model/onnx/cic_encoder.onnx";
        const std::string MODEL_HYPER_DECODER = "./model/onnx/cic_hyper_decoder.onnx";
        int ENC_BATCH = 4;
        int HYP_BATCH = ENC_BATCH * 4;

        std::string file_stem = fs::path(tif_path).stem().string();
        result.output_dir = (fs::path(base_bin_dir) / file_stem).string();
        if (!fs::exists(result.output_dir)) {
            fs::create_directories(result.output_dir);
        }

        cv::Mat img = cv::imread(tif_path, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        if (img.empty()) {
            throw std::runtime_error("Cannot read image");
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat img_f; img.convertTo(img_f, CV_32FC3, 1.0/255.0);

        int ps = 256;
        int nr = (img_f.rows + ps - 1) / ps, nc = (img_f.cols + ps - 1) / ps;
        result.total_patches = nr * nc;

        SharedORTResource res(MODEL_ENCODER, MODEL_HYPER_DECODER);
        PureDynamicRansEncoder2Way rans;
        auto m_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<AlignedVector<float>> all_y_raw(result.total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_z_hat(result.total_patches, AlignedVector<float>(32*4*4));
        std::vector<AlignedVector<int32_t>> all_z_sym(result.total_patches, AlignedVector<int32_t>(32*4*4));
        std::vector<AlignedVector<float>> all_scales(result.total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_means(result.total_patches, AlignedVector<float>(192*16*16));

        std::vector<AlignedVector<float>> patches_raw(result.total_patches, AlignedVector<float>(3*ps*ps));
        for (int i = 0; i < result.total_patches; ++i) {
            extract_patch_to_buffer(img_f, patches_raw[i].data(), i / nc, i % nc, ps);
        }

        for (int i = 0; i < result.total_patches; i += ENC_BATCH) {
            int cur_batch = std::min(ENC_BATCH, result.total_patches - i);
            AlignedVector<float> stacked_in(cur_batch * 3 * ps * ps);
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(patches_raw[i+b].begin(), patches_raw[i+b].end(), stacked_in.begin() + b*(3*ps*ps));
            }

            std::vector<int64_t> enc_shape = {cur_batch, 3, ps, ps};
            Ort::Value in_t = Ort::Value::CreateTensor<float>(m_info, stacked_in.data(), stacked_in.size(), enc_shape.data(), enc_shape.size());
            auto outs = res.enc_sess->Run(Ort::RunOptions{nullptr}, res.enc_in, &in_t, 1, res.enc_out, 2);

            float* y_ptr = outs[0].GetTensorMutableData<float>();
            float* z_ptr = outs[1].GetTensorMutableData<float>();
            for (int b = 0; b < cur_batch; ++b) {
                std::copy(y_ptr + b*(192*16*16), y_ptr + (b+1)*(192*16*16), all_y_raw[i+b].begin());
                quantize_z_a53(z_ptr + b*(32*4*4), FIXED_EB_MEDIANS.data(), all_z_sym[i+b].data(), all_z_hat[i+b].data(), 32, 4, 4);
            }
        }

        for (int i = 0; i < result.total_patches; i += HYP_BATCH) {
            int cur_batch = std::min(HYP_BATCH, result.total_patches - i);
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
        }

        for (int i = 0; i < result.total_patches; ++i) {
            try {
                AlignedVector<int32_t> y_sym(192*16*16), y_idx(192*16*16);
                quantize_and_index_y_a53(all_y_raw[i].data(), all_scales[i].data(), all_means[i].data(), 
                                         FIXED_GC_SCALE_TABLE.data(), FIXED_GC_SCALE_TABLE.size(), 
                                         y_sym.data(), y_idx.data(), 192, 16, 16);
                auto y_bits = rans.encode_with_indexes(y_sym.data(), y_idx.data(), 192*16*16, FIXED_GC_CDF, FIXED_GC_LENGTH, FIXED_GC_OFFSET);

                AlignedVector<int32_t> z_idx_map(32*4*4);
                for (size_t k = 0; k < 32*4*4; ++k) z_idx_map[k] = (k / (4*4)) % 32;
                auto z_bits = rans.encode_with_indexes(all_z_sym[i].data(), z_idx_map.data(), 32*4*4, FIXED_EB_CDF, FIXED_EB_LENGTH, FIXED_EB_OFFSET);

                CICPacketHeader header = { {'C', 'I', 'C'}, 1, (uint8_t)(i / nc), (uint8_t)(i % nc), 4, 4, (uint32_t)y_bits.size(), (uint32_t)z_bits.size() };
                
                std::vector<uint8_t> full_payload;
                full_payload.reserve(sizeof(header) + y_bits.size() + z_bits.size() + 4);
                full_payload.insert(full_payload.end(), (uint8_t*)&header, (uint8_t*)&header + sizeof(header));
                full_payload.insert(full_payload.end(), y_bits.begin(), y_bits.end());
                full_payload.insert(full_payload.end(), z_bits.begin(), z_bits.end());
                
                uint32_t crc = crc32(0L, full_payload.data(), full_payload.size());
                full_payload.insert(full_payload.end(), (uint8_t*)&crc, (uint8_t*)&crc + 4);

                std::string fpath = result.output_dir + "/patch_" + std::to_string(i) + ".bin";
                std::ofstream f(fpath, std::ios::binary);
                f.write((char*)full_payload.data(), full_payload.size());
                
                if (f.good()) {
                    result.success_patches++;
                    result.total_bytes += full_payload.size();
                }
                f.close();
            } catch (...) {
                continue;
            }
        }

        result.success = (result.success_patches == result.total_patches);
        return result;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
        return result;
    }
}

// ---------------------------------------------------------
// Main Service Loop
// ---------------------------------------------------------
int main() {
    const std::string UART_DEV = "/dev/ttyPS0"; 
    const std::string BASE_BIN_DIR = "/media/emmc/compressed_data"; 
    
    OBCInterface obc(UART_DEV);
    if (!obc.init()) {
        std::cerr << "CRITICAL: Failed to initialize OBC UART.\n";
        return 1;
    }

    obc.send_msg("SYS_READY\r\n");

    while (true) {
        std::string cmd = obc.read_line();
        if (cmd.empty()) {
            usleep(50000); 
            continue;
        }

        if (cmd.rfind("CMD_COMPRESS", 0) == 0) {
            obc.send_msg("ACK_COMPRESS\r\n");
            
            std::istringstream iss(cmd);
            std::string cmd_name, token, img_path, scp_target;            
            iss >> cmd_name >> token >> img_path >> scp_target;

            if (img_path.empty() || scp_target.empty()) {
                obc.send_msg("ERR_INVALID_ARGS\r\n");
                continue;
            }

            CompressionResult res = process_image(img_path, BASE_BIN_DIR);
            
            if (res.success) {
                std::string scp_cmd = "scp -q -r " + res.output_dir + " " + scp_target;
                int scp_result = std::system(scp_cmd.c_str());

                if (scp_result == 0) {
                    std::ostringstream response;
                    response << "DONE_COMPRESS_AND_TRANSFER " << token << " " << res.output_dir << " " 
                             << res.total_bytes << " " << res.success_patches << "\r\n";
                    obc.send_msg(response.str());
                } else {
                    obc.send_msg("ERR_SCP_FAILED " + token + "\r\n");
                }
            } else {
                obc.send_msg("ERR_COMPRESSION_FAILED " + token + " " + res.error_msg + "\r\n");
            }
        } else if (cmd == "CMD_PING") {
            obc.send_msg("PONG\r\n");
        } else if (cmd == "CMD_EXIT") {
            obc.send_msg("ACK_EXIT_SYSTEM\r\n");
            break; // 跳出 while 迴圈，執行後面的 close_port() 並結束程式
        }
    }

    obc.close_port();
    return 0;
}