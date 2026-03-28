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
#include <zlib.h> 

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

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

// [修正 1] 同步將訊息印到螢幕上，並移除會讓 FIFO 報錯的 tcdrain
void send_uart_msg(int fd, const std::string& msg) {
    std::cout << msg; 
    if (fd >= 0) {
        write(fd, msg.c_str(), msg.length());
    }
}

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

bool process_image(int uart_fd, const std::string& tif_path, const std::string& base_bin_dir, std::string& out_final_dir, bool show_stats) {
    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        size_t total_compressed_bytes = 0;

        const std::string MODEL_ENCODER = "./model/onnx/cic_encoder.onnx";
        const std::string MODEL_HYPER_DECODER = "./model/onnx/cic_hyper_decoder.onnx";
        int ENC_BATCH = 4;
        int HYP_BATCH = ENC_BATCH * 4;

        std::string file_stem = fs::path(tif_path).stem().string();
        out_final_dir = (fs::path(base_bin_dir) / file_stem).string();
        if (!fs::exists(out_final_dir)) fs::create_directories(out_final_dir);

        send_uart_msg(uart_fd, "STATUS: LOADING_IMAGE\n");
        cv::Mat img = cv::imread(tif_path, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        if (img.empty()) throw std::runtime_error("Cannot read image: " + tif_path);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat img_f; img.convertTo(img_f, CV_32FC3, 1.0/255.0);

        int ps = 256;
        int nr = (img_f.rows + ps - 1) / ps, nc = (img_f.cols + ps - 1) / ps;
        int total_patches = nr * nc;

        send_uart_msg(uart_fd, "STATUS: INIT_MODELS\n");
        SharedORTResource res(MODEL_ENCODER, MODEL_HYPER_DECODER);
        PureDynamicRansEncoder2Way rans;
        auto m_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<AlignedVector<float>> all_y_raw(total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_z_hat(total_patches, AlignedVector<float>(32*4*4));
        std::vector<AlignedVector<int32_t>> all_z_sym(total_patches, AlignedVector<int32_t>(32*4*4));
        std::vector<AlignedVector<float>> all_scales(total_patches, AlignedVector<float>(192*16*16));
        std::vector<AlignedVector<float>> all_means(total_patches, AlignedVector<float>(192*16*16));

        std::vector<AlignedVector<float>> patches_raw(total_patches, AlignedVector<float>(3*ps*ps));
        for (int i = 0; i < total_patches; ++i) {
            extract_patch_to_buffer(img_f, patches_raw[i].data(), i / nc, i % nc, ps);
        }

        send_uart_msg(uart_fd, "STATUS: ENCODING\n");
        for (int i = 0; i < total_patches; i += ENC_BATCH) {
            int cur_batch = std::min(ENC_BATCH, total_patches - i);
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

        send_uart_msg(uart_fd, "STATUS: HYPER_DECODING\n");
        for (int i = 0; i < total_patches; i += HYP_BATCH) {
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
        }

        send_uart_msg(uart_fd, "STATUS: ENTROPY_AND_SAVE\n");
        int successful_patches = 0;
        int failed_patches = 0;

        for (int i = 0; i < total_patches; ++i) {
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

                std::string fpath = out_final_dir + "/patch_" + std::to_string(i) + ".bin";
                std::ofstream f(fpath, std::ios::binary);
                f.write((char*)full_payload.data(), full_payload.size());
                
                if (f.good()) {
                    successful_patches++;
                    total_compressed_bytes += full_payload.size();
                } else {
                    failed_patches++;
                }
                f.close();
            } catch (const std::exception& e) {
                failed_patches++;
            }
        }

        std::ostringstream p_ss;
        p_ss << "STATUS: PATCH_RESULT TOTAL=" << total_patches 
             << " SUCCESS=" << successful_patches 
             << " FAIL=" << failed_patches << "\n";
        send_uart_msg(uart_fd, p_ss.str());

        if (show_stats) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double>(t_end - t_start).count();
            double total_pixels = (double)img.cols * img.rows;
            double avg_bpp = ((double)total_compressed_bytes * 8.0) / total_pixels;
            
            std::ostringstream ss;
            ss << "STATS: TIME=" << std::fixed << std::setprecision(4) << total_time 
               << " BPP=" << std::fixed << std::setprecision(4) << avg_bpp << "\n";
            send_uart_msg(uart_fd, ss.str());
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Compression Error: " << e.what() << "\n";
        send_uart_msg(uart_fd, std::string("ERROR: ") + e.what() + "\n");
        return false;
    }
}

int init_uart(const char* device) {
    int fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("Unable to open UART device");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    options.c_cflag |= (CLOCAL | CREAD);

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10; 

    tcsetattr(fd, TCSANOW, &options);
    fcntl(fd, F_SETFL, 0); 
    return fd;
}

int main() {
    const char* UART_DEV = "/dev/ttyPS0";
    const std::string BASE_BIN_DIR = "./compressed_bins";
    
    int uart_fd = init_uart(UART_DEV);
    if (uart_fd < 0) {
        std::cerr << "Failed to initialize UART.\n";
        return 1;
    }

    std::cout << "A53 Compression Service Started. Listening on " << UART_DEV << " at 115200 8N1...\n";
    send_uart_msg(uart_fd, "READY\n");

    char rx_buffer[256];
    std::string command_buffer = "";

    while (true) {
        int bytes_read = read(uart_fd, rx_buffer, sizeof(rx_buffer) - 1);
        if (bytes_read > 0) {
            rx_buffer[bytes_read] = '\0';
            command_buffer += rx_buffer;

            size_t pos;
            // [修正 2] 將 if 改成 while,確保一次 read 的多個指令都能被依序解析
            while ((pos = command_buffer.find('\n')) != std::string::npos) {
                std::string cmd_line = command_buffer.substr(0, pos);
                command_buffer.erase(0, pos + 1);

                if (cmd_line.rfind("CMD:START", 0) == 0) {
                    send_uart_msg(uart_fd, "ACK:START\n");
                    
                    std::istringstream iss(cmd_line);
                    std::string token, img_path, scp_target;
                    int show_stats = 0; 
                    
                    iss >> token >> img_path >> scp_target;
                    
                    if (iss >> show_stats) {}

                    if (img_path.empty() || scp_target.empty()) {
                        send_uart_msg(uart_fd, "ERROR: INVALID_PARAMETERS\n");
                        continue;
                    }

                    std::string final_output_dir;
                    bool success = process_image(uart_fd, img_path, BASE_BIN_DIR, final_output_dir, show_stats > 0);
                    
                    if (success) {
                        send_uart_msg(uart_fd, "STATUS: TRANSFERRING_SCP\n");
                        
                        std::string scp_cmd = "scp -r " + final_output_dir + " " + scp_target;
                        std::cout << "Executing: " << scp_cmd << "\n";
                        int scp_result = std::system(scp_cmd.c_str());

                        if (scp_result == 0) {
                            send_uart_msg(uart_fd, "DONE: SUCCESS\n");
                        } else {
                            send_uart_msg(uart_fd, "ERROR: SCP_FAILED\n");
                        }
                    } else {
                        send_uart_msg(uart_fd, "ERROR: COMPRESSION_FAILED\n");
                    }
                }
            }
        }
        usleep(10000); 
    }

    close(uart_fd);
    return 0;
}
