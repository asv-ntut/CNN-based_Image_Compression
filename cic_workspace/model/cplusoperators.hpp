#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Dense>

namespace cic_edge {

    template <typename T>
    using AlignedVector = std::vector<T, std::allocator<T>>;

    // ==========================================================
    // 1. 高效 Conv2D (無 ReLU) - Eigen + 指標化 Im2Col
    // ==========================================================
    template<int K, int S>
    inline void conv2d_a53(
        const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
        float* __restrict__ output, float* __restrict__ col_workspace, 
        int C_in, int C_out, int H_in, int W_in, int pad) 
    {
        int H_out = (H_in + 2 * pad - K) / S + 1;
        int W_out = (W_in + 2 * pad - K) / S + 1;
        int HW_out = H_out * W_out;
        int K2C = C_in * K * K;

        // Step 1: 🚀 優化版 Im2Col (指標連續遞增，消除乘法尋址開銷)
        std::fill(col_workspace, col_workspace + (K2C * HW_out), 0.0f);
        float* col_ptr = col_workspace;
        for (int cin = 0; cin < C_in; ++cin) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    for (int h = 0; h < H_out; ++h) {
                        int h_in = h * S - pad + kh;
                        // 邊界外直接跳過，因為已經 fill 過 0.0f
                        if (h_in < 0 || h_in >= H_in) {
                            col_ptr += W_out; 
                            continue;
                        }
                        const float* in_row_ptr = input + (cin * H_in * W_in) + (h_in * W_in);
                        for (int w = 0; w < W_out; ++w) {
                            int w_in = w * S - pad + kw;
                            if (w_in >= 0 && w_in < W_in) {
                                *col_ptr = in_row_ptr[w_in];
                            }
                            col_ptr++;
                        }
                    }
                }
            }
        }

        // Step 2: Eigen 極速矩陣乘法 (GEMM)
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        
        Eigen::Map<const MatrixType> W(weight, C_out, K2C);
        Eigen::Map<const MatrixType> X(col_workspace, K2C, HW_out);
        Eigen::Map<MatrixType> Y(output, C_out, HW_out);
        Eigen::Map<const Eigen::VectorXf> B(bias, C_out);

        Y.noalias() = W * X;
        Y.colwise() += B; 
    }

    // ==========================================================
    // 2. 高效 Conv2D + ReLU 融合 - Eigen + 指標化 Im2Col
    // ==========================================================
    template<int K, int S>
    inline void conv2d_relu_a53(
        const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
        float* __restrict__ output, float* __restrict__ col_workspace,
        int C_in, int C_out, int H_in, int W_in, int pad) 
    {
        int H_out = (H_in + 2 * pad - K) / S + 1;
        int W_out = (W_in + 2 * pad - K) / S + 1;
        int HW_out = H_out * W_out;
        int K2C = C_in * K * K;

        std::fill(col_workspace, col_workspace + (K2C * HW_out), 0.0f);
        float* col_ptr = col_workspace;
        for (int cin = 0; cin < C_in; ++cin) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    for (int h = 0; h < H_out; ++h) {
                        int h_in = h * S - pad + kh;
                        if (h_in < 0 || h_in >= H_in) {
                            col_ptr += W_out; 
                            continue;
                        }
                        const float* in_row_ptr = input + (cin * H_in * W_in) + (h_in * W_in);
                        for (int w = 0; w < W_out; ++w) {
                            int w_in = w * S - pad + kw;
                            if (w_in >= 0 && w_in < W_in) {
                                *col_ptr = in_row_ptr[w_in];
                            }
                            col_ptr++;
                        }
                    }
                }
            }
        }

        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        
        Eigen::Map<const MatrixType> W(weight, C_out, K2C);
        Eigen::Map<const MatrixType> X(col_workspace, K2C, HW_out);
        Eigen::Map<MatrixType> Y(output, C_out, HW_out);
        Eigen::Map<const Eigen::VectorXf> B(bias, C_out);

        Y.noalias() = W * X;
        Y.colwise() += B;
        Y = Y.cwiseMax(0.0f); 
    }

    // ==========================================================
    // 3. Deconv 融合 PixelShuffle + ReLU - Eigen + 指標化 Im2Col
    // ==========================================================
    template<int K, int UpscaleFactor>
    inline void deconv_relu_fused_pixelshuffle_a53(
        const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
        float* __restrict__ output, int C_in, int C_internal, int H_in, int W_in, int pad) 
    {
        int HW_in = H_in * W_in;
        int K2C = C_in * K * K;
        
        std::vector<float> local_col_workspace(K2C * HW_in, 0.0f);
        
        // Step 1: 優化版 Im2Col (針對 stride=1)
        float* col_ptr = local_col_workspace.data();
        for (int cin = 0; cin < C_in; ++cin) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    for (int h = 0; h < H_in; ++h) {
                        int h_in = h - pad + kh;
                        if (h_in < 0 || h_in >= H_in) {
                            col_ptr += W_in;
                            continue;
                        }
                        const float* in_row_ptr = input + (cin * H_in * W_in) + (h_in * W_in);
                        for (int w = 0; w < W_in; ++w) {
                            int w_in = w - pad + kw;
                            if (w_in >= 0 && w_in < W_in) {
                                *col_ptr = in_row_ptr[w_in];
                            }
                            col_ptr++;
                        }
                    }
                }
            }
        }

        // Step 2: Eigen 矩陣乘法 + Bias + ReLU
        std::vector<float> temp_conv_result(C_internal * HW_in, 0.0f);
        
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        
        Eigen::Map<const MatrixType> W(weight, C_internal, K2C);
        Eigen::Map<const MatrixType> X(local_col_workspace.data(), K2C, HW_in);
        Eigen::Map<MatrixType> Y(temp_conv_result.data(), C_internal, HW_in);
        Eigen::Map<const Eigen::VectorXf> B(bias, C_internal);

        Y.noalias() = W * X;
        Y.colwise() += B;
        Y = Y.cwiseMax(0.0f);

        // Step 3: PixelShuffle 重排
        int H_out_final = H_in * UpscaleFactor;
        int W_out_final = W_in * UpscaleFactor;
        
        for (int c_int = 0; c_int < C_internal; ++c_int) {
            int c_out_final = c_int / (UpscaleFactor * UpscaleFactor);
            int sub_h       = (c_int % (UpscaleFactor * UpscaleFactor)) / UpscaleFactor;
            int sub_w       = c_int % UpscaleFactor;
            int c_int_offset = c_int * HW_in;

            for (int h = 0; h < H_in; ++h) {
                for (int w = 0; w < W_in; ++w) {
                    int h_out_final_idx = h * UpscaleFactor + sub_h;
                    int w_out_final_idx = w * UpscaleFactor + sub_w;
                    float val = temp_conv_result[c_int_offset + (h * W_in + w)];
                    output[(c_out_final * H_out_final * W_out_final) + (h_out_final_idx * W_out_final) + w_out_final_idx] = val;
                }
            }
        }
    }

    // ==========================================================
    // 4. GDN / IGDN 算子 - 🌟 32 Bytes 記憶體對齊 (NEON 極限優化)
    // ==========================================================
    template<bool Inverse>
    inline void gdn_a53(
        const float* __restrict__ input, const float* __restrict__ gamma, const float* __restrict__ bias_beta,
        float* __restrict__ output, int C, int H, int W)
    {
        int HW = H * W;
        // alignas(32) 確保陣列在記憶體中的起始位址是 32 的倍數
        alignas(32) float px[256];
        alignas(32) float px_sq[256];

        for (int hw_offset = 0; hw_offset < HW; ++hw_offset) {
            for (int c = 0; c < C; ++c) {
                float val = input[c * HW + hw_offset];
                px[c] = val;
                px_sq[c] = val * val;
            }

            for (int cout = 0; cout < C; ++cout) {
                float norm = bias_beta[cout];
                int gamma_offset = cout * C;
                
                #pragma omp simd
                for (int cin = 0; cin < C; ++cin) {
                    norm += gamma[gamma_offset + cin] * px_sq[cin];
                }

                if constexpr (Inverse) {
                    output[cout * HW + hw_offset] = px[cout] * std::sqrt(norm);
                } else {
                    output[cout * HW + hw_offset] = px[cout] / std::sqrt(norm);
                }
            }
        }
    }

    // ==========================================================
    // 5. 熵編碼量化算子 (z 與 y) - 二元搜尋維持不變
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
    // 6. 熵解碼反量化算子 (z 與 y)
    // ==========================================================
    inline void dequantize_z_a53(
        const int32_t* __restrict__ symbols, const float* __restrict__ medians, 
        float* __restrict__ z_hat, int C, int H, int W) 
    {
        int HW = H * W;
        for (int c = 0; c < C; ++c) {
            float med = medians[c];
            int c_offset = c * HW;
            for (int hw = 0; hw < HW; ++hw) {
                int idx = c_offset + hw;
                z_hat[idx] = static_cast<float>(symbols[idx]) + med;
            }
        }
    }

    inline void compute_y_indexes_a53(
        const float* __restrict__ scales, const float* __restrict__ means,
        const float* __restrict__ scale_table, int num_scales, 
        int32_t* __restrict__ indexes, int C, int H, int W)
    {
        int total = C * H * W;
        // const float EPS = 1e-6f;
        for (int i = 0; i < total; ++i) {
            // float target_scale = scales[i]+EPS;
            float target_scale = scales[i];
            //float target_scale = std::round(scales[i] * 100000.0f) / 100000.0f;
            if (target_scale <= scale_table[0]) { indexes[i] = 0; continue; }
            if (target_scale >= scale_table[num_scales - 1]) { indexes[i] = num_scales - 1; continue; }

            const float* it = std::lower_bound(scale_table, scale_table + num_scales, target_scale);
            float diff_right = *it - target_scale;
            float diff_left = target_scale - *(it - 1);
            
            indexes[i] = (diff_right < diff_left) ? std::distance(scale_table, it) : std::distance(scale_table, it - 1);
        }
    }

    inline void dequantize_y_a53(
        const int32_t* __restrict__ symbols, const float* __restrict__ means,
        float* __restrict__ y_hat, int C, int H, int W)
    {
        int total = C * H * W;
        for (int i = 0; i < total; ++i) {
            // 將整數符號加上分佈的平均值，還原回浮點數特徵
            y_hat[i] = static_cast<float>(symbols[i]) + means[i];
        }
    }

    inline void quantize_and_index_y_a53(
        const float* __restrict__ y, const float* __restrict__ scales, const float* __restrict__ means,
        const float* __restrict__ scale_table, int num_scales, 
        int32_t* __restrict__ symbols, int32_t* __restrict__ indexes, int C, int H, int W)
    {
        int total = C * H * W;
        const float EPS = 1e-6f;
        for (int i = 0; i < total; ++i) {
            symbols[i] = static_cast<int32_t>(std::round(y[i] - means[i]));
            // float target_scale = scales[i]+EPS;
            float target_scale = scales[i];
            // float target_scale = std::round(scales[i] * 100000.0f) / 100000.0f;
            
            if (target_scale <= scale_table[0]) { indexes[i] = 0; continue; }
            if (target_scale >= scale_table[num_scales - 1]) { indexes[i] = num_scales - 1; continue; }

            const float* it = std::lower_bound(scale_table, scale_table + num_scales, target_scale);
            float diff_right = *it - target_scale;
            float diff_left = target_scale - *(it - 1);
            
            indexes[i] = (diff_right < diff_left) ? std::distance(scale_table, it) : std::distance(scale_table, it - 1);
        }
    }

} // namespace cic_edge