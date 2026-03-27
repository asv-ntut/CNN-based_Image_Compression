import cv2
import numpy as np
import argparse
import sys

# ==========================================================
# 1. MS-SSIM 計算工具 (學術標準實作，針對 0.0 ~ 1.0 範圍修正)
# ==========================================================
def calculate_ssim_components(img1, img2):
    L = 1.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)

    # 11x11 Gaussian blur, sigma=1.5
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(I1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2

    # 計算亮度項 (L) 與 對比結構項 (CS)
    t_L = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    t_CS = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    # 取全圖平均
    l = np.mean(t_L)
    cs = np.mean(t_CS)

    return l, cs

def compute_ms_ssim(img1, img2):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    msssim = 1.0
    im1 = img1.copy()
    im2 = img2.copy()

    for i in range(5):
        l, cs = calculate_ssim_components(im1, im2)
        
        if i < 4:
            msssim *= max(0.0, cs) ** weights[i]
            im1 = cv2.resize(im1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            im2 = cv2.resize(im2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        else:
            msssim *= max(0.0, l * cs) ** weights[i]

    return msssim

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

# ==========================================================
# 2. 主程式
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment (PSNR & MS-SSIM)")
    parser.add_argument("original", help="Path to the original image (e.g., .tif)")
    parser.add_argument("reconstructed", help="Path to the reconstructed image (e.g., .png)")
    args = parser.parse_args()

    # 讀取影像 (支援 16-bit TIF 與一般 8-bit PNG)
    img1 = cv2.imread(args.original, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img2 = cv2.imread(args.reconstructed, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    if img1 is None:
        print(f"Error: Cannot read original image: {args.original}")
        sys.exit(1)
    if img2 is None:
        print(f"Error: Cannot read reconstructed image: {args.reconstructed}")
        sys.exit(1)

    if img1.shape != img2.shape:
        print(f"Error: Image dimensions do not match. Original: {img1.shape}, Reconstructed: {img2.shape}")
        sys.exit(1)

    # 色彩空間轉換 BGR -> RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 歸一化至 0.0 ~ 1.0 (Float32)
    # 根據影像深度自動判斷除數 (8-bit 或 16-bit)
    max_val_1 = 65535.0 if img1.dtype == np.uint16 else 255.0
    max_val_2 = 65535.0 if img2.dtype == np.uint16 else 255.0
    
    img1_f = img1.astype(np.float32) / max_val_1
    img2_f = img2.astype(np.float32) / max_val_2

    print("Calculating metrics...")

    psnr_val = compute_psnr(img1_f, img2_f)
    msssim_val = compute_ms_ssim(img1_f, img2_f)

    print("=========================================")
    print(" Image Quality Assessment Report")
    print("=========================================")
    print(f"{'Original':<25}: {args.original}")
    print(f"{'Reconstructed':<25}: {args.reconstructed}")
    print("-" * 41)
    print(f"{'PSNR':<25}: {psnr_val:.4f} dB")
    print(f"{'MS-SSIM':<25}: {msssim_val:.6f}")
    print("=========================================")

if __name__ == "__main__":
    main()