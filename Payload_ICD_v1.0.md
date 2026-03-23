# TIC 衛星影像壓縮酬載 - 系統整合與介面控制文件 (ICD)

## 1. 系統架構總覽 (System Architecture)
本系統運行於 A53 開發板

* **Dispatcher (主控接口)**：專職監聽 OBC UART 訊號，並進行任務分發。
* **AI Worker (壓縮核心)**：常駐於背景。預先載入模型，收到指令後可達成零延遲啟動推論。

---

## 2. 介面規格 (Interface Specifications)

### 2.1 OBC 指令接收端 (UART)
* **硬體介面**：`/dev/ttyUSB0` *(待與 OBC 團隊確認最終 Device Node)*
* **通訊參數**：Baud Rate 115200, 8N1
* **指令格式**：ASCII 純文字，以換行符號 `\n` 或 `\r\n` 結尾。
* **指令語法**：`START <MODE_ID> <ABSOLUTE_FILE_PATH>\n`
    * `MODE_ID`: 任務模式代碼 (1=Bypass, 2=AI 壓縮, 3=Residual+AI壓縮)
    * `ABSOLUTE_FILE_PATH`: 待處理影像之絕對路徑 (例: `/data/images/capture_001.tif`)
* **傳輸範例**：`START 2 /data/images/capture_001.tif\n`

### 2.2 通訊團隊傳輸端 (TCP/IP)
* **傳輸協定**：TCP Client (本系統主動向通訊團隊的 Server 發起連線)
* **目標 IP 位址**：`192.168.70.1`
* **目標 Port**：`5000` *(待與通訊團隊確認)*
* **傳輸行為**：
    * **Mode 1**：將原始 `.tif` 檔案轉為 Byte Stream 直接發送。
    * **Mode 2 & 3**：將壓縮產生的多個 Patch 封包 (`.bin`) 依序發送。

### 2.3 壓縮封包格式 (TIC Payload Format)
Mode 2 與 Mode 3 輸出的 `.bin` 檔案採用 Little-Endian (`<`) 編碼，完整的封包結構包含 Header (18 Bytes)、Payload (壓縮資料) 與 Footer (4 Bytes CRC32)。

**Header 定義 (`struct.pack('<3sBBBHHII')`)**

| 欄位名稱 | 型別 (C Type) | 大小 (Bytes) | 說明 |
| :--- | :--- | :--- | :--- |
| **Magic Number** | `char[3]` | 3 | 固定為 `TIC`，用於封包辨識 |
| **Image ID** | `uint8_t` | 1 | 影像編號 |
| **Row Index** | `uint8_t` | 1 | 該 Patch 於原圖的列索引 (Row) |
| **Col Index** | `uint8_t` | 1 | 該 Patch 於原圖的行索引 (Col) |
| **Z Shape 0** | `uint16_t` | 2 | 潛在空間維度 0 (Height) |
| **Z Shape 1** | `uint16_t` | 2 | 潛在空間維度 1 (Width) |
| **Y Length** | `uint32_t` | 4 | Y Bitstream 的長度 (Bytes) |
| **Z Length** | `uint32_t` | 4 | Z Bitstream 的長度 (Bytes) |
| *(Header 總計)* | | *18* | |

**Payload & Footer 定義**

| 欄位名稱 | 大小 (Bytes) | 說明 |
| :--- | :--- | :--- |
| **Y Bitstream** | `Y Length` | 動態 ANS 壓縮後的主要資料 |
| **Z Bitstream** | `Z Length` | 動態 ANS 壓縮後的 Hyper 資料 |
| **Footer (CRC32)** | 4 | `uint32_t`，針對 `Header + Payload` 計算的 CRC32 校驗碼 |

---

## 3. 任務模式說明 (Operation Modes)
1. **Mode 1 (Bypass 模式)**：不進行壓縮。Dispatcher 收到指令後，直接讀取硬碟原始影像並透過 TCP 傳送給通訊團隊。
2. **Mode 2 (AI 壓縮模式)**：Dispatcher 喚醒背景 AI Worker。Worker 讀取影像進行 Vectorized Patch Extraction 與 ONNX 推論，產生壓縮封包後發送。
3. **Mode 3 (進階 AI 壓縮模式)**：啟動前置 Residual 運算，處理完畢後接續 Mode 2 的 AI 壓縮流程。

---

## 4. 首次環境建置 (First-Time Setup)
若為首次於 A53 開發板上操作本系統，請整測人員務必先完成以下環境建置步驟，以建立虛擬環境、編譯 C++ 擴充模組，並安裝專案相依套件。

### 步驟一：專案初始化與環境建置
你可以透過手動逐步輸入指令完成，或使用專案內建的 `setup.sh` 進行一鍵安裝。

**【方法 A：一鍵執行腳本】(推薦)**
```bash
cd ~/CNN-based_Image_compression
chmod +x setup.sh
source ./setup.sh  # 或使用簡短寫法： . ./setup.sh
```
### 步驟二 編譯 C 語言主控接口 (Dispatcher)
AI Worker 依賴 C Dispatcher 來接收硬體 UART 訊號，首次使用前需將其編譯為可執行檔：
```bash
# 確保位於專案目錄下
gcc dispatcher.c -o dispatcher
```
## 5. 系統啟動與執行腳本 (System Startup)
環境建置完成後，後續每次開機只需透過啟動腳本執行即可。請建立 start_payload.sh 腳本，並強烈建議將其設定為開機自動執行。

給予執行腳本權限：
```bash
chmod +x start_payload.sh
```
一鍵啟動系統
```bash
./start_payload.sh
```
注意： 啟動前台 C Dispatcher 可能需要 sudo 權限才能讀取 UART Port (/dev/ttyUSB0)。


OBC執行
```bash
echo "START <任務模式 (Mode ID)> <image_path>" > /dev/ttyUSB0
echo "START 2 ../../Taiwan/Kaohsiung_RGB_Normalized_tile_r0_c0.tif" > /dev/ttyUSB0
```
