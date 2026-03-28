# test: 虛擬管線 sim_uart
```bash
sed -i 's|/dev/ttyPS0|/tmp/sim_uart|g' compress.cpp
chmod +x build_compress.sh
./build_compress.sh
```
# test: 清理環境與重建虛擬通道
```bash
killall -9 compress cat
rm -f /tmp/sim_uart
mkfifo /tmp/sim_uart
```
# test: OBC 下指令
```bash
echo -e "CMD:START ./test.tif root@127.0.0.1:/tmp/ 1\n" > /tmp/sim_uart
```
# test: 在背景執行compress應用程式
```bash
./compress
```

# 使用手冊 : 編譯前確認UART以調整為目標接口&執行編譯
```bash
sed -i 's|/tmp/sim_uart|/dev/ttyPS0|g' compress.cpp
chmod +x build_compress.sh
./build_compress.sh
```

# 步驟1: 在背景執行compress應用程式
```bash
./compress
```
# 步驟2: 在OBC 下指令
```bash
echo -e "CMD:START ./test.tif root@127.0.0.1:/tmp/ 1\n" > /dev/ttyPS0
```
# 指令詳細內容
## CMD:START : 觸發關鍵字 (Token)，告訴主程式準備開工。
## ./test.tif : 指定要載入壓縮的目標圖片路徑。
## root@127.0.0.1:/tmp/ : 指定 SCP 傳輸的目的地連線資訊與存放路徑。
## 1 : 統計旗標 (Stats flag)，告訴程式壓縮完成後要印出 TIME 與 BPP 統計數據（對應程式中的 show_stats 變數）。
## > (標準輸出重導向 Redirection)
## /dev/ttyPS0 (目標設備 FIFO)



# 編譯decompress
```bash
chmod +x build_decompress.sh
sed -i 's/\r//' build_decompress.sh
./build_decompress.sh
```
# 執行decompress應用程式(MacOS)
```bash
./decompress ./test.tif ./compressed_bins
```