#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>

// --- 系統設定參數 ---
#define UART_PORT "/dev/ttyUSB0"
#define BAUD_RATE B115200
#define AI_WORKER_IP "127.0.0.1"
#define AI_WORKER_PORT 6000
#define COMM_IP "192.168.70.1"
#define COMM_PORT 5000
#define BUFFER_SIZE 4096

// --- 設定 UART (8N1, 阻塞模式) ---
int init_uart(const char* portname) {
    int fd = open(portname, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        printf("錯誤: 無法開啟 UART 埠 %s: %s\n", portname, strerror(errno));
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        printf("錯誤: tcgetattr 失敗: %s\n", strerror(errno));
        return -1;
    }

    cfsetospeed(&tty, BAUD_RATE);
    cfsetispeed(&tty, BAUD_RATE);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit characters
    tty.c_cflag |= (CLOCAL | CREAD);            // ignore modem controls, enable reading
    tty.c_cflag &= ~(PARENB | PARODD);          // no parity bit
    tty.c_cflag &= ~CSTOPB;                     // only need 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                    // no hardware flowcontrol

    // 設定為非規範模式 (Non-canonical mode)
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    tty.c_oflag &= ~OPOST;

    // 關鍵：設定阻塞式讀取，收到至少 1 個 byte 才喚醒，不浪費 CPU
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        printf("錯誤: tcsetattr 失敗: %s\n", strerror(errno));
        return -1;
    }
    return fd;
}

// --- Mode 1: 將原始檔案直接丟給通訊團隊 ---
void handle_mode_1_bypass(const char* file_path) {
    printf("[Mode 1] 準備傳輸原始檔案: %s\n", file_path);
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE];

    FILE *fp = fopen(file_path, "rb");
    if (fp == NULL) {
        printf("錯誤: 找不到要傳輸的檔案 %s\n", file_path);
        return;
    }

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("錯誤: 無法建立 Socket\n");
        fclose(fp);
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(COMM_PORT);
    if (inet_pton(AF_INET, COMM_IP, &serv_addr.sin_addr) <= 0) {
        printf("錯誤: 無效的 IP 位址\n");
        close(sock);
        fclose(fp);
        return;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("錯誤: 無法連線至通訊團隊 %s:%d\n", COMM_IP, COMM_PORT);
        close(sock);
        fclose(fp);
        return;
    }

    // 讀取檔案並傳輸
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        send(sock, buffer, bytes_read, 0);
    }

    printf("[Mode 1] 原始檔案傳輸完成！\n");
    fclose(fp);
    close(sock);
}

// --- Mode 2 & 3: 觸發本機 Python AI Worker ---
void trigger_ai_worker(int mode, const char* file_path) {
    printf("[Mode %d] 呼叫背景 AI Worker 處理: %s\n", mode, file_path);
    int sock = 0;
    struct sockaddr_in serv_addr;
    char payload[512];

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("錯誤: 無法建立 Socket\n");
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(AI_WORKER_PORT);
    if (inet_pton(AF_INET, AI_WORKER_IP, &serv_addr.sin_addr) <= 0) {
        printf("錯誤: 無效的 IP 位址\n");
        close(sock);
        return;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("連線失敗:AI Worker 似乎沒有啟動，請確認 compress.py 是否在背景運行。\n");
        close(sock);
        return;
    }

    // 組裝 JSON 格式字串傳給 Python
    snprintf(payload, sizeof(payload), "{\"mode\": %d, \"file_path\": \"%s\"}", mode, file_path);
    send(sock, payload, strlen(payload), 0);
    printf("已發送 IPC 任務指令給 AI Worker。\n");

    close(sock);
}

// --- 主程式 ---
int main() {
    printf("🛰️  C Dispatcher 啟動，準備監聽 UART (%s)...\n", UART_PORT);
    
    int uart_fd = init_uart(UART_PORT);
    if (uart_fd < 0) {
        return 1; // 初始化失敗直接退出
    }

    char read_buf[256];
    char command_line[256];
    int cmd_idx = 0;

    // 進入無窮迴圈監聽
    while (1) {
        int n = read(uart_fd, read_buf, sizeof(read_buf));
        
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                char c = read_buf[i];
                
                // 遇到換行符號代表收完一行完整指令
                if (c == '\n' || c == '\r') {
                    if (cmd_idx > 0) {
                        command_line[cmd_idx] = '\0'; // 加上字串結束字元
                        printf("\n[UART 收到指令] %s\n", command_line);

                        // 簡易字串解析 (假設格式: "START <MODE> <FILE_PATH>")
                        if (strncmp(command_line, "START ", 6) == 0) {
                            int mode;
                            char file_path[256];
                            // 擷取 Mode 與 File Path
                            if (sscanf(command_line + 6, "%d %255s", &mode, file_path) == 2) {
                                if (mode == 1) {
                                    handle_mode_1_bypass(file_path);
                                } else if (mode == 2 || mode == 3) {
                                    trigger_ai_worker(mode, file_path);
                                } else {
                                    printf("警告: 未知的模式代碼 %d\n", mode);
                                }
                            } else {
                                printf("警告: 指令格式解析失敗，請確認是否為 START <MODE> <FILE_PATH>\n");
                            }
                        }
                        cmd_idx = 0; // 重置 buffer
                    }
                } else {
                    // 將字元存入 buffer
                    if (cmd_idx < sizeof(command_line) - 1) {
                        command_line[cmd_idx++] = c;
                    }
                }
            }
        }
    }

    close(uart_fd);
    return 0;
}