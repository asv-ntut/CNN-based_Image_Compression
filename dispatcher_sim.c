#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>

// --- 系統設定參數 (純軟體模擬版) ---
#define UART_PORT "/tmp/ttyVIRTUAL"  // 指向剛建立的 Named Pipe
#define AI_WORKER_IP "127.0.0.1"
#define AI_WORKER_PORT 6000
#define COMM_IP "127.0.0.1"          // 測試時也指向本機的 Mock Server
#define COMM_PORT 5000
#define BUFFER_SIZE 4096

// --- 模擬 UART 初始化 ---
int init_uart_sim(const char* portname) {
    // 使用 O_RDWR 避免在沒有寫入者時讀取端直接非阻塞或收到 EOF
    int fd = open(portname, O_RDWR);
    if (fd < 0) {
        printf("錯誤: 無法開啟虛擬 UART 埠 %s: %s\n", portname, strerror(errno));
        return -1;
    }
    printf("✅ 成功開啟虛擬 UART: %s\n", portname);
    return fd;
}

// --- Mode 1: Bypass ---
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
        fclose(fp); return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(COMM_PORT);
    inet_pton(AF_INET, COMM_IP, &serv_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("錯誤: 無法連線至通訊模擬器 %s:%d\n", COMM_IP, COMM_PORT);
        close(sock); fclose(fp); return;
    }

    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fp)) > 0) {
        send(sock, buffer, bytes_read, 0);
    }
    printf("[Mode 1] 原始檔案傳輸完成！\n");
    fclose(fp); close(sock);
}

// --- Mode 2 & 3: 觸發 AI Worker ---
void trigger_ai_worker(int mode, const char* file_path) {
    printf("[Mode %d] 呼叫背景 AI Worker 處理: %s\n", mode, file_path);
    int sock = 0;
    struct sockaddr_in serv_addr;
    char payload[512];

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return;

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(AI_WORKER_PORT);
    inet_pton(AF_INET, AI_WORKER_IP, &serv_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("連線失敗：AI Worker 似乎沒有啟動 (Port %d)。\n", AI_WORKER_PORT);
        close(sock); return;
    }

    snprintf(payload, sizeof(payload), "{\"mode\": %d, \"file_path\": \"%s\"}", mode, file_path);
    send(sock, payload, strlen(payload), 0);
    printf("已發送 IPC 任務指令給 AI Worker。\n");
    close(sock);
}

int main() {
    printf("🛰️  C Dispatcher (模擬模式) 啟動，準備監聽 %s...\n", UART_PORT);
    
    int uart_fd = init_uart_sim(UART_PORT);
    if (uart_fd < 0) return 1;

    char read_buf[256];
    char command_line[256];
    int cmd_idx = 0;

    while (1) {
        int n = read(uart_fd, read_buf, sizeof(read_buf));
        if (n > 0) {
            for (int i = 0; i < n; i++) {
                char c = read_buf[i];
                if (c == '\n' || c == '\r') {
                    if (cmd_idx > 0) {
                        command_line[cmd_idx] = '\0';
                        printf("\n[收到虛擬 UART 指令] %s\n", command_line);

                        if (strncmp(command_line, "START ", 6) == 0) {
                            int mode; char file_path[256];
                            if (sscanf(command_line + 6, "%d %255s", &mode, file_path) == 2) {
                                if (mode == 1) handle_mode_1_bypass(file_path);
                                else if (mode == 2 || mode == 3) trigger_ai_worker(mode, file_path);
                            }
                        }
                        cmd_idx = 0;
                    }
                } else {
                    if (cmd_idx < sizeof(command_line) - 1) command_line[cmd_idx++] = c;
                }
            }
        }
    }
    close(uart_fd);
    return 0;
}