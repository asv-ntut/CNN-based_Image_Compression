import socket

def start_mock_server():
    HOST = '127.0.0.1'  # 測試時改聽本機
    PORT = 5000
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"🎧 [Mock Server] 通訊團隊模擬器已啟動，監聽 {HOST}:{PORT}...")
        
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"📥 收到來自 {addr} 的連線！")
                total_bytes = 0
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    total_bytes += len(data)
                print(f"✅ 成功接收檔案，共 {total_bytes} bytes。\n")

if __name__ == '__main__':
    start_mock_server()