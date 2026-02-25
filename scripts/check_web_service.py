"""检查Web服务状态"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import socket

def check_port(host, port):
    """检查端口是否在监听"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

port = 8000
if check_port('127.0.0.1', port):
    print("=" * 60)
    print("Web服务运行中")
    print("=" * 60)
    print()
    print(f"访问地址:")
    print(f"  下面及捞面检测界面: http://localhost:{port}/web/boiling_scooping_detection.html")
    print(f"  API文档: http://localhost:{port}/docs")
    print(f"  抻面检测界面: http://localhost:{port}/web/video_detection.html")
    print()
    print("=" * 60)
else:
    print(f"Web服务未运行在端口{port}")

