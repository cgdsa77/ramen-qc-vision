"""查找训练进程的PID"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import psutil
    
    print("=" * 60)
    print("查找训练进程PID")
    print("=" * 60)
    print()
    
    found = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('train_boiling_scooping' in str(cmd) for cmd in cmdline):
                print(f"找到训练进程:")
                print(f"  PID: {proc.info['pid']}")
                print(f"  命令行: {' '.join(cmdline[:5])}...")
                print(f"  状态: {proc.status()}")
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not found:
        print("未找到训练进程，可能已经停止")
    
except ImportError:
    print("需要安装 psutil: pip install psutil")
    print("\n尝试使用 tasklist 查找...")
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                          capture_output=True, text=True, encoding='gbk')
    print(result.stdout)

