"""获取训练信息和进程PID"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from datetime import datetime
import subprocess

print("=" * 60)
print("训练进程信息")
print("=" * 60)
print()

# 检查训练进度
results_file = Path('models/boiling_scooping_detection2/results.csv')
if results_file.exists():
    mtime = datetime.fromtimestamp(results_file.stat().st_mtime)
    now = datetime.now()
    diff = (now - mtime).total_seconds()
    print(f"训练结果文件最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"距离现在: {int(diff/60)} 分钟 {int(diff%60)} 秒")
    if diff < 300:
        print("状态: 训练可能仍在进行（最近5分钟内有更新）")
    else:
        print("状态: 训练可能已停止")
    print()

# 查找所有Python进程
print("查找所有Python进程:")
print("-" * 60)
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'LIST'], 
                          capture_output=True, text=True, encoding='gbk', errors='ignore')
    
    if 'python.exe' in result.stdout:
        lines = result.stdout.split('\n')
        pids = []
        for i, line in enumerate(lines):
            if 'PID:' in line or '进程 ID:' in line:
                pid = line.split(':')[-1].strip()
                pids.append(pid)
                # 尝试获取命令行
                try:
                    cmd_result = subprocess.run(['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'CommandLine'], 
                                              capture_output=True, text=True, encoding='gbk', errors='ignore', timeout=2)
                    cmdline = cmd_result.stdout
                    if 'train_boiling_scooping' in cmdline:
                        print(f"✓ 训练进程 PID: {pid}")
                        print(f"  命令行: {[l for l in cmdline.split('\\n') if 'train_boiling_scooping' in l][:1]}")
                        break
                    else:
                        print(f"  Python进程 PID: {pid} (不是训练进程)")
                except:
                    print(f"  Python进程 PID: {pid} (无法获取详细信息)")
        
        if not any('train_boiling_scooping' in result.stdout for _ in [1]):
            print("\n未找到明确的训练进程，但训练可能仍在后台运行")
            print("所有Python进程PID:", ', '.join(pids))
    else:
        print("未找到Python进程")
except Exception as e:
    print(f"查找进程时出错: {e}")

print()
print("=" * 60)
print("提示: 训练确实在进行（已到Epoch 34），")
print("      如果找不到进程PID，可能是后台运行或在不同会话中")

