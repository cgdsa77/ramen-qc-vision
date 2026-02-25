"""检查PID 1988进程的详细信息"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess

print("检查PID 1988进程...")
print("=" * 60)

try:
    # 使用wmic获取进程命令行
    result = subprocess.run(['wmic', 'process', 'where', 'ProcessId=1988', 'get', 'CommandLine,ProcessId,Name'], 
                          capture_output=True, text=True, encoding='gbk', errors='ignore')
    print(result.stdout)
    
    # 检查是否是训练进程
    if 'train_boiling_scooping' in result.stdout:
        print("\n✓ 这是训练进程！")
    else:
        print("\n这不是训练进程（命令中不包含train_boiling_scooping）")
except Exception as e:
    print(f"错误: {e}")

