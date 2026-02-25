"""查找所有Python进程"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess
import re

print("=" * 60)
print("查找Python进程")
print("=" * 60)
print()

# 使用tasklist查找所有python进程
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/V', '/FO', 'CSV'], 
                          capture_output=True, text=True, encoding='gbk', errors='ignore')
    
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        # 解析CSV格式
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(result.stdout))
        headers = next(reader)
        
        print(f"{'PID':<10} {'内存使用':<15} {'会话名':<15} {'窗口标题'}")
        print("-" * 60)
        
        for row in reader:
            if len(row) >= len(headers):
                pid = row[1] if len(row) > 1 else 'N/A'
                mem = row[4] if len(row) > 4 else 'N/A'
                session = row[6] if len(row) > 6 else 'N/A'
                window_title = row[8] if len(row) > 8 else 'N/A'
                print(f"{pid:<10} {mem:<15} {session:<15} {window_title[:40]}")
    else:
        print("未找到Python进程")
except Exception as e:
    print(f"错误: {e}")
    print("\n尝试使用wmic...")
    try:
        result = subprocess.run(['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'], 
                              capture_output=True, text=True, encoding='gbk', errors='ignore')
        print(result.stdout)
    except Exception as e2:
        print(f"wmic也失败: {e2}")

