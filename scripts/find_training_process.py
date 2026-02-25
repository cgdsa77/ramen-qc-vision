"""查找训练进程PID"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess

print("查找训练进程...")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                          capture_output=True, text=True, encoding='gbk', errors='ignore')
    
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(result.stdout))
        headers = next(reader)
        
        found = False
        for row in reader:
            if len(row) > 1:
                pid = row[1]
                # 获取命令行
                try:
                    cmd_result = subprocess.run(['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'CommandLine'], 
                                              capture_output=True, text=True, encoding='gbk', errors='ignore', timeout=2)
                    cmdline = cmd_result.stdout
                    if 'train_boiling_scooping' in cmdline:
                        print(f"找到训练进程 PID: {pid}")
                        found = True
                        # 返回PID以便后续使用
                        sys.exit(int(pid))
                except:
                    pass
        
        if not found:
            print("未找到训练进程")
            sys.exit(0)
    else:
        print("未找到Python进程")
        sys.exit(0)
except Exception as e:
    print(f"错误: {e}")
    sys.exit(0)

