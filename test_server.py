#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试服务器"""
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("正在启动服务器...")
print(f"项目根目录: {project_root}")

try:
    from src.pipelines.api_server import run_api
    print("导入成功，启动服务器...")
    run_api(host="127.0.0.1", port=8000)
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

