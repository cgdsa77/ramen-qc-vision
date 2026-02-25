#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动Web服务器
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.api_server import run_api

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动Ramen QC检测系统")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    run_api(host=args.host, port=args.port, config_path=args.config)

