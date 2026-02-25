#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试Web服务启动"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("测试Web服务启动")
print("="*60)

try:
    print("\n1. 检查依赖...")
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import uvicorn
    print("   [OK] 依赖检查通过")
    
    print("\n2. 检查文件路径...")
    web_dir = project_root / "web"
    if web_dir.exists():
        print(f"   [OK] Web目录存在: {web_dir}")
    else:
        print(f"   [错误] Web目录不存在: {web_dir}")
    
    scoring_tool = web_dir / "stretch_scoring_tool.html"
    if scoring_tool.exists():
        print(f"   [OK] 评分工具文件存在: {scoring_tool}")
    else:
        print(f"   [警告] 评分工具文件不存在: {scoring_tool}")
    
    print("\n3. 检查数据目录...")
    labels_dir = project_root / "data" / "labels" / "抻面"
    if labels_dir.exists():
        videos = [d.name for d in labels_dir.iterdir() if d.is_dir() and d.name.startswith('cm')]
        print(f"   [OK] 找到 {len(videos)} 个视频: {videos[:5]}")
    else:
        print(f"   [警告] 标注目录不存在: {labels_dir}")
    
    print("\n4. 尝试导入start_web...")
    import start_web
    print("   [OK] start_web模块导入成功")
    
    print("\n" + "="*60)
    print("测试完成！可以尝试启动服务:")
    print("  python start_web.py")
    print("="*60)
    
except Exception as e:
    print(f"\n[错误] 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

