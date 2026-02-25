#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复制最新训练的模型到标准位置"""
import sys
from pathlib import Path
import shutil

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

# 找到最新的训练目录
training_dirs = sorted([d for d in models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('stretch_detection')],
                      key=lambda x: int(x.name.replace('stretch_detection', '') or '0'))

if not training_dirs:
    print("[ERROR] 未找到训练目录")
    sys.exit(1)

latest_dir = training_dirs[-1]
src = latest_dir / "weights" / "best.pt"

# 目标位置
dst1 = project_root / "models" / "stretch_detection" / "weights" / "best.pt"
dst2 = project_root / "models" / "stretch_detection_model.pt"

print("="*60)
print("复制最佳模型到标准位置")
print("="*60)
print(f"\n源模型: {src}")
print(f"训练目录: {latest_dir.name}")

if src.exists():
    # 确保目标目录存在
    dst1.parent.mkdir(parents=True, exist_ok=True)
    
    # 复制到标准位置
    shutil.copy2(src, dst1)
    print(f"\n[OK] 已复制到: {dst1}")
    
    # 复制到最终模型位置
    shutil.copy2(src, dst2)
    print(f"[OK] 已复制到: {dst2}")
    
    # 获取文件大小
    size_mb = src.stat().st_size / (1024 * 1024)
    print(f"\n模型大小: {size_mb:.2f} MB")
    print("\n✓ 模型文件已更新，Web服务器将使用此最佳模型")
    print("="*60)
else:
    print(f"\n[ERROR] 源文件不存在: {src}")
    sys.exit(1)

