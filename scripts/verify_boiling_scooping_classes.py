#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证下面及捞面部分的classes.txt一致性"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "下面及捞面"

print("="*60)
print("验证下面及捞面部分classes.txt一致性")
print("="*60)

# 读取主classes.txt
main_classes_file = labels_base / "classes.txt"
if not main_classes_file.exists():
    print(f"错误：主classes.txt文件不存在: {main_classes_file}")
    sys.exit(1)

with open(main_classes_file, 'r', encoding='utf-8') as f:
    main_classes = [line.strip() for line in f if line.strip()]

print(f"\n主classes.txt类别（{len(main_classes)}个）:")
for i, cls in enumerate(main_classes):
    print(f"  {i}: {cls}")

# 检查所有视频子目录
video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir()])
print(f"\n检查 {len(video_dirs)} 个视频目录...")

all_consistent = True
for video_dir in video_dirs:
    video_classes_file = video_dir / "classes.txt"
    if not video_classes_file.exists():
        print(f"  [警告] {video_dir.name}: classes.txt 不存在")
        all_consistent = False
        continue
    
    with open(video_classes_file, 'r', encoding='utf-8') as f:
        video_classes = [line.strip() for line in f if line.strip()]
    
    if video_classes != main_classes:
        print(f"  [错误] {video_dir.name}: classes.txt不一致!")
        print(f"    主文件: {main_classes}")
        print(f"    视频文件: {video_classes}")
        all_consistent = False
    else:
        print(f"  [✓] {video_dir.name}: 一致")

print("\n" + "="*60)
if all_consistent:
    print("✓ 所有视频目录的classes.txt都与主文件一致")
else:
    print("✗ 发现不一致的classes.txt文件，请先修复！")
print("="*60)

