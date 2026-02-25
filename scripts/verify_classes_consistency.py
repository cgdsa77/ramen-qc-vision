#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证所有视频目录下的classes.txt文件是否与主classes.txt一致
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
main_classes_file = labels_base / "classes.txt"

# 读取主classes.txt
if not main_classes_file.exists():
    print(f"错误：主classes.txt文件不存在: {main_classes_file}")
    sys.exit(1)

with open(main_classes_file, 'r', encoding='utf-8') as f:
    main_classes = [line.strip() for line in f if line.strip()]

print("="*60)
print("验证classes.txt一致性")
print("="*60)
print(f"\n主classes.txt ({main_classes_file.name}):")
for i, cls in enumerate(main_classes):
    print(f"  {i}: {cls}")

print("\n" + "-"*60)

# 检查每个视频目录
video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith('cm')])

if not video_dirs:
    print("未找到视频目录")
    sys.exit(1)

all_consistent = True
inconsistent_videos = []

for video_dir in video_dirs:
    video_classes_file = video_dir / "classes.txt"
    
    if not video_classes_file.exists():
        print(f"\n[警告] {video_dir.name}: classes.txt 不存在")
        all_consistent = False
        inconsistent_videos.append(video_dir.name)
        continue
    
    with open(video_classes_file, 'r', encoding='utf-8') as f:
        video_classes = [line.strip() for line in f if line.strip()]
    
    if video_classes != main_classes:
        print(f"\n[不一致] {video_dir.name}:")
        print(f"  主文件: {main_classes}")
        print(f"  视频文件: {video_classes}")
        all_consistent = False
        inconsistent_videos.append(video_dir.name)
    else:
        print(f"[一致] {video_dir.name}")

print("\n" + "="*60)
if all_consistent:
    print("✓ 所有视频目录的classes.txt都与主文件一致！")
    sys.exit(0)
else:
    print(f"✗ 发现 {len(inconsistent_videos)} 个不一致的视频目录:")
    for v in inconsistent_videos:
        print(f"  - {v}")
    print("\n需要修复这些文件以确保训练一致性！")
    sys.exit(1)

