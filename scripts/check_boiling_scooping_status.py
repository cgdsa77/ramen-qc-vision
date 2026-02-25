#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查下面及捞面标注状态"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "下面及捞面"
processed_base = project_root / "data" / "processed" / "下面及捞面"

print("="*60)
print("下面及捞面视频标注状态检查")
print("="*60)

if not processed_base.exists():
    print(f"错误：processed目录不存在: {processed_base}")
    sys.exit(1)

video_dirs = sorted([d for d in processed_base.iterdir() if d.is_dir()])

if not video_dirs:
    print(f"警告：在 {processed_base} 中未找到视频目录")
    sys.exit(1)

annotated = []
not_annotated = []

for video_dir in video_dirs:
    video_name = video_dir.name
    
    # 统计图片数量
    image_files = list(video_dir.glob("*.jpg"))
    image_count = len(image_files)
    
    # 统计标注文件数量
    label_dir = labels_base / video_name
    if label_dir.exists():
        label_files = [f for f in label_dir.glob("*.txt") if f.name != "classes.txt"]
        label_count = len(label_files)
        
        if label_count > 0:
            annotated.append((video_name, image_count, label_count))
        else:
            not_annotated.append((video_name, image_count, 0))
    else:
        not_annotated.append((video_name, image_count, 0))

print(f"\n已标注 ({len(annotated)}个):")
total_images = 0
total_labels = 0
for name, img_count, label_count in annotated:
    progress = (label_count / img_count * 100) if img_count > 0 else 0
    print(f"  - {name}: {label_count}/{img_count} 张 ({progress:.1f}%)")
    total_images += img_count
    total_labels += label_count

if not_annotated:
    print(f"\n未标注 ({len(not_annotated)}个):")
    for name, img_count, label_count in not_annotated:
        print(f"  - {name}: {img_count} 张图片")

print("\n" + "="*60)
print(f"总计: {len(video_dirs)} 个视频")
print(f"已标注: {len(annotated)} 个")
print(f"待标注: {len(not_annotated)} 个")
if annotated:
    print(f"已标注图片: {total_labels} 张")
print(f"总图片数: {sum(img for _, img, _ in annotated) + sum(img for _, img, _ in not_annotated)} 张")
print("="*60)

