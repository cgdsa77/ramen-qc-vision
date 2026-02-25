#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查下面及捞面标注数据的统计信息
"""
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.stdout.reconfigure(encoding='utf-8')

labels_base = project_root / "data" / "labels" / "下面及捞面"
classes_file = labels_base / "classes.txt"

# 读取类别
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]

print("=" * 60)
print("下面及捞面标注数据检查")
print("=" * 60)
print(f"\n类别定义（共{len(classes)}个）:")
for i, cls in enumerate(classes):
    print(f"  {i}: {cls}")

# 统计每个视频的标注
video_dirs = sorted([d for d in labels_base.iterdir() 
                    if d.is_dir() and d.name.startswith('xl')])

total_images = 0
total_labels = 0
class_counts = defaultdict(int)
video_stats = []

for video_dir in video_dirs:
    label_files = [f for f in video_dir.glob("*.txt") if f.name != "classes.txt"]
    video_label_count = 0
    video_class_counts = defaultdict(int)
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(classes):
                            video_class_counts[cls_id] += 1
                            class_counts[cls_id] += 1
                            video_label_count += 1
                            total_labels += 1
        except Exception as e:
            print(f"  [错误] 读取 {label_file.name} 失败: {e}")
    
    if label_files:
        total_images += len(label_files)
        video_stats.append({
            'name': video_dir.name,
            'images': len(label_files),
            'labels': video_label_count,
            'class_counts': dict(video_class_counts)
        })

print(f"\n数据集统计:")
print(f"  - 视频数量: {len(video_stats)}")
print(f"  - 图片数量: {total_images}")
print(f"  - 标注数量: {total_labels}")

print(f"\n各类别标注统计:")
for i, cls in enumerate(classes):
    count = class_counts[i]
    percentage = (count / total_labels * 100) if total_labels > 0 else 0
    print(f"  {i} ({cls}): {count} ({percentage:.1f}%)")

print(f"\n各视频详细统计:")
for stat in video_stats[:10]:  # 只显示前10个
    print(f"\n  {stat['name']}:")
    print(f"    - 图片数: {stat['images']}")
    print(f"    - 标注数: {stat['labels']}")
    if stat['labels'] > 0:
        print(f"    - 各类别:")
        for cls_id, count in sorted(stat['class_counts'].items()):
            print(f"      {cls_id} ({classes[cls_id]}): {count}")

# 检查数据集目录
dataset_labels_dir = project_root / "datasets" / "boiling_scooping_detection" / "labels" / "train"
if dataset_labels_dir.exists():
    dataset_label_files = list(dataset_labels_dir.glob("*.txt"))
    print(f"\n训练数据集:")
    print(f"  - 标注文件数: {len(dataset_label_files)}")
    
    # 检查数据集中的类别分布
    dataset_class_counts = defaultdict(int)
    for label_file in dataset_label_files[:100]:  # 检查前100个文件
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(classes):
                                dataset_class_counts[cls_id] += 1
        except:
            pass
    
    print(f"  - 数据集中的类别分布（前100个文件样本）:")
    for cls_id, count in sorted(dataset_class_counts.items()):
        print(f"    {cls_id} ({classes[cls_id]}): {count}")

print("\n" + "=" * 60)

