#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析标注数据，检查类别分布和潜在问题"""
from pathlib import Path
from collections import Counter, defaultdict

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
classes_file = labels_base / "classes.txt"

# 读取类别
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]

print("="*60)
print("标注数据分析")
print("="*60)
print(f"\n类别定义（共{len(class_names)}个）:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("\n" + "="*60)
print("各视频的标注统计:")
print("="*60)

# 分析每个视频
for video_dir in sorted(labels_base.glob("cm*")):
    if not video_dir.is_dir() or video_dir.name in ['cm8', 'cm9', 'cm10', 'cm11', 'cm12']:
        continue
    
    label_files = list(video_dir.glob("*.txt"))
    label_files = [f for f in label_files if f.name != 'classes.txt']
    
    if not label_files:
        continue
    
    class_counts = Counter()
    total_boxes = 0
    frames_with_labels = 0
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                frames_with_labels += 1
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_id] += 1
                                total_boxes += 1
                        except ValueError:
                            pass
    
    if total_boxes > 0:
        print(f"\n{video_dir.name}:")
        print(f"  标注的图片数: {frames_with_labels}/{len(label_files)}")
        print(f"  总标注框数: {total_boxes}")
        print(f"  各类别数量:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_boxes) * 100
            print(f"    {class_id} ({class_names[class_id]}): {count} ({percentage:.1f}%)")
        
        # 检查是否使用了未标注的类别
        used_classes = set(class_counts.keys())
        unused_classes = set(range(len(class_names))) - used_classes
        if unused_classes:
            print(f"  [注意] 未使用的类别: {[class_names[i] for i in unused_classes]}")

print("\n" + "="*60)
print("总结:")
print("="*60)

# 总体统计
all_class_counts = Counter()
total_all_boxes = 0

for video_dir in sorted(labels_base.glob("cm*")):
    if not video_dir.is_dir():
        continue
    
    label_files = list(video_dir.glob("*.txt"))
    label_files = [f for f in label_files if f.name != 'classes.txt']
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(class_names):
                            all_class_counts[class_id] += 1
                            total_all_boxes += 1
                    except ValueError:
                        pass

print(f"\n所有视频总计:")
print(f"  总标注框数: {total_all_boxes}")
for class_id in sorted(all_class_counts.keys()):
    count = all_class_counts[class_id]
    percentage = (count / total_all_boxes) * 100 if total_all_boxes > 0 else 0
    print(f"  {class_id} ({class_names[class_id]}): {count} ({percentage:.1f}%)")

unused = set(range(len(class_names))) - set(all_class_counts.keys())
if unused:
    print(f"\n[警告] 以下类别在标注中从未使用: {[class_names[i] for i in unused]}")
    print(f"  建议：从classes.txt中移除这些类别，或补充标注")

