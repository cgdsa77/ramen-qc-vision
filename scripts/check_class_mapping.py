#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查类别ID映射是否正确"""
from pathlib import Path
import os
from collections import Counter

# 读取类别定义
classes_file = Path("data/labels/抻面/classes.txt")
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]

print("="*60)
print("类别定义（classes.txt）:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
print("="*60)

# 检查实际标注文件中的类别ID分布
labels_dir = Path("data/labels/抻面")
video_dirs = ['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']

class_id_counter = Counter()
total_files = 0
sample_data = {}

for video_dir in video_dirs:
    video_path = labels_dir / video_dir
    if not video_path.exists():
        continue
    
    # 跳过classes.txt文件
    label_files = [f for f in video_path.glob("*.txt") if f.name != "classes.txt"]
    
    for label_file in label_files[:10]:  # 每个视频检查前10个文件
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    class_id_counter[cls_id] += 1
                    if video_dir not in sample_data:
                        sample_data[video_dir] = []
                    if len(sample_data[video_dir]) < 3:
                        sample_data[video_dir].append((label_file.name, cls_id))
        total_files += 1

print(f"\n检查了 {total_files} 个标注文件")
print("\n类别ID分布（在所有标注文件中）:")
for cls_id in sorted(class_id_counter.keys()):
    count = class_id_counter[cls_id]
    expected_name = class_names[cls_id] if cls_id < len(class_names) else "未知"
    print(f"  类别ID {cls_id} ({expected_name}): {count} 次")

print("\n每个视频的示例标注（前3个文件）:")
for video_dir, samples in sample_data.items():
    print(f"\n  {video_dir}:")
    for filename, cls_id in samples:
        expected_name = class_names[cls_id] if cls_id < len(class_names) else "未知"
        print(f"    {filename}: 类别ID {cls_id} ({expected_name})")

# 检查训练时使用的data.yaml
dataset_yaml = Path("datasets/stretch_detection/data.yaml")
if dataset_yaml.exists():
    import yaml
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    print("\n" + "="*60)
    print("训练时使用的data.yaml中的类别定义:")
    if 'names' in yaml_data:
        for cls_id, name in yaml_data['names'].items():
            print(f"  {cls_id}: {name}")
    print("="*60)

# 检查模型文件中的类别名称
model_path = Path("models/stretch_detection/weights/best.pt")
if model_path.exists():
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print("\n" + "="*60)
        print("模型文件中的类别定义:")
        if hasattr(model, 'names'):
            for cls_id, name in model.names.items():
                print(f"  {cls_id}: {name}")
        print("="*60)
    except Exception as e:
        print(f"\n无法读取模型文件: {e}")

print("\n分析完成！")

