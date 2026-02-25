#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""紧急检查cm1的标注文件，查看noodle_rope被替换的情况"""
from pathlib import Path
from collections import Counter

labels_dir = Path("data/labels/抻面/cm1")

print("="*60)
print("紧急检查cm1标注文件")
print("="*60)

# 读取类别定义
classes_file = Path("data/labels/抻面/classes.txt")
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]

print(f"\n类别定义:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# 统计cm1中各类别的使用情况
class_counter = Counter()
files_with_class = {0: [], 1: [], 2: []}  # 记录包含每个类别的文件

label_files = [f for f in labels_dir.glob("*.txt") if f.name != "classes.txt"]

for label_file in label_files:
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        file_classes = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id < len(class_names):
                    class_counter[cls_id] += 1
                    file_classes.add(cls_id)
        
        # 记录包含每个类别的文件
        for cls_id in file_classes:
            files_with_class[cls_id].append(label_file.name)

print(f"\ncm1标注统计（共{len(label_files)}个文件）:")
total = sum(class_counter.values())
for cls_id in sorted(class_counter.keys()):
    count = class_counter[cls_id]
    percentage = (count / total * 100) if total > 0 else 0
    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"未知({cls_id})"
    print(f"  {cls_id} ({cls_name}): {count} 次 ({percentage:.1f}%)")
    print(f"    包含此类别的文件数: {len(files_with_class[cls_id])}")

# 检查是否有noodle_rope（ID=1）
rope_count = class_counter.get(1, 0)
bundle_count = class_counter.get(2, 0)

print("\n" + "="*60)
if rope_count == 0 and bundle_count > 0:
    print("[严重警告] cm1中没有noodle_rope (ID=1)，但有很多noodle_bundle (ID=2)！")
    print("这确认了noodle_rope被错误替换为noodle_bundle的问题。")
    print(f"\n需要修复的文件数量: {len(files_with_class[2])}")
else:
    print(f"noodle_rope (ID=1): {rope_count} 次")
    print(f"noodle_bundle (ID=2): {bundle_count} 次")

