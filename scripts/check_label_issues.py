#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查标注文件中是否有错误的类别ID"""
from pathlib import Path
from collections import Counter, defaultdict

# 读取类别定义
classes_file = Path("data/labels/抻面/classes.txt")
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]

print("="*60)
print("类别定义:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
print("="*60)

# 检查所有标注文件
labels_dir = Path("data/labels/抻面")
video_dirs = ['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7', 'cm8', 'cm9', 'cm10', 'cm11', 'cm12']

# 统计每个视频中各类别的数量
video_stats = defaultdict(lambda: Counter())
problem_files = []

for video_dir in video_dirs:
    video_path = labels_dir / video_dir
    if not video_path.exists():
        continue
    
    label_files = [f for f in video_path.glob("*.txt") if f.name != "classes.txt"]
    
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id < len(class_names):
                        video_stats[video_dir][cls_id] += 1
                    else:
                        # 发现无效的类别ID
                        problem_files.append((video_dir, label_file.name, cls_id))

print("\n各视频的类别统计:")
for video_dir in sorted(video_stats.keys()):
    stats = video_stats[video_dir]
    print(f"\n  {video_dir}:")
    total = sum(stats.values())
    for cls_id in sorted(stats.keys()):
        count = stats[cls_id]
        percentage = (count / total * 100) if total > 0 else 0
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"未知({cls_id})"
        print(f"    {cls_id} ({cls_name}): {count} 次 ({percentage:.1f}%)")

# 特别检查cm4，看noodle_rope和noodle_bundle的情况
if 'cm4' in video_stats:
    print("\n" + "="*60)
    print("cm4详细分析:")
    cm4_stats = video_stats['cm4']
    total = sum(cm4_stats.values())
    
    # 检查noodle_rope (ID=1) 和 noodle_bundle (ID=2)
    rope_count = cm4_stats.get(1, 0)
    bundle_count = cm4_stats.get(2, 0)
    
    print(f"  总标注数: {total}")
    print(f"  noodle_rope (ID=1): {rope_count} 次 ({rope_count/total*100:.1f}% if total > 0 else 0)")
    print(f"  noodle_bundle (ID=2): {bundle_count} 次 ({bundle_count/total*100:.1f}% if total > 0 else 0)")
    
    if rope_count == 0 and bundle_count > 0:
        print("\n  [警告] cm4中没有发现noodle_rope，但有很多noodle_bundle！")
        print("  这可能是标注被错误修改了。")
    
    # 检查cm4的所有标注文件，找出包含类别1和2的文件
    cm4_path = labels_dir / 'cm4'
    rope_files = []
    bundle_files = []
    
    for label_file in cm4_path.glob("*.txt"):
        if label_file.name == "classes.txt":
            continue
        with open(label_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == 1:
                        rope_files.append(label_file.name)
                    elif cls_id == 2:
                        bundle_files.append(label_file.name)
    
    print(f"\n  包含noodle_rope (ID=1)的文件: {len(set(rope_files))} 个")
    print(f"  包含noodle_bundle (ID=2)的文件: {len(set(bundle_files))} 个")

if problem_files:
    print("\n" + "="*60)
    print("发现的问题文件（包含无效类别ID）:")
    for video_dir, filename, cls_id in problem_files:
        print(f"  {video_dir}/{filename}: 无效类别ID {cls_id}")

print("\n" + "="*60)
print("检查完成！")

