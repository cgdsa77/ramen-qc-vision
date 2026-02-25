#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查标注状态"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "抻面"

print("="*60)
print("抻面视频标注状态检查")
print("="*60)

video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith('cm')])

annotated = []
partially_annotated = []
not_annotated = []

for video_dir in video_dirs:
    txt_files = list(video_dir.glob("*.txt"))
    # 排除classes.txt
    annotation_files = [f for f in txt_files if f.name != "classes.txt"]
    
    if len(annotation_files) > 50:  # 假设至少50个标注文件才算完整标注
        annotated.append((video_dir.name, len(annotation_files)))
    elif len(annotation_files) > 0:
        partially_annotated.append((video_dir.name, len(annotation_files)))
    else:
        not_annotated.append(video_dir.name)

print(f"\n已完整标注 ({len(annotated)}个):")
for name, count in annotated:
    print(f"  - {name}: {count} 个标注文件")

if partially_annotated:
    print(f"\n部分标注 ({len(partially_annotated)}个):")
    for name, count in partially_annotated:
        print(f"  - {name}: {count} 个标注文件")

if not_annotated:
    print(f"\n未标注 ({len(not_annotated)}个):")
    for name in not_annotated:
        print(f"  - {name}")

print("\n" + "="*60)
print(f"总计: {len(video_dirs)} 个视频")
print(f"已标注: {len(annotated)} 个")
print(f"待标注: {len(not_annotated) + len(partially_annotated)} 个")
print("="*60)

