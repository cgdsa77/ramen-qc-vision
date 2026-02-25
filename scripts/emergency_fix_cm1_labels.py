#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
紧急修复cm1的标注文件
将错误的noodle_bundle (ID=2) 改回 noodle_rope (ID=1)
注意：这个脚本假设所有ID=2的标注本来都应该是ID=1的noodle_rope
"""
from pathlib import Path
import shutil
from datetime import datetime

labels_dir = Path("data/labels/抻面/cm1")

print("="*60)
print("紧急修复cm1标注文件")
print("="*60)

# 读取类别定义
classes_file = Path("data/labels/抻面/classes.txt")
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f if line.strip()]

print(f"\n类别定义:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# 目标：将ID=2 (noodle_bundle) 改为 ID=1 (noodle_rope)
OLD_CLASS_ID = 2  # noodle_bundle
NEW_CLASS_ID = 1  # noodle_rope

print(f"\n修复规则: 将类别ID {OLD_CLASS_ID} ({class_names[OLD_CLASS_ID]}) 改为 {NEW_CLASS_ID} ({class_names[NEW_CLASS_ID]})")

# 先备份
backup_dir = Path("data/labels_backup/cm1_emergency")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = backup_dir / f"backup_{timestamp}"

if labels_dir.exists():
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(labels_dir, backup_path)
    print(f"\n[OK] 已备份cm1到: {backup_path}")
else:
    print(f"\n[错误] cm1目录不存在: {labels_dir}")
    exit(1)

# 修复文件
label_files = [f for f in labels_dir.glob("*.txt") if f.name != "classes.txt"]
fixed_count = 0
total_modified_lines = 0

for label_file in label_files:
    modified = False
    modified_lines = 0
    
    # 读取文件
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修改内容
    new_lines = []
    for line in lines:
        original_line = line
        line = line.strip()
        if not line:
            new_lines.append(original_line)
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            if cls_id == OLD_CLASS_ID:
                # 替换类别ID
                parts[0] = str(NEW_CLASS_ID)
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)
                modified = True
                modified_lines += 1
            else:
                new_lines.append(original_line)
        else:
            new_lines.append(original_line)
    
    # 如果修改了，写回文件
    if modified:
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        fixed_count += 1
        total_modified_lines += modified_lines
        print(f"  [修复] {label_file.name}: 修改了 {modified_lines} 个标注")

print("\n" + "="*60)
print(f"修复完成！")
print(f"  - 修复了 {fixed_count} 个文件")
print(f"  - 总共修改了 {total_modified_lines} 个标注")
print(f"  - 备份位置: {backup_path}")
print("\n请运行 python scripts/emergency_check_cm1.py 验证修复结果")

