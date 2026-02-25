#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复classes.txt，只保留3个已标注的类别"""
from pathlib import Path

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
classes_file = labels_base / "classes.txt"

# 只保留3个类别
new_classes = [
    "hand",
    "noodle_rope", 
    "noodle_bundle"
]

# 备份原文件
if classes_file.exists():
    backup_file = classes_file.with_suffix('.txt.bak')
    import shutil
    shutil.copy2(classes_file, backup_file)
    print(f"[备份] 原文件已备份到: {backup_file}")

# 写入新的classes.txt
with open(classes_file, 'w', encoding='utf-8') as f:
    for cls in new_classes:
        f.write(cls + '\n')

print(f"[OK] classes.txt已更新为3个类别:")
for i, cls in enumerate(new_classes):
    print(f"  {i}: {cls}")

print("\n[注意] 标注文件的类别ID已经是正确的（0, 1, 2），无需修改")
print("[下一步] 请重新训练模型以使用正确的类别配置")

