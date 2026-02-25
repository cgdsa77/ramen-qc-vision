#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证模型类别是否正确"""
from pathlib import Path
from ultralytics import YOLO

# 读取期望的类别
classes_file = Path("data/labels/抻面/classes.txt")
with open(classes_file, 'r', encoding='utf-8') as f:
    expected_classes = [line.strip() for line in f if line.strip()]

print("="*60)
print("期望的类别（从classes.txt）:")
for i, name in enumerate(expected_classes):
    print(f"  {i}: {name}")
print("="*60)

# 检查各个模型文件
model_files = [
    Path("models/stretch_detection/weights/best.pt"),
    Path("models/stretch_detection_model.pt"),
    Path("models/stretch_detection6/weights/best.pt"),
]

for model_file in model_files:
    if not model_file.exists():
        print(f"\n[跳过] {model_file} (不存在)")
        continue
    
    print(f"\n检查模型: {model_file}")
    try:
        model = YOLO(str(model_file))
        model_classes = list(model.names.values())
        
        print(f"  模型类别数: {len(model_classes)}")
        print(f"  期望类别数: {len(expected_classes)}")
        
        if len(model_classes) != len(expected_classes):
            print(f"  [错误] 类别数量不匹配！")
        else:
            match = True
            for i, (expected, actual) in enumerate(zip(expected_classes, model_classes)):
                if expected != actual:
                    print(f"  [错误] 位置 {i}: 期望 '{expected}', 实际 '{actual}'")
                    match = False
            
            if match:
                print(f"  [正确] 类别映射完全匹配！")
            else:
                print(f"  [警告] 存在类别映射不匹配")
        
        print(f"  模型中的类别:")
        for i, name in model.names.items():
            print(f"    {i}: {name}")
            
    except Exception as e:
        print(f"  [错误] 无法加载模型: {e}")

print("\n" + "="*60)

