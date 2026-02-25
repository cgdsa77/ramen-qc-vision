#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查数据集准备情况"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

dataset_yaml = Path('datasets/stretch_detection/data.yaml')
print(f'数据集配置文件存在: {dataset_yaml.exists()}')

if dataset_yaml.exists():
    print(f'数据集目录: {dataset_yaml.parent}')
    train_dir = dataset_yaml.parent / 'images' / 'train'
    val_dir = dataset_yaml.parent / 'images' / 'val'
    
    train_imgs = list(train_dir.glob('*.jpg')) if train_dir.exists() else []
    val_imgs = list(val_dir.glob('*.jpg')) if val_dir.exists() else []
    
    print(f'训练集目录存在: {train_dir.exists()}, 图片数: {len(train_imgs)}')
    print(f'验证集目录存在: {val_dir.exists()}, 图片数: {len(val_imgs)}')
    
    if len(train_imgs) == 0:
        print('\n[错误] 训练集为空！')
    if len(val_imgs) == 0:
        print('[警告] 验证集为空！')
else:
    print('\n[错误] 数据集配置文件不存在！')

