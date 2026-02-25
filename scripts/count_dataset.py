#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计数据集大小"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

videos = ['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7']
base = Path('data/processed/抻面')

print('数据集统计:')
total = 0
for v in videos:
    d = base / v
    if d.exists():
        count = len(list(d.glob('*.jpg')))
        print(f'  {v}: {count} 张')
        total += count
    else:
        print(f'  {v}: 目录不存在')

print(f'\n总计: {total} 张图片')
train_imgs = int(total * 0.8)
val_imgs = total - train_imgs
print(f'训练集: {train_imgs} 张 (80%)')
print(f'验证集: {val_imgs} 张 (20%)')

