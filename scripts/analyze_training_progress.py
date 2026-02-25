#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练进度，查找最佳mAP50和预计剩余时间问题
"""
import sys
import csv
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

# 找到最新的训练目录
training_dirs = sorted([d for d in models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('stretch_detection')],
                      key=lambda x: int(x.name.replace('stretch_detection', '') or '0'))

if not training_dirs:
    print("未找到训练目录")
    sys.exit(1)

latest_dir = training_dirs[-1]
results_file = latest_dir / "results.csv"

print("="*60)
print(f"分析训练进度: {latest_dir.name}")
print("="*60)

if not results_file.exists():
    print("结果文件不存在")
    sys.exit(1)

# 读取结果
with open(results_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

if len(rows) < 2:
    print("数据不足")
    sys.exit(1)

header = rows[0]
epoch_idx = header.index('epoch')
time_idx = header.index('time')
map50_idx = header.index('metrics/mAP50(B)')

data_rows = rows[1:]

print(f"\n总epoch数: {len(data_rows)}")
print("\n" + "-"*60)
print("mAP50趋势分析:")
print("-"*60)

# 查找最高mAP50
max_map50 = 0
max_epoch = 0
map50_history = []

for row in data_rows:
    try:
        epoch = int(row[epoch_idx])
        map50 = float(row[map50_idx])
        time_val = float(row[time_idx])
        map50_history.append((epoch, map50, time_val))
        if map50 > max_map50:
            max_map50 = map50
            max_epoch = epoch
    except (ValueError, IndexError):
        continue

# 显示最近的mAP50值
print("\n最近10个epoch的mAP50:")
for epoch, map50, time_val in map50_history[-10:]:
    marker = " <-- 最高" if map50 == max_map50 else ""
    print(f"  Epoch {epoch}: mAP50={map50:.4f}, 时间={time_val:.1f}秒{marker}")

print(f"\n最高mAP50: {max_map50:.4f} (Epoch {max_epoch})")
current_epoch = map50_history[-1][0]
current_map50 = map50_history[-1][1]
print(f"当前mAP50: {current_map50:.4f} (Epoch {current_epoch})")

print("\n" + "-"*60)
print("时间分析 (预计剩余时间计算问题):")
print("-"*60)

# 分析时间数据
if len(map50_history) >= 2:
    times = [t for _, _, t in map50_history]
    print(f"\n每个epoch的时间 (秒):")
    print(f"  最小: {min(times):.1f}")
    print(f"  最大: {max(times):.1f}")
    print(f"  平均: {sum(times)/len(times):.1f}")
    print(f"  中位数: {sorted(times)[len(times)//2]:.1f}")
    
    # 最近的epoch时间
    recent_times = [t for _, _, t in map50_history[-10:]]
    print(f"\n最近10个epoch的时间:")
    print(f"  平均: {sum(recent_times)/len(recent_times):.1f}秒")
    print(f"  中位数: {sorted(recent_times)[len(recent_times)//2]:.1f}秒")
    
    # 检查是否有异常值
    if len(times) > 1:
        avg_time = sum(times) / len(times)
        # 找出异常大的值（超过平均值的3倍）
        outliers = [t for t in times if t > avg_time * 3]
        if outliers:
            print(f"\n警告: 发现异常大的epoch时间值: {outliers}")
            print("这些异常值会导致预计剩余时间计算不准确")

print("\n" + "="*60)

