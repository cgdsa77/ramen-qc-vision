#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查最新训练进度"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import csv
from datetime import datetime

# 查找最新的训练目录
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

# 找到所有boiling_scooping_detection*目录
training_dirs = sorted([d for d in models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('boiling_scooping_detection')],
                      key=lambda x: int(x.name.replace('boiling_scooping_detection', '') or '0'))

if not training_dirs:
    print("未找到训练目录")
    sys.exit(1)

latest_dir = training_dirs[-1]
results_file = latest_dir / "results.csv"

print("="*60)
print("下面及捞面训练进度检查")
print("="*60)
print(f"[信息] 训练目录: {latest_dir.name}")
print(f"[信息] 结果文件: {results_file}")

if not results_file.exists():
    print("\n[状态] 训练刚开始，结果文件尚未生成")
    print("       请稍等片刻再查看...")
    sys.exit(0)

# 读取结果
with open(results_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

if len(rows) < 2:
    print("\n[状态] 训练刚开始，数据尚未生成")
    sys.exit(0)

# 解析表头
header = rows[0]
# 找到关键列的索引
epoch_idx = header.index('epoch') if 'epoch' in header else None
precision_idx = header.index('metrics/precision(B)') if 'metrics/precision(B)' in header else None
recall_idx = header.index('metrics/recall(B)') if 'metrics/recall(B)' in header else None
map50_idx = header.index('metrics/mAP50(B)') if 'metrics/mAP50(B)' in header else None
map50_95_idx = header.index('metrics/mAP50-95(B)') if 'metrics/mAP50-95(B)' in header else None

if epoch_idx is None:
    print("\n[错误] 无法解析结果文件格式")
    sys.exit(1)

# 获取最新一行
latest_row = rows[-1]
current_epoch = int(float(latest_row[epoch_idx]))

# 从args.yaml读取总epochs
args_file = latest_dir / "args.yaml"
total_epochs = 150  # 默认值
if args_file.exists():
    with open(args_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('epochs:'):
                total_epochs = int(line.split(':')[1].strip())
                break

progress = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0

print(f"\n[进度] 训练进度: {current_epoch}/{total_epochs} epochs ({progress:.1f}%)")

if precision_idx is not None and recall_idx is not None and map50_idx is not None:
    current_map50 = float(latest_row[map50_idx])
    
    # 查找历史最高mAP50
    best_map50 = current_map50
    best_epoch = current_epoch
    for row in rows[1:]:
        try:
            epoch_num = int(float(row[epoch_idx]))
            map50_val = float(row[map50_idx])
            if map50_val > best_map50:
                best_map50 = map50_val
                best_epoch = epoch_num
        except (ValueError, IndexError):
            continue
    
    print(f"\n[指标] 最新指标 (Epoch {current_epoch}):")
    print(f"  - 精确率 (Precision): {float(latest_row[precision_idx]):.4f}")
    print(f"  - 召回率 (Recall): {float(latest_row[recall_idx]):.4f}")
    print(f"  - mAP50: {current_map50:.4f}")
    if map50_95_idx is not None:
        print(f"  - mAP50-95: {float(latest_row[map50_95_idx]):.4f}")
    
    # 显示历史最佳mAP50
    if best_epoch != current_epoch:
        print(f"\n[最佳] 历史最高mAP50: {best_map50:.4f} (Epoch {best_epoch})")
        if current_map50 < 0.6 and best_map50 >= 0.6:
            print(f"       ✓ 已超过0.6目标！最佳模型在Epoch {best_epoch}")
        elif current_map50 < 0.6:
            improvement_needed = 0.6 - current_map50
            print(f"       (距离0.6目标还需提升: {improvement_needed:.4f})")

# 检查最佳模型
best_model = latest_dir / "weights" / "best.pt"
if best_model.exists():
    mtime = datetime.fromtimestamp(best_model.stat().st_mtime)
    print(f"\n[模型] 最佳模型存在，最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print(f"\n[模型] 最佳模型尚未生成")

# 计算预计剩余时间
# 需要至少2行数据（表头+至少2行数据）才能计算单个epoch的时间差
if len(rows) >= 3:
    # 获取最近几行的时间戳（如果存在）
    time_idx = header.index('time') if 'time' in header else None
    epoch_idx = header.index('epoch') if 'epoch' in header else None
    if time_idx is not None and epoch_idx is not None:
        try:
            # 获取所有数据行（排除表头）
            data_rows = rows[1:]
            
            # 计算最近几个epoch的时间差（单个epoch的实际时间）
            # 使用最近10行来计算，以获得更稳定的估计
            recent_rows = data_rows[-min(10, len(data_rows)):]
            epoch_times = []
            
            for i in range(1, len(recent_rows)):
                try:
                    prev_time = float(recent_rows[i-1][time_idx])
                    curr_time = float(recent_rows[i][time_idx])
                    # 计算时间差（单个epoch的时间）
                    epoch_duration = curr_time - prev_time
                    # 只使用合理的时间值（排除异常值，比如小于0或大于1小时）
                    if 0 < epoch_duration < 3600:  # 单个epoch应该在0-1小时之间
                        epoch_times.append(epoch_duration)
                except (ValueError, IndexError):
                    continue
            
            if epoch_times:
                # 使用中位数而不是平均值，以避免异常值的影响
                sorted_times = sorted(epoch_times)
                median_time_per_epoch = sorted_times[len(sorted_times) // 2]
                
                remaining_epochs = total_epochs - current_epoch
                estimated_seconds = median_time_per_epoch * remaining_epochs
                estimated_minutes = estimated_seconds / 60
                
                if estimated_minutes > 0:
                    hours = estimated_minutes / 60
                    if hours >= 1:
                        print(f"\n[时间] 预计剩余时间: {estimated_minutes:.1f} 分钟 ({hours:.1f} 小时)")
                        print(f"       (基于最近{len(epoch_times)}个epoch的平均时间: {median_time_per_epoch/60:.1f}分钟/epoch)")
                    else:
                        print(f"\n[时间] 预计剩余时间: {estimated_minutes:.1f} 分钟")
        except Exception as e:
            # 静默失败，不显示错误
            pass

print("\n" + "="*60)
print("[提示] 可以多次运行此脚本查看进度")
print("="*60)

