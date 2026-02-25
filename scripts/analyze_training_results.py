#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析训练结果"""
from pathlib import Path
import csv

results_file = Path("models/stretch_detection6/results.csv")

with open(results_file, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print("="*60)
print("训练结果分析")
print("="*60)
print(f"\n总训练轮数: {len(rows)}")
print(f"最后完成的epoch: {rows[-1]['epoch']}")

# 找到最佳mAP50
best_epoch = max(rows, key=lambda x: float(x['metrics/mAP50(B)']))
print(f"\n最佳mAP50: {best_epoch['metrics/mAP50(B)']} (Epoch {best_epoch['epoch']})")

# 最后几个epoch的情况
print(f"\n最后5个epoch的情况:")
for row in rows[-5:]:
    epoch = row['epoch']
    map50 = row['metrics/mAP50(B)']
    precision = row['metrics/precision(B)']
    recall = row['metrics/recall(B)']
    print(f"  Epoch {epoch}: mAP50={map50}, Precision={precision}, Recall={recall}")

print("\n" + "="*60)

