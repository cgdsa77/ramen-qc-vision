#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看训练进度
"""
import sys
import os
from pathlib import Path
import csv
from datetime import datetime

# 设置输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path(__file__).parent.parent

def find_latest_training_results():
    """查找最新的训练结果文件"""
    models_dir = project_root / "models"
    # 查找所有stretch_detection*目录下的results.csv
    results_files = list(models_dir.glob("stretch_detection*/results.csv"))
    if not results_files:
        return None
    # 返回最新修改的文件
    return max(results_files, key=lambda p: p.stat().st_mtime)

results_file = find_latest_training_results()

def check_training_progress():
    """检查训练进度"""
    print("="*60)
    print("训练进度检查")
    print("="*60)
    
    # 检查结果文件是否存在
    if results_file is None or not results_file.exists():
        print("[!] 训练结果文件不存在，训练可能还没开始")
        return
    
    print(f"[信息] 查看训练目录: {results_file.parent.name}")
    print(f"[信息] 结果文件: {results_file}")
    
    # 读取CSV文件
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("[!] 训练结果文件为空")
        return
    
    # 获取最新一行（最后一个epoch的结果）
    latest = rows[-1]
    
    total_epochs = 150  # 目标训练轮数
    current_epoch = int(latest['epoch'])
    progress = (current_epoch / total_epochs) * 100
    
    print(f"\n[进度] 训练进度: {current_epoch}/{total_epochs} epochs ({progress:.1f}%)")
    print(f"[时间] 累计训练时间: {float(latest['time']):.1f} 秒 ({float(latest['time'])/60:.1f} 分钟)")
    
    # 显示关键指标
    print(f"\n[指标] 最新指标 (Epoch {current_epoch}):")
    print(f"  - 精确率 (Precision): {float(latest['metrics/precision(B)']):.4f}")
    print(f"  - 召回率 (Recall): {float(latest['metrics/recall(B)']):.4f}")
    print(f"  - mAP50: {float(latest['metrics/mAP50(B)']):.4f}")
    print(f"  - mAP50-95: {float(latest['metrics/mAP50-95(B)']):.4f}")
    
    # 训练损失
    print(f"\n[损失] 训练损失:")
    print(f"  - Box Loss: {float(latest['train/box_loss']):.4f}")
    print(f"  - Class Loss: {float(latest['train/cls_loss']):.4f}")
    print(f"  - DFL Loss: {float(latest['train/dfl_loss']):.4f}")
    
    # 验证损失
    if 'val/box_loss' in latest:
        print(f"\n[损失] 验证损失:")
        print(f"  - Box Loss: {float(latest['val/box_loss']):.4f}")
        print(f"  - Class Loss: {float(latest['val/cls_loss']):.4f}")
        print(f"  - DFL Loss: {float(latest['val/dfl_loss']):.4f}")
    
    # 显示趋势（对比前一个epoch）
    if len(rows) > 1:
        prev = rows[-2]
        prev_map50 = float(prev['metrics/mAP50(B)'])
        curr_map50 = float(latest['metrics/mAP50(B)'])
        improvement = curr_map50 - prev_map50
        
        trend_symbol = "↑ 提升" if improvement > 0 else "↓ 下降" if improvement < 0 else "→ 持平"
        print(f"\n[趋势] 趋势分析:")
        print(f"  - mAP50变化: {improvement:+.4f} ({trend_symbol})")
        
        # 估算剩余时间
        if current_epoch > 1:
            avg_time_per_epoch = float(latest['time']) / current_epoch
            remaining_epochs = total_epochs - current_epoch
            estimated_remaining_time = avg_time_per_epoch * remaining_epochs
            print(f"  - 预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
    
    # 检查最佳模型
    best_model = project_root / "models" / "stretch_detection" / "weights" / "best.pt"
    if best_model.exists():
        mtime = datetime.fromtimestamp(best_model.stat().st_mtime)
        print(f"\n[模型] 最佳模型存在，最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*60)
    print("[提示] 训练会持续到150轮或达到早停条件")
    print("       可以多次运行此脚本查看进度")
    print("="*60)

if __name__ == "__main__":
    try:
        check_training_progress()
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()

