#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时监控训练进度
"""
import sys
import os
import time
from pathlib import Path
import csv
from datetime import datetime

# 设置输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = Path(__file__).parent.parent
results_file = project_root / "models" / "stretch_detection" / "results.csv"

def clear_screen():
    """清屏（Windows和Linux兼容）"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_epoch():
    """获取最新的epoch数据"""
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        return rows[-1]
    except Exception:
        return None

def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"

def monitor_training():
    """实时监控训练进度"""
    total_epochs = 150
    last_epoch = -1
    last_update_time = None
    
    print("="*70)
    print("实时训练进度监控")
    print("="*70)
    print("按 Ctrl+C 停止监控（不会停止训练）")
    print("="*70 + "\n")
    
    try:
        while True:
            latest = get_latest_epoch()
            
            if latest:
                current_epoch = int(latest['epoch'])
                
                # 只有在epoch更新时才刷新屏幕
                if current_epoch != last_epoch:
                    clear_screen()
                    last_epoch = current_epoch
                    last_update_time = datetime.now()
                    
                    progress = (current_epoch / total_epochs) * 100
                    elapsed_time = float(latest['time'])
                    
                    # 计算进度条
                    bar_length = 50
                    filled_length = int(bar_length * current_epoch // total_epochs)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    print("="*70)
                    print(f"训练进度监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("="*70)
                    print(f"\n[进度] Epoch: {current_epoch}/{total_epochs} ({progress:.1f}%)")
                    print(f"[进度条] |{bar}|")
                    print(f"\n[时间] 累计训练时间: {format_time(elapsed_time)}")
                    
                    # 估算剩余时间
                    if current_epoch > 1:
                        avg_time_per_epoch = elapsed_time / current_epoch
                        remaining_epochs = total_epochs - current_epoch
                        estimated_remaining = avg_time_per_epoch * remaining_epochs
                        print(f"[预计] 剩余时间: {format_time(estimated_remaining)}")
                    
                    print(f"\n[指标] 最新指标 (Epoch {current_epoch}):")
                    print(f"  - 精确率 (Precision): {float(latest['metrics/precision(B)']):.4f}")
                    print(f"  - 召回率 (Recall):    {float(latest['metrics/recall(B)']):.4f}")
                    print(f"  - mAP50:              {float(latest['metrics/mAP50(B)']):.4f}")
                    print(f"  - mAP50-95:           {float(latest['metrics/mAP50-95(B)']):.4f}")
                    
                    print(f"\n[损失] 训练损失:")
                    print(f"  - Box Loss:  {float(latest['train/box_loss']):.4f}")
                    print(f"  - Class Loss: {float(latest['train/cls_loss']):.4f}")
                    print(f"  - DFL Loss:   {float(latest['train/dfl_loss']):.4f}")
                    
                    if 'val/box_loss' in latest:
                        print(f"\n[损失] 验证损失:")
                        print(f"  - Box Loss:  {float(latest['val/box_loss']):.4f}")
                        print(f"  - Class Loss: {float(latest['val/cls_loss']):.4f}")
                        print(f"  - DFL Loss:   {float(latest['val/dfl_loss']):.4f}")
                    
                    # 显示趋势（对比前一个epoch，需要读取所有行）
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            all_rows = list(reader)
                        
                        if len(all_rows) > 1:
                            prev = all_rows[-2]
                            prev_map50 = float(prev['metrics/mAP50(B)'])
                            curr_map50 = float(latest['metrics/mAP50(B)'])
                            improvement = curr_map50 - prev_map50
                            
                            trend_symbol = "↑ 提升" if improvement > 0 else "↓ 下降" if improvement < 0 else "→ 持平"
                            print(f"\n[趋势] mAP50变化: {improvement:+.4f} ({trend_symbol})")
                    except Exception:
                        pass
                    
                    print("\n" + "="*70)
                    print("[提示] 训练会自动继续，按 Ctrl+C 仅停止监控")
                    print("="*70)
            
            time.sleep(5)  # 每5秒检查一次
            
    except KeyboardInterrupt:
        print("\n\n[提示] 监控已停止，但训练仍在后台继续")
        print("       可以运行 'python scripts/check_training_progress.py' 查看最新进度")
        print("="*70)

if __name__ == "__main__":
    monitor_training()

