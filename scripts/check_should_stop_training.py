"""检查是否应该停止训练（防止过拟合）"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import csv

def check_should_stop(results_file: Path, no_improvement_epochs: int = 25, min_improvement: float = 0.001):
    """检查是否应该停止训练"""
    if not results_file.exists():
        print("结果文件不存在")
        return False
    
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) < no_improvement_epochs + 5:
        print(f"数据点不足（需要至少{no_improvement_epochs + 5}个epoch），继续训练")
        return False
    
    # 获取最近的指标
    recent_rows = rows[-no_improvement_epochs:]
    map50s = [float(row['metrics/mAP50(B)']) for row in recent_rows]
    
    current_map50 = map50s[-1]
    best_in_window = max(map50s)
    
    # 检查是否在最近N个epoch中有显著提升
    improvement = best_in_window - map50s[0]
    
    print("=" * 60)
    print("训练停止检查")
    print("=" * 60)
    print(f"当前mAP50: {current_map50:.4f}")
    print(f"最近{no_improvement_epochs}个epoch中最高mAP50: {best_in_window:.4f}")
    print(f"最近{no_improvement_epochs}个epoch中的提升: {improvement:+.4f}")
    print()
    
    # 判断条件
    should_stop = False
    reasons = []
    
    if current_map50 >= 0.95:
        reasons.append(f"✓ mAP50已达到0.95+（{current_map50:.4f}），性能优秀")
    
    if improvement < min_improvement:
        should_stop = True
        reasons.append(f"⚠ 最近{no_improvement_epochs}个epoch中提升小于{min_improvement}，可能已收敛")
    
    if current_map50 < best_in_window - 0.01:
        should_stop = True
        reasons.append(f"⚠ 当前mAP50低于最近最高值{best_in_window:.4f}，可能出现过拟合")
    
    print("分析结果:")
    for reason in reasons:
        print(f"  {reason}")
    print()
    
    if should_stop:
        print("=" * 60)
        print("建议：可以停止训练，防止过拟合")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("建议：继续训练，指标仍在提升或稳定")
        print("=" * 60)
        return False

if __name__ == "__main__":
    results_file = Path("models/boiling_scooping_detection4/results.csv")
    should_stop = check_should_stop(results_file, no_improvement_epochs=25, min_improvement=0.001)
    sys.exit(0 if not should_stop else 1)

