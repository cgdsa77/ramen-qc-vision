"""分析训练趋势，判断是否过拟合"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import csv
from collections import deque

def analyze_training_trend(results_file: Path, window_size: int = 20):
    """分析最近N个epoch的训练趋势"""
    if not results_file.exists():
        print("结果文件不存在")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) < 10:
        print("数据点太少，无法分析趋势")
        return
    
    # 获取最近的指标
    recent_rows = rows[-window_size:]
    
    epochs = [int(row['epoch']) for row in recent_rows]
    map50s = [float(row['metrics/mAP50(B)']) for row in recent_rows]
    precisions = [float(row['metrics/precision(B)']) for row in recent_rows]
    recalls = [float(row['metrics/recall(B)']) for row in recent_rows]
    
    print("=" * 60)
    print("训练趋势分析（最近20个epoch）")
    print("=" * 60)
    print()
    
    # 计算趋势
    def calculate_trend(values):
        """计算趋势：返回斜率（正数=上升，负数=下降，接近0=稳定）"""
        if len(values) < 2:
            return 0
        # 使用线性回归的简单版本
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        return numerator / denominator
    
    map50_trend = calculate_trend(map50s)
    precision_trend = calculate_trend(precisions)
    recall_trend = calculate_trend(recalls)
    
    print(f"当前指标（Epoch {epochs[-1]}）:")
    print(f"  mAP50: {map50s[-1]:.4f}")
    print(f"  精确率: {precisions[-1]:.4f}")
    print(f"  召回率: {recalls[-1]:.4f}")
    print()
    
    print(f"最近{window_size}个epoch的趋势:")
    print(f"  mAP50趋势: {map50_trend:+.6f} {'↑上升' if map50_trend > 0.0001 else '↓下降' if map50_trend < -0.0001 else '→稳定'}")
    print(f"  精确率趋势: {precision_trend:+.6f} {'↑上升' if precision_trend > 0.0001 else '↓下降' if precision_trend < -0.0001 else '→稳定'}")
    print(f"  召回率趋势: {recall_trend:+.6f} {'↑上升' if recall_trend > 0.0001 else '↓下降' if recall_trend < -0.0001 else '→稳定'}")
    print()
    
    # 计算波动性
    def calculate_volatility(values):
        """计算波动性（标准差）"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    map50_vol = calculate_volatility(map50s)
    
    print(f"mAP50波动性: {map50_vol:.6f} {'(稳定)' if map50_vol < 0.01 else '(波动较大)'}")
    print()
    
    # 给出建议
    print("=" * 60)
    print("建议:")
    print("=" * 60)
    
    if map50s[-1] >= 0.95:
        print("✓ mAP50已达到0.95+，性能优秀")
    
    if map50_trend < -0.001:
        print("⚠ mAP50呈下降趋势，可能出现过拟合，建议停止训练")
    elif map50_trend < 0.0001:
        print("→ mAP50已稳定，继续训练提升空间有限")
    else:
        print("↑ mAP50仍在上升，可以继续训练但需密切监控")
    
    if map50_vol > 0.02:
        print("⚠ 指标波动较大，可能训练不稳定")
    
    print()
    print("注意：由于没有验证集，无法直接判断过拟合。")
    print("建议：如果指标已稳定或开始下降，可以提前停止训练。")

if __name__ == "__main__":
    results_file = Path("models/boiling_scooping_detection4/results.csv")
    analyze_training_trend(results_file)

