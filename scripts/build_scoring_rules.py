#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
建立评分规则和阈值
基于标准数据集分析评分数据，确定各属性的评分阈值和规则
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent

def load_standard_dataset():
    """加载标准数据集"""
    dataset_file = project_root / "data" / "scores" / "抻面" / "standard_dataset.json"
    
    if not dataset_file.exists():
        print(f"[错误] 标准数据集不存在: {dataset_file}")
        return None
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    return dataset

def analyze_scores(dataset):
    """分析评分数据，计算统计信息"""
    if not dataset:
        return None
    
    # 收集所有评分
    attribute_scores = defaultdict(list)
    
    for video_name, frames in dataset.get('videos', {}).items():
        for frame_data in frames:
            scores = frame_data.get('scores', {})
            for det_key, det_scores in scores.items():
                for attr, value in det_scores.items():
                    if attr != 'notes' and isinstance(value, (int, float)):
                        attribute_scores[attr].append(value)
    
    # 计算统计信息
    stats = {}
    for attr, scores_list in attribute_scores.items():
        if not scores_list:
            continue
        
        scores_array = np.array(scores_list)
        stats[attr] = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75)),
            'count': len(scores_list),
            'distribution': {
                int(score): int(np.sum(scores_array == score))
                for score in range(1, 6)
            }
        }
    
    return stats

def determine_thresholds(stats):
    """基于统计数据确定评分阈值"""
    thresholds = {}
    
    for attr, stat in stats.items():
        mean = stat['mean']
        std = stat['std']
        
        # 基于均值和标准差确定阈值
        # 优秀(5分): >= mean + 0.5*std
        # 良好(4分): mean - 0.5*std <= score < mean + 0.5*std
        # 一般(3分): mean - 1.5*std <= score < mean - 0.5*std
        # 较差(2分): mean - 2*std <= score < mean - 1.5*std
        # 差(1分): < mean - 2*std
        
        thresholds[attr] = {
            'excellent': max(4.5, mean + 0.5 * std),  # 5分阈值
            'good': max(3.5, mean - 0.5 * std),       # 4分阈值
            'fair': max(2.5, mean - 1.5 * std),      # 3分阈值
            'poor': max(1.5, mean - 2 * std),        # 2分阈值
            'mean': mean,
            'std': std
        }
    
    return thresholds

def build_scoring_rules(dataset):
    """建立评分规则"""
    print("=" * 60)
    print("建立评分规则和阈值")
    print("=" * 60)
    
    # 分析评分数据
    print("\n1. 分析评分数据...")
    stats = analyze_scores(dataset)
    
    if not stats:
        print("[错误] 无法分析评分数据")
        return None
    
    print(f"   找到 {len(stats)} 个评分属性")
    
    # 显示统计信息
    print("\n2. 评分统计信息:")
    print("-" * 60)
    for attr, stat in sorted(stats.items()):
        print(f"\n{attr}:")
        print(f"  平均分: {stat['mean']:.2f}")
        print(f"  标准差: {stat['std']:.2f}")
        print(f"  中位数: {stat['median']:.2f}")
        print(f"  评分数量: {stat['count']}")
        print(f"  分数分布: {stat['distribution']}")
    
    # 确定阈值
    print("\n3. 确定评分阈值...")
    thresholds = determine_thresholds(stats)
    
    print("\n评分阈值:")
    print("-" * 60)
    for attr, thresh in sorted(thresholds.items()):
        print(f"\n{attr}:")
        print(f"  优秀(5分): >= {thresh['excellent']:.2f}")
        print(f"  良好(4分): {thresh['good']:.2f} ~ {thresh['excellent']:.2f}")
        print(f"  一般(3分): {thresh['fair']:.2f} ~ {thresh['good']:.2f}")
        print(f"  较差(2分): {thresh['poor']:.2f} ~ {thresh['fair']:.2f}")
        print(f"  差(1分): < {thresh['poor']:.2f}")
    
    # 建立评分规则
    rules = {
        'version': '1.0',
        'stage': '抻面',
        'thresholds': thresholds,
        'statistics': stats,
        'weights': {
            'noodle_rope': {
                'thickness': 0.30,
                'elasticity': 0.25,
                'gloss': 0.20,
                'integrity': 0.25
            },
            'hand': {
                'position': 0.25,
                'action': 0.30,
                'angle': 0.20,
                'coordination': 0.25
            },
            'noodle_bundle': {
                'tightness': 0.50,
                'uniformity': 0.50
            }
        },
        'overall_weights': {
            'noodle_rope': 0.60,
            'hand': 0.30,
            'noodle_bundle': 0.10
        }
    }
    
    # 保存评分规则
    rules_file = project_root / "data" / "scores" / "抻面" / "scoring_rules.json"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(rules_file, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评分规则已保存到: {rules_file}")
    
    return rules

def main():
    # 加载标准数据集
    dataset = load_standard_dataset()
    
    if not dataset:
        print("[错误] 无法加载标准数据集")
        return
    
    print(f"\n标准数据集信息:")
    print(f"  阶段: {dataset.get('stage', '未知')}")
    print(f"  总帧数: {dataset.get('total_frames', 0)}")
    print(f"  视频数: {len(dataset.get('videos', {}))}")
    
    # 建立评分规则
    rules = build_scoring_rules(dataset)
    
    if rules:
        print("\n" + "=" * 60)
        print("评分规则建立完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 可以使用这些规则进行自动评分")
        print("2. 应用到新视频的评分")
        print("3. 根据实际效果调整阈值和权重")

if __name__ == "__main__":
    main()

