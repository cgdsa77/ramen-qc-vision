"""
调试评分系统：检查特征值和阈值映射
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.stretch_scorer import StretchScorer
from src.features.image_feature_extractor import ImageFeatureExtractor
import cv2

# 加载评分规则
scorer = StretchScorer()
rules = scorer.rules

print("="*60)
print("评分阈值分析")
print("="*60)

# 显示各属性的阈值
for attr, thresholds in rules['thresholds'].items():
    print(f"\n{attr}:")
    print(f"  excellent: {thresholds['excellent']:.2f}")
    print(f"  good: {thresholds['good']:.2f}")
    print(f"  fair: {thresholds['fair']:.2f}")
    print(f"  poor: {thresholds['poor']:.2f}")
    print(f"  mean: {thresholds['mean']:.2f}")

print("\n" + "="*60)
print("测试特征提取值范围")
print("="*60)

# 测试图像
test_image_path = project_root / "data" / "processed" / "抻面" / "cm1" / "cm1_00001.jpg"
if test_image_path.exists():
    image = cv2.imread(str(test_image_path.resolve()), cv2.IMREAD_COLOR)
    if image is not None:
        extractor = ImageFeatureExtractor()
        h, w = image.shape[:2]
        
        # 测试不同类别的特征提取
        test_detections = [
            {'class': 'noodle_rope', 'xyxy': [w*0.2, h*0.3, w*0.8, h*0.7], 'conf': 0.9},
            {'class': 'hand', 'xyxy': [w*0.1, h*0.2, w*0.4, h*0.6], 'conf': 0.85},
            {'class': 'noodle_bundle', 'xyxy': [w*0.3, h*0.5, w*0.7, h*0.9], 'conf': 0.88},
        ]
        
        for det in test_detections:
            features = extractor.extract_features(image, det)
            print(f"\n{det['class']} 特征值:")
            for attr, value in features.items():
                # 检查这个值会被映射到多少分
                if attr in rules['thresholds']:
                    thresholds = rules['thresholds'][attr]
                    if value >= thresholds['excellent']:
                        score = 5
                    elif value >= thresholds['good']:
                        score = 4
                    elif value >= thresholds['fair']:
                        score = 3
                    elif value >= thresholds['poor']:
                        score = 2
                    else:
                        score = 1
                    print(f"  {attr}: {value:.2f} -> 评分: {score} (阈值: poor={thresholds['poor']:.2f}, fair={thresholds['fair']:.2f})")
                else:
                    print(f"  {attr}: {value:.2f} (无阈值配置)")

print("\n" + "="*60)
print("问题诊断")
print("="*60)
print("如果特征值远低于 'poor' 阈值，说明：")
print("1. 特征提取算法返回的值范围与手动评分数据（1-5分）不匹配")
print("2. 需要将特征提取的原始值映射到与手动评分数据相同的范围")
print("3. 或者调整特征提取算法，使其返回的值在合理范围内")

