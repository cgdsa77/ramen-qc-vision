"""
详细调试评分系统：检查特征值提取和校准过程
"""
import sys
from pathlib import Path
import json
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.stretch_scorer import StretchScorer
from src.features.image_feature_extractor import ImageFeatureExtractor

# 初始化
scorer = StretchScorer()
extractor = ImageFeatureExtractor()
rules = scorer.rules

print("="*70)
print("特征提取和校准详细调试")
print("="*70)

# 找一个测试图像
test_image_path = project_root / "data" / "processed" / "抻面" / "cm1"
if not test_image_path.exists():
    # 尝试其他路径
    test_image_path = project_root / "data" / "processed" / "抻面"
    
if test_image_path.is_dir():
    # 找到第一个jpg文件
    jpg_files = list(test_image_path.rglob("*.jpg"))
    if jpg_files:
        test_image_path = jpg_files[0]
        print(f"\n使用测试图像: {test_image_path}")
    else:
        print("\n未找到测试图像，使用模拟数据")
        test_image_path = None
else:
    print(f"\n使用测试图像: {test_image_path}")

# 模拟检测框
h, w = 640, 480
test_detections = [
    {'class': 'noodle_rope', 'xyxy': [w*0.2, h*0.3, w*0.8, h*0.7], 'conf': 0.9},
    {'class': 'hand', 'xyxy': [w*0.1, h*0.2, w*0.4, h*0.6], 'conf': 0.85},
    {'class': 'noodle_bundle', 'xyxy': [w*0.3, h*0.5, w*0.7, h*0.9], 'conf': 0.88},
]

# 如果找到图像，读取它
if test_image_path and test_image_path.exists():
    # 使用OpenCV读取（处理中文路径）
    image_array = np.fromfile(str(test_image_path.resolve()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is not None:
        h, w = image.shape[:2]
        print(f"图像尺寸: {w}x{h}")
        
        # 更新检测框坐标为实际图像尺寸
        test_detections = [
            {'class': 'noodle_rope', 'xyxy': [int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.7)], 'conf': 0.9},
            {'class': 'hand', 'xyxy': [int(w*0.1), int(h*0.2), int(w*0.4), int(h*0.6)], 'conf': 0.85},
            {'class': 'noodle_bundle', 'xyxy': [int(w*0.3), int(h*0.5), int(w*0.7), int(h*0.9)], 'conf': 0.88},
        ]
else:
    print("\n使用模拟图像（640x480）")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (128, 128, 128)  # 灰色背景

print("\n" + "="*70)
print("1. 原始特征提取值")
print("="*70)

for det in test_detections:
    class_name = det['class']
    print(f"\n【{class_name}】")
    
    try:
        # 提取原始特征
        features = extractor.extract_features(image, det)
        
        print(f"  原始特征值:")
        for attr, value in features.items():
            print(f"    {attr}: {value:.4f}")
        
        # 校准特征值
        calibrated = scorer._calibrate_features(features, class_name)
        print(f"\n  校准后特征值:")
        for attr, cal_value in calibrated.items():
            # 查找阈值
            if attr in rules['thresholds']:
                thresholds = rules['thresholds'][attr]
                mean_value = thresholds.get('mean', 3.0)
                poor_threshold = thresholds.get('poor', 2.0)
                
                # 计算评分
                score = scorer.score_attribute(attr, cal_value)
                print(f"    {attr}: {cal_value:.4f} -> 评分: {score} (mean={mean_value:.2f}, poor={poor_threshold:.2f})")
            else:
                print(f"    {attr}: {cal_value:.4f} (无阈值配置)")
        
        # 计算加权总分
        scores_dict = {}
        if class_name == 'noodle_rope':
            scores_dict = {
                'thickness': scorer.score_attribute('thickness', calibrated.get('thickness', 3.0)),
                'elasticity': scorer.score_attribute('elasticity', calibrated.get('elasticity', 3.0)),
                'gloss': scorer.score_attribute('gloss', calibrated.get('gloss', 3.0)),
                'integrity': scorer.score_attribute('integrity', calibrated.get('integrity', 3.0)),
            }
        elif class_name == 'hand':
            scores_dict = {
                'position': scorer.score_attribute('position', calibrated.get('position', 3.0)),
                'action': scorer.score_attribute('action', calibrated.get('action', 3.0)),
                'angle': scorer.score_attribute('angle', calibrated.get('angle', 3.0)),
                'coordination': scorer.score_attribute('coordination', calibrated.get('coordination', 3.0)),
            }
        elif class_name == 'noodle_bundle':
            scores_dict = {
                'tightness': scorer.score_attribute('tightness', calibrated.get('tightness', 3.0)),
                'uniformity': scorer.score_attribute('uniformity', calibrated.get('uniformity', 3.0)),
            }
        
        weighted_score = scorer.calculate_weighted_score(scores_dict, class_name)
        print(f"\n  加权总分: {weighted_score:.2f}")
        print(f"  各属性评分: {scores_dict}")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("2. 校准策略分析")
print("="*70)
print("""
如果原始特征值 < 1.0：
  - 使用线性映射：0 -> 1.0, 1.0 -> mean（约3-4分）
  - 公式：calibrated = 1.0 + (mean - 1.0) * raw_value

如果原始特征值在 1.0-5.0 范围内：
  - 直接使用

如果原始特征值 > 5.0：
  - 截断到5.0
""")

print("\n" + "="*70)
print("3. 建议优化方向")
print("="*70)
print("""
如果校准后的值仍然偏低，可能需要：
1. 调整特征提取算法，使其返回的值更接近手动评分的范围
2. 改进校准策略，使用更复杂的映射（如基于统计分布）
3. 检查视频质量是否确实较低
4. 对比手动评分数据集，验证特征提取算法的准确性
""")
