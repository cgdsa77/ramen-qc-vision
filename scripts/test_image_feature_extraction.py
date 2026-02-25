"""
测试图像特征提取功能
"""
import sys
from pathlib import Path
import cv2

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.image_feature_extractor import ImageFeatureExtractor


def test_feature_extraction():
    """测试特征提取"""
    extractor = ImageFeatureExtractor()
    
    # 测试图像路径
    test_image_path = project_root / "data" / "processed" / "抻面" / "cm1" / "cm1_00001.jpg"
    
    if not test_image_path.exists():
        print(f"[错误] 测试图像不存在: {test_image_path}")
        return
    
    # 读取图像（使用绝对路径避免编码问题）
    image_path_str = str(test_image_path.resolve())
    image = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[错误] 无法读取图像: {image_path_str}")
        print(f"文件是否存在: {test_image_path.exists()}")
        # 尝试使用numpy读取
        try:
            import numpy as np
            with open(test_image_path, 'rb') as f:
                image_bytes = f.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                print("[错误] 使用imdecode也无法读取图像")
                return
        except Exception as e:
            print(f"[错误] 尝试其他方式读取失败: {e}")
            return
    
    print(f"[OK] 成功读取图像: {test_image_path}")
    print(f"图像尺寸: {image.shape}")
    
    # 模拟检测结果
    h, w = image.shape[:2]
    
    # 测试 noodle_rope 特征提取
    print("\n" + "="*60)
    print("测试 noodle_rope 特征提取")
    print("="*60)
    noodle_detection = {
        'class': 'noodle_rope',
        'xyxy': [w*0.2, h*0.3, w*0.8, h*0.7],
        'conf': 0.9
    }
    noodle_features = extractor.extract_features(image, noodle_detection)
    print(f"提取的特征: {noodle_features}")
    
    # 测试 hand 特征提取
    print("\n" + "="*60)
    print("测试 hand 特征提取")
    print("="*60)
    hand_detection = {
        'class': 'hand',
        'xyxy': [w*0.1, h*0.2, w*0.4, h*0.6],
        'conf': 0.85
    }
    hand_features = extractor.extract_features(image, hand_detection)
    print(f"提取的特征: {hand_features}")
    
    # 测试 noodle_bundle 特征提取
    print("\n" + "="*60)
    print("测试 noodle_bundle 特征提取")
    print("="*60)
    bundle_detection = {
        'class': 'noodle_bundle',
        'xyxy': [w*0.3, h*0.5, w*0.7, h*0.9],
        'conf': 0.88
    }
    bundle_features = extractor.extract_features(image, bundle_detection)
    print(f"提取的特征: {bundle_features}")
    
    print("\n" + "="*60)
    print("特征提取测试完成")
    print("="*60)


if __name__ == "__main__":
    test_feature_extraction()

