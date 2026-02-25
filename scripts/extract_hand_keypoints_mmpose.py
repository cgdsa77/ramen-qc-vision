"""
使用MMPose提取手部关键点（推荐方案）
MMPose是专业姿态估计框架，精度高，安装简单
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入MMPose
try:
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("[错误] 未安装MMPose")
    print("请先安装: pip install mmpose mmcv mmengine")
    print("\n或运行: python scripts/setup_mmpose.py")


def init_mmpose():
    """初始化MMPose"""
    if not MMPOSE_AVAILABLE:
        return None
    
    try:
        # 使用RTMPose模型（轻量级，效果好）
        # 注意：MMPose主要用于身体姿态，手部检测建议使用MediaPipe
        # 这里提供一个示例，实际可能需要使用其他模型
        print("[INFO] MMPose主要用于身体姿态估计")
        print("[INFO] 手部检测建议使用MediaPipe（已安装）")
        print("[INFO] 或使用MMPose的手部模型（需要单独配置）")
        
        # 如果需要使用MMPose，可以这样初始化：
        # inferencer = MMPoseInferencer('rtmpose-m')
        # return inferencer
        
        return None
    except Exception as e:
        print(f"[错误] MMPose初始化失败: {e}")
        return None


def main():
    print("="*60)
    print("MMPose手部关键点提取")
    print("="*60)
    
    if not MMPOSE_AVAILABLE:
        print("\n[错误] MMPose未安装")
        print("\n推荐方案：")
        print("  1. 使用MediaPipe（已安装）: python scripts/extract_hand_keypoints_cm1_cm3.py")
        print("  2. 安装MMPose: python scripts/setup_mmpose.py")
        sys.exit(1)
    
    print("\n注意：MMPose主要用于身体姿态估计")
    print("手部检测建议使用MediaPipe（已安装）")
    print("\n如果必须使用MMPose，需要：")
    print("  1. 配置手部检测模型")
    print("  2. 或使用MMPose + MediaPipe混合方案")
    
    print("\n推荐：继续使用MediaPipe")
    print("  python scripts/extract_hand_keypoints_cm1_cm3.py")


if __name__ == "__main__":
    main()
