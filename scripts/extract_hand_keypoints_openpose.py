"""
使用OpenPose提取手部关键点
注意：需要先安装OpenPose
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入OpenPose
try:
    import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("[错误] 未安装OpenPose")
    print("请先安装OpenPose：")
    print("  1. 下载OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
    print("  2. 安装pyopenpose: pip install pyopenpose")
    print("  3. 设置环境变量 OPENPOSE_DIR")
    print("\n或者使用替代方案：")
    print("  - MediaPipe（已安装）: 运行 scripts/extract_hand_keypoints_cm1_cm3.py")
    print("  - MMPose: pip install mmpose")


def init_openpose():
    """初始化OpenPose"""
    if not OPENPOSE_AVAILABLE:
        return None
    
    # OpenPose参数
    params = dict()
    params["model_folder"] = str(project_root / "weights" / "openpose")
    params["hand"] = True  # 启用手部检测
    params["hand_detector"] = 2  # 使用OpenPose手部检测器
    params["body"] = 0  # 禁用身体检测（只检测手部）
    params["face"] = False  # 禁用面部检测
    params["number_people_max"] = 1  # 最多检测1个人
    
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        print("[OK] OpenPose初始化成功")
        return opWrapper
    except Exception as e:
        print(f"[错误] OpenPose初始化失败: {e}")
        return None


def detect_hands_openpose(opWrapper, frame: np.ndarray) -> List[Dict[str, Any]]:
    """使用OpenPose检测手部关键点"""
    if opWrapper is None:
        return []
    
    try:
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        hands_detected = []
        
        # 检测左手
        if datum.handKeypoints[0] is not None and len(datum.handKeypoints[0]) > 0:
            for hand_idx, hand in enumerate(datum.handKeypoints[0]):
                keypoints = []
                for kp_idx, kp in enumerate(hand):
                    if len(kp) >= 3:
                        keypoints.append({
                            "x": float(kp[0]),
                            "y": float(kp[1]),
                            "z": float(kp[2]) if len(kp) > 2 else 0.0,
                            "confidence": 1.0  # OpenPose不提供confidence，设为1.0
                        })
                if keypoints:
                    hands_detected.append({
                        "id": hand_idx,
                        "hand_type": "left",
                        "keypoints": keypoints
                    })
        
        # 检测右手
        if datum.handKeypoints[1] is not None and len(datum.handKeypoints[1]) > 0:
            for hand_idx, hand in enumerate(datum.handKeypoints[1]):
                keypoints = []
                for kp_idx, kp in enumerate(hand):
                    if len(kp) >= 3:
                        keypoints.append({
                            "x": float(kp[0]),
                            "y": float(kp[1]),
                            "z": float(kp[2]) if len(kp) > 2 else 0.0,
                            "confidence": 1.0
                        })
                if keypoints:
                    hands_detected.append({
                        "id": len(hands_detected),
                        "hand_type": "right",
                        "keypoints": keypoints
                    })
        
        return hands_detected
    except Exception as e:
        print(f"[错误] OpenPose检测失败: {e}")
        return []


def process_video(video_name: str, opWrapper):
    """处理单个视频"""
    video_path = project_root / "data" / "raw" / "抻面" / f"{video_name}.mp4"
    if not video_path.exists():
        print(f"[错误] 视频文件不存在: {video_path}")
        return None
    
    output_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints_openpose"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_data = []
    frame_index = 0
    
    print(f"\n处理视频: {video_name}")
    print(f"总帧数: {total_frames}, FPS: {fps:.2f}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测手部关键点
        hands = detect_hands_openpose(opWrapper, frame)
        
        frames_data.append({
            "frame_index": frame_index,
            "hands": hands
        })
        
        frame_index += 1
        
        if frame_index % 30 == 0:
            print(f"  已处理 {frame_index}/{total_frames} 帧")
    
    cap.release()
    
    # 保存JSON
    output_data = {
        "video": video_name,
        "fps": fps,
        "total_frames": len(frames_data),
        "frames": frames_data
    }
    
    output_file = output_dir / f"hand_keypoints_{video_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[完成] {video_name}: 处理 {len(frames_data)} 帧")
    print(f"  => 输出: {output_file}")
    
    return output_file


def main():
    print("="*60)
    print("使用OpenPose提取手部关键点")
    print("="*60)
    
    if not OPENPOSE_AVAILABLE:
        print("\n[错误] OpenPose未安装")
        print("\n推荐使用替代方案：")
        print("  1. MediaPipe（已安装）: python scripts/extract_hand_keypoints_cm1_cm3.py")
        print("  2. MMPose: pip install mmpose")
        sys.exit(1)
    
    # 初始化OpenPose
    opWrapper = init_openpose()
    if opWrapper is None:
        sys.exit(1)
    
    # 处理视频
    videos = ["cm1", "cm2", "cm3"]
    output_files = []
    
    for video_name in videos:
        output_file = process_video(video_name, opWrapper)
        if output_file:
            output_files.append(output_file)
    
    print("\n" + "="*60)
    print("提取完成！")
    print("="*60)
    print(f"输出目录: data/scores/抻面/hand_keypoints_openpose")
    print(f"共处理 {len(output_files)} 个视频")


if __name__ == "__main__":
    main()
