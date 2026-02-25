"""
从视频文件直接提取手部关键点，生成JSON格式
用于前端实时动态显示（无延迟）
支持抻面和下面及捞面视频
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_hand_landmarker():
    """加载MediaPipe HandLandmarker"""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
    except ImportError:
        print("[错误] 未安装 mediapipe，请先 pip install mediapipe")
        return None, None

    model_dir = project_root / "weights" / "mediapipe"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    if not model_path.exists():
        try:
            print(f"[INFO] 下载模型: {model_url}")
            import requests
            resp = requests.get(model_url, timeout=30)
            resp.raise_for_status()
            model_path.write_bytes(resp.content)
            print(f"[OK] hand_landmarker 已保存: {model_path}")
        except Exception as e:
            print(f"[错误] 模型下载失败: {e}")
            return None, None

    try:
        base_options = BaseOptions(model_asset_path=str(model_path))
        # 降低检测阈值以提高召回率
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.3,   # 降低检测阈值（默认0.5）
            min_hand_presence_confidence=0.3,    # 降低存在阈值（默认0.5）
            min_tracking_confidence=0.3,         # 降低跟踪阈值（默认0.5）
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        print("[OK] MediaPipe HandLandmarker 加载成功（低阈值模式，提高召回率）")
        return landmarker, vision
    except Exception as e:
        print(f"[错误] 创建 HandLandmarker 失败: {e}")
        return None, None


def detect_hands_in_frame(frame: np.ndarray, landmarker, vision_module) -> List[Dict[str, Any]]:
    """检测单帧中的手部关键点"""
    import mediapipe as mp
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    
    if not result.hand_landmarks:
        return []
    
    H, W = frame.shape[:2]
    hands_data = []
    
    for hand_idx, hand in enumerate(result.hand_landmarks):
        keypoints = []
        for kp in hand:
            visibility = getattr(kp, "visibility", None)
            keypoints.append({
                "x": float(kp.x * W),
                "y": float(kp.y * H),
                "z": float(kp.z),
                "confidence": float(visibility if visibility is not None else 1.0)
            })
        hands_data.append({
            "id": hand_idx,
            "keypoints": keypoints
        })
    
    return hands_data


def process_video(video_path: Path, output_dir: Path, video_name: str = None) -> Path:
    """处理单个视频，提取所有帧的手部关键点"""
    if not video_path.exists():
        print(f"[错误] 视频文件不存在: {video_path}")
        return None
    
    if video_name is None:
        video_name = video_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n处理视频: {video_name}")
    print(f"  路径: {video_path}")
    print(f"  分辨率: {width}x{height}")
    print(f"  总帧数: {total_frames}, FPS: {fps:.2f}")
    
    # 加载MediaPipe模型
    landmarker, vision_module = load_hand_landmarker()
    if landmarker is None:
        cap.release()
        return None
    
    frames_data = []
    frame_index = 0
    detected_count = 0
    missing_count = 0
    
    print("开始处理...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测手部关键点
        hands = detect_hands_in_frame(frame, landmarker, vision_module)
        
        if hands:
            detected_count += 1
        else:
            missing_count += 1
        
        frames_data.append({
            "frame_index": frame_index,
            "hands": hands
        })
        
        frame_index += 1
        
        # 进度显示
        if frame_index % 30 == 0:
            progress = (frame_index / total_frames) * 100
            print(f"  进度: {frame_index}/{total_frames} ({progress:.1f}%), "
                  f"检测到: {detected_count}, 缺失: {missing_count}")
    
    cap.release()
    
    # 构建输出数据
    output_data = {
        "video": video_name,
        "fps": float(fps),
        "total_frames": len(frames_data),
        "width": width,
        "height": height,
        "detected_frames": detected_count,
        "missing_frames": missing_count,
        "frames": frames_data
    }
    
    # 保存JSON
    output_file = output_dir / f"hand_keypoints_{video_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] {video_name}")
    print(f"  总帧数: {len(frames_data)}")
    print(f"  检测到骨架: {detected_count} 帧 ({detected_count/len(frames_data)*100:.1f}%)")
    print(f"  缺失骨架: {missing_count} 帧 ({missing_count/len(frames_data)*100:.1f}%)")
    print(f"  输出文件: {output_file}")
    
    return output_file


def main():
    """主函数：处理所有视频。抻面优先 data/raw/抻面，缺失时用 data/processed_videos/抻面。"""
    import argparse
    parser = argparse.ArgumentParser(description="从视频提取手部关键点，输出到 data/scores/抻面/hand_keypoints/")
    parser.add_argument("--video", type=str, default=None, help="仅处理指定视频名（如 cm8），否则处理所有")
    args = parser.parse_args()
    
    print("=" * 60)
    print("从视频提取手部关键点（用于实时动态显示）")
    print("=" * 60)
    
    stretch_output_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    stretch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 抻面：先 raw，再补 processed_videos 中未出现的（如 cm8 仅在 processed 时）
    stretch_dir = project_root / "data" / "raw" / "抻面"
    processed_stretch_dir = project_root / "data" / "processed_videos" / "抻面"
    stretch_videos = []
    seen_stems = set()
    if stretch_dir.exists():
        for p in stretch_dir.glob("*.mp4"):
            stem = p.stem
            if args.video and stem != args.video:
                continue
            seen_stems.add(stem)
            stretch_videos.append((p, stretch_output_dir, stem))
    if processed_stretch_dir.exists():
        for p in processed_stretch_dir.glob("*.mp4"):
            stem = p.stem
            if args.video and stem != args.video:
                continue
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            stretch_videos.append((p, stretch_output_dir, stem))
    
    # 处理下面及捞面视频
    boiling_dir = project_root / "data" / "raw" / "下面及捞面"
    boiling_output_dir = project_root / "data" / "scores" / "下面及捞面" / "hand_keypoints"
    boiling_output_dir.mkdir(parents=True, exist_ok=True)
    boiling_videos = []
    if boiling_dir.exists():
        for p in boiling_dir.glob("*.mp4"):
            if args.video and p.stem != args.video:
                continue
            boiling_videos.append(p)
    
    all_videos = []
    if stretch_videos:
        print(f"\n找到 {len(stretch_videos)} 个抻面视频")
        for item in stretch_videos:
            all_videos.append(item)  # (path, output_dir, video_name=stem)
    
    if boiling_videos:
        print(f"\n找到 {len(boiling_videos)} 个下面及捞面视频")
        for video in boiling_videos:
            all_videos.append((video, boiling_output_dir, "下面及捞面"))
    
    if not all_videos:
        print("\n[警告] 未找到任何视频文件")
        print("  请确保视频在 data/raw/抻面、data/processed_videos/抻面 或 data/raw/下面及捞面")
        return
    
    print(f"\n总共需要处理 {len(all_videos)} 个视频")
    print("=" * 60)
    
    output_files = []
    for i, item in enumerate(all_videos, 1):
        if len(item) == 3 and item[2] != "下面及捞面":
            video_path, output_dir, video_name = item
            stage_label = "抻面"
        else:
            video_path, output_dir, _ = item
            video_name = None
            stage_label = "下面及捞面"
        print(f"\n[{i}/{len(all_videos)}] 处理 {stage_label} 视频: {video_path.name}")
        output_file = process_video(video_path, output_dir, video_name=video_name)
        if output_file:
            output_files.append(output_file)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"成功处理 {len(output_files)} 个视频")
    print("\n输出目录：")
    print(f"  抻面: {stretch_output_dir}")
    print(f"  下面及捞面: {boiling_output_dir}")
    print("\n说明：")
    print("  - 每个视频生成一个 JSON 文件，包含逐帧的手部关键点数据")
    print("  - 前端可以直接读取这些 JSON 文件，实现无延迟的实时显示")
    print("  - 如果某帧检测不到手部，hands 数组为空（前端会跳过显示）")


if __name__ == "__main__":
    main()
