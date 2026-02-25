"""
从 cm1~cm3 提取手部关键点序列，输出 JSON 格式，用于前端动态展示。
基于 hand 标签进行 ROI 检测，不生成伪骨架，只记录真实检测结果。
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
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
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
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        return landmarker, vision
    except Exception as e:
        print(f"[错误] 创建 HandLandmarker 失败: {e}")
        return None, None


def parse_yolo_label(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    return cls, cx, cy, w, h


def bbox_to_xyxy(cx, cy, w, h, img_w, img_h, margin=0.1):
    bw = w * img_w
    bh = h * img_h
    x = cx * img_w
    y = cy * img_h
    x1 = x - bw / 2
    y1 = y - bh / 2
    x2 = x + bw / 2
    y2 = y + bh / 2
    # margin
    mx = margin * bw
    my = margin * bh
    x1 -= mx
    y1 -= my
    x2 += mx
    y2 += my
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)
    return int(x1), int(y1), int(x2), int(y2)


def run_hand_on_roi(frame, bbox, landmarker, vision):
    import mediapipe as mp
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # 多尺度/扩展尝试，提升召回
    scales = [1.0, 1.2, 1.4, 1.6]
    for s in scales:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = (x2 - x1) * s
        bh = (y2 - y1) * s
        nx1 = int(max(0, cx - bw / 2))
        ny1 = int(max(0, cy - bh / 2))
        nx2 = int(min(W - 1, cx + bw / 2))
        ny2 = int(min(H - 1, cy + bh / 2))
        roi = frame[ny1:ny2, nx1:nx2]
        if roi.size == 0:
            continue
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            h, w = roi.shape[:2]
            kps = []
            for kp in hand:
                visibility = getattr(kp, "visibility", None)
                kps.append({
                    "x": float(nx1 + kp.x * w),
                    "y": float(ny1 + kp.y * h),
                    "z": float(kp.z),
                    "confidence": float(visibility if visibility is not None else 1.0),
                })
            return kps
    return None


def detect_full_frame(frame, landmarker, vision):
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return []
    h, w = frame.shape[:2]
    hands = []
    for hand in result.hand_landmarks:
        kps = []
        for kp in hand:
            visibility = getattr(kp, "visibility", None)
            kps.append({
                "x": float(kp.x * w),
                "y": float(kp.y * h),
                "z": float(kp.z),
                "confidence": float(visibility if visibility is not None else 1.0),
            })
        hands.append(kps)
    return hands


def get_video_fps(video_path: Path) -> float:
    """获取视频帧率"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0  # 默认值
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def process_video(name: str, landmarker, vision):
    img_dir = project_root / "data" / "processed" / "抻面" / name
    label_dir = project_root / "data" / "labels" / "抻面" / name
    output_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试找到原始视频文件以获取 FPS
    video_path = project_root / "data" / "videos" / "抻面" / f"{name}.mp4"
    if not video_path.exists():
        video_path = project_root / "data" / "videos" / f"{name}.mp4"
    fps = get_video_fps(video_path) if video_path.exists() else 30.0

    img_files = sorted(img_dir.glob("*.jpg"))
    total = len(img_files)
    
    frames_data = []
    missing_count = 0

    for img_path in img_files:
        stem = img_path.stem
        # 从文件名提取帧号（例如：cm1_00001 -> 1）
        try:
            frame_index = int(stem.split('_')[-1]) - 1  # 从0开始
        except:
            frame_index = len(frames_data)
        
        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # 读 hand 标签
        hand_boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_label(line)
            if not parsed:
                continue
            cls, cx, cy, w, h = parsed
            if cls != 0:  # 只处理 hand (class 0)
                continue
            bbox = bbox_to_xyxy(cx, cy, w, h, W, H, margin=0.1)
            hand_boxes.append(bbox)

        # 如果没有 hand 标签，跳过这一帧（不记录）
        if not hand_boxes:
            continue

        # 全图检测作为备用
        hands_ff = detect_full_frame(img, landmarker, vision)

        # 为每个 hand 标签检测关键点
        hands_detected = []
        for bbox_idx, bbox in enumerate(hand_boxes):
            kps = run_hand_on_roi(img, bbox, landmarker, vision)
            
            # 如果 ROI 失败，尝试全图匹配最近手
            if kps is None and hands_ff:
                bx1, by1, bx2, by2 = bbox
                bc = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
                best = None
                best_dist = 1e9
                for hand in hands_ff:
                    cx = np.mean([p["x"] for p in hand])
                    cy = np.mean([p["y"] for p in hand])
                    dist = np.linalg.norm(bc - np.array([cx, cy]))
                    # 要求手中心在bbox内
                    in_box = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
                    if in_box and dist < best_dist:
                        best = hand
                        best_dist = dist
                if best:
                    kps = best

            if kps is not None:
                hands_detected.append({
                    "id": bbox_idx,
                    "keypoints": kps
                })
            else:
                missing_count += 1

        # 记录这一帧的数据（即使 hands_detected 为空也记录，前端可以处理）
        frames_data.append({
            "frame_index": frame_index,
            "hands": hands_detected
        })

        if len(frames_data) % 100 == 0 and len(frames_data) > 0:
            print(f"[{name}] 已处理 {len(frames_data)}/{total} 帧")

    # 构建输出 JSON
    output_data = {
        "video": name,
        "fps": fps,
        "total_frames": len(frames_data),
        "frames": frames_data
    }

    # 保存 JSON
    output_file = output_dir / f"hand_keypoints_{name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[完成] {name}: 处理 {len(frames_data)} 帧，缺失检测 {missing_count} 个 hand 标签")
    print(f"  => 输出: {output_file}")
    return output_file


def main():
    print("=" * 60)
    print("提取手部关键点序列（JSON格式，用于前端动态展示）")
    print("=" * 60)
    landmarker, vision = load_hand_landmarker()
    if landmarker is None:
        sys.exit(1)

    output_files = []
    for name in ["cm1", "cm2", "cm3"]:
        output_file = process_video(name, landmarker, vision)
        output_files.append(output_file)

    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)
    print(f"输出目录: data/scores/抻面/hand_keypoints")
    print("\n说明：")
    print("  - 每个视频生成一个 JSON 文件，包含逐帧的手部关键点数据")
    print("  - 如果某帧的 hand 标签未能检测到关键点，hands 数组为空")
    print("  - 前端可以根据 frame_index 和视频时间戳进行同步显示")


if __name__ == "__main__":
    main()
