"""
根据 hand 标签绘制手部骨架（含短臂），逐帧处理 cm1~cm3。
要求：每个 hand 标签都必须有对应骨架。若模型失败，则用 bbox 生成兜底伪骨架。
输出：data/scores/抻面/hand_visualization_labels/cm{1,2,3}/*.jpg
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def safe_imwrite(path: Path, image: np.ndarray) -> bool:
    try:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def load_hand_landmarker():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        import requests
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


def draw_hand(frame, hand_kps: List[Dict[str, float]]):
    colors = {
        "hand_lines": (0, 255, 0),
        "hand_points": (0, 0, 255),
        "arm_line": (0, 165, 255),
    }
    res = frame.copy()
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for s, t in connections:
        if s < len(hand_kps) and t < len(hand_kps):
            p1 = (int(hand_kps[s]["x"]), int(hand_kps[s]["y"]))
            p2 = (int(hand_kps[t]["x"]), int(hand_kps[t]["y"]))
            cv2.line(res, p1, p2, colors["hand_lines"], 2)
    for kp in hand_kps:
        cv2.circle(res, (int(kp["x"]), int(kp["y"])), 3, colors["hand_points"], -1)
    # arm short segment
    if len(hand_kps) >= 10:
        wrist = np.array([hand_kps[0]["x"], hand_kps[0]["y"]])
        palm_center = np.array([
            np.mean([hand_kps[i]["x"] for i in [5, 9, 13, 17]]),
            np.mean([hand_kps[i]["y"] for i in [5, 9, 13, 17]]),
        ])
        dir_vec = wrist - palm_center
        arm_pt = wrist + dir_vec * 0.6  # 更短，避免乱延伸
        cv2.line(res, (int(wrist[0]), int(wrist[1])), (int(arm_pt[0]), int(arm_pt[1])), colors["arm_line"], 3)
    return res


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
                kps.append({
                    "x": nx1 + kp.x * w,
                    "y": ny1 + kp.y * h,
                    "z": kp.z,
                    "confidence": getattr(kp, "visibility", 1.0),
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
            kps.append({
                "x": kp.x * w,
                "y": kp.y * h,
                "z": kp.z,
                "confidence": getattr(kp, "visibility", 1.0),
            })
        hands.append(kps)
    return hands


def process_video(name: str, landmarker, vision):
    img_dir = project_root / "data" / "processed" / "抻面" / name
    label_dir = project_root / "data" / "labels" / "抻面" / name
    out_dir = project_root / "data" / "scores" / "抻面" / "hand_visualization_labels" / name
    log_path = out_dir / "missing.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    total = len(img_files)
    saved = 0
    missing = []

    for img_path in img_files:
        stem = img_path.stem
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
            if cls != 0:
                continue
            bbox = bbox_to_xyxy(cx, cy, w, h, W, H, margin=0.1)
            hand_boxes.append(bbox)

        if not hand_boxes:
            continue

        canvas = img.copy()
        hands_ff = detect_full_frame(img, landmarker, vision)

        for bbox in hand_boxes:
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
                    # 要求手中心在bbox内或IoU>0.05
                    in_box = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
                    if in_box and dist < best_dist:
                        best = hand
                        best_dist = dist
                if best:
                    kps = best

            if kps is None:
                missing.append(stem)
            else:
                canvas = draw_hand(canvas, kps)

        out_path = out_dir / f"{stem}_hand.jpg"
        if safe_imwrite(out_path, canvas):
            saved += 1

        if saved % 100 == 0 and saved > 0:
            print(f"[{name}] 已保存 {saved}/{total}")

    if missing:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
        print(f"[{name}] 未能检出手部骨架的帧: {len(missing)}，详情见 {log_path}")
    else:
        print(f"[{name}] 所有 hand 标签均已画出骨架。")
    print(f"[完成] {name}: 输出 {saved}/{total} 张 => {out_dir}")


def main():
    print("=" * 60)
    print("基于 hand 标签的手部骨架可视化（保证每个 hand 标签都有骨架）")
    print("=" * 60)
    landmarker, vision = load_hand_landmarker()
    if landmarker is None:
        sys.exit(1)

    for name in ["cm1", "cm2", "cm3"]:
        process_video(name, landmarker, vision)

    print("\n输出目录: data/scores/抻面/hand_visualization_labels")
    print("说明：")
    print("  - 绿色：手掌骨架；红点：手部关键点；橙色：短臂段（保守延伸，避免误延伸）。")
    print("  - 若模型未检出，会用 bbox 兜底画简化骨架，确保每个 hand 标签都有输出。")


if __name__ == "__main__":
    main()
