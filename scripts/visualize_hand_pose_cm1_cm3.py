"""
仅可视化手部关键点（手部“骨架”），不绘制全身。
依赖：mediapipe>=0.10（tasks API），自动下载 hand_landmarker.task。
输出：data/scores/抻面/hand_visualization/{cm1,cm2,cm3} 下的示例帧。
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def safe_imwrite(path: Path, image: np.ndarray) -> bool:
    """兼容中文路径的安全写图函数"""
    try:
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def load_hand_landmarker():
    """加载 MediaPipe hand landmarker（tasks API），自动下载模型"""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        import requests
    except ImportError:
        print("[错误] 未安装 mediapipe，请先执行: pip install mediapipe")
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
            resp = requests.get(model_url, timeout=30)
            resp.raise_for_status()
            model_path.write_bytes(resp.content)
            print(f"[OK] hand_landmarker 模型已保存到: {model_path}")
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


def draw_hand_skeleton(frame: np.ndarray, hand_landmarks, colors=None) -> np.ndarray:
    """在帧上绘制单只手的骨架，并向上延伸短臂段"""
    if not colors:
        colors = {
            "hand_lines": (0, 255, 0),   # 手掌绿色
            "hand_points": (0, 0, 255),  # 手掌红色点
            "arm_line": (0, 165, 255),   # 手臂橙色
        }
    result = frame.copy()

    # Blaze Hand 21点的连接关系
    connections = [
        # 拇指
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 食指
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 中指
        (0, 9), (9, 10), (10, 11), (11, 12),
        # 无名指
        (0, 13), (13, 14), (14, 15), (15, 16),
        # 小指
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    # 绘制手掌骨架
    for s, t in connections:
        if s < len(hand_landmarks) and t < len(hand_landmarks):
            ls, lt = hand_landmarks[s], hand_landmarks[t]
            pt1 = (int(ls['x']), int(ls['y']))
            pt2 = (int(lt['x']), int(lt['y']))
            cv2.line(result, pt1, pt2, colors["hand_lines"], 2)

    # 绘制关键点
    for kp in hand_landmarks:
        x, y = int(kp['x']), int(kp['y'])
        cv2.circle(result, (x, y), 3, colors["hand_points"], -1)

    # 向上延伸短臂段（从手腕0点沿手掌法向反方向延长）
    if len(hand_landmarks) >= 10:
        wrist = hand_landmarks[0]
        # 手掌中心：四个MCP点平均（5,9,13,17）
        palm_center = np.array([
            np.mean([hand_landmarks[i]['x'] for i in [5, 9, 13, 17]]),
            np.mean([hand_landmarks[i]['y'] for i in [5, 9, 13, 17]])
        ])
        wrist_pt = np.array([wrist['x'], wrist['y']])
        # 方向：从掌心指向手腕，再延长 1.2 倍作为前臂短段
        dir_vec = wrist_pt - palm_center
        arm_pt = wrist_pt + dir_vec * 1.2
        pt1 = (int(wrist_pt[0]), int(wrist_pt[1]))
        pt2 = (int(arm_pt[0]), int(arm_pt[1]))
        cv2.line(result, pt1, pt2, colors["arm_line"], 3)

    return result


def run_hand_landmarker(frame: np.ndarray, landmarker, vision) -> List[List[Dict[str, Any]]]:
    """运行手部关键点检测，返回每只手的关键点列表"""
    import mediapipe as mp
    # 转 RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return []

    h, w = frame.shape[:2]
    hands = []
    for hand in result.hand_landmarks:
        kp_list = []
        for kp in hand:
            kp_list.append({
                "x": kp.x * w,
                "y": kp.y * h,
                "z": kp.z,
                "confidence": getattr(kp, "visibility", 1.0),
            })
        hands.append(kp_list)
    return hands


def visualize_video(video_path: Path, landmarker, vision, output_dir: Path, sample_count: int = 20, search_window: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[警告] 无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // sample_count)
    video_name = video_path.stem

    print(f"\n处理视频: {video_name}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 采样间隔: 每 {frame_interval} 帧提取一张")
    print(f"  - 目标帧数: {sample_count}")

    saved = 0
    frame_idx = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    while saved < sample_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            hands = run_hand_landmarker(frame, landmarker, vision)

            # 若当前帧未检测到手，向前看 search_window 帧尝试补一次（提升召回）
            if not hands:
                cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                for _ in range(search_window):
                    ret2, frame2 = cap.read()
                    if not ret2:
                        break
                    hands = run_hand_landmarker(frame2, landmarker, vision)
                    frame_idx += 1
                    if hands:
                        frame = frame2  # 用找到手的帧
                        break
                else:
                    # 没找到手，恢复指针到当前进度
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)

            if hands:
                frame_drawn = frame.copy()
                for hand_kps in hands:
                    frame_drawn = draw_hand_skeleton(frame_drawn, hand_kps)

                info = f"Frame {frame_idx}/{total_frames} | Hands: {len(hands)}"
                cv2.putText(frame_drawn, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out_file = output_dir / f"{video_name}_frame_{frame_idx:06d}_hand.jpg"
                if safe_imwrite(out_file, frame_drawn):
                    saved += 1
                    if saved % 5 == 0:
                        print(f"  已保存 {saved}/{sample_count} 张")

        frame_idx += 1

    cap.release()
    print(f"  [完成] 已保存 {saved} 张手部可视化帧到: {output_dir}")


def main():
    print("=" * 60)
    print("手部关键点可视化（仅手，不包含全身）")
    print("=" * 60)

    landmarker, vision = load_hand_landmarker()
    if landmarker is None:
        sys.exit(1)

    video_dir = project_root / "data" / "raw" / "抻面"
    output_root = project_root / "data" / "scores" / "抻面" / "hand_visualization"
    video_names = ["cm1", "cm2", "cm3"]

    for name in video_names:
        vp = video_dir / f"{name}.mp4"
        if not vp.exists():
            print(f"[警告] 缺少视频: {vp}")
            continue
        out_dir = output_root / name
        visualize_video(vp, landmarker, vision, out_dir, sample_count=12)

    print("\n输出目录:", output_root)
    print("提示：这里只画手部骨架线，避免出现脚、腿等误检。")


if __name__ == "__main__":
    main()
