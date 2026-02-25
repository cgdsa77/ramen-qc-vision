#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为下面及捞面（xl 视频）选择关键帧用于手动评分
仿照抻面部分，每个 xl 视频挑选一定数量的关键帧，写入 key_frames.json
"""
import sys
from pathlib import Path
import json
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
labels_base = project_root / "data" / "labels" / "下面及捞面"
processed_base = project_root / "data" / "processed" / "下面及捞面"
scores_base = project_root / "data" / "scores" / "下面及捞面"


def load_yolo_labels(label_path: Path, class_names: List[str]) -> List[Dict]:
    """加载 YOLO 格式标注，返回类别信息"""
    annotations = []
    if not label_path.exists():
        return annotations
    try:
        with open(label_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    annotations.append({"class": name, "class_id": cls_id})
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"[警告] 读取标注失败 {label_path}: {e}")
    return annotations


def select_keyframes(video_name: str, num_frames: int = 10) -> List[Dict[str, Any]]:
    """
    为下面及捞面视频选择关键帧
    策略：均匀分段，每段内优先选检测丰富（noodle_rope/hand/tools_noodle/soup_noodle 多样）的帧
    """
    labels_dir = labels_base / video_name
    processed_dir = processed_base / video_name

    if not labels_dir.exists():
        print(f"[错误] 标注目录不存在: {labels_dir}")
        return []

    classes_file = labels_dir / "classes.txt"
    if not classes_file.exists():
        print(f"[错误] 找不到 classes.txt: {classes_file}")
        return []

    class_names = []
    with open(classes_file, "r", encoding="utf-8-sig") as f:
        class_names = [line.strip() for line in f if line.strip()]

    label_files = sorted(
        [f for f in labels_dir.glob("*.txt") if f.name != "classes.txt"],
        key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0,
    )

    if not label_files:
        print(f"[警告] {video_name} 没有标注文件")
        return []

    frame_info = []
    for label_file in label_files:
        frame_name = label_file.stem + ".jpg"
        if not (processed_dir / frame_name).exists():
            continue
        annotations = load_yolo_labels(label_file, class_names)
        has_noodle_rope = any(a["class"] == "noodle_rope" for a in annotations)
        has_hand = any(a["class"] == "hand" for a in annotations)
        has_tools = any(a["class"] == "tools_noodle" for a in annotations)
        has_soup = any(a["class"] == "soup_noodle" for a in annotations)
        priority = sum([has_noodle_rope, has_hand, has_tools, has_soup]) * 2 + len(annotations)
        try:
            frame_index = int(label_file.stem.split("_")[-1])
        except (ValueError, IndexError):
            frame_index = 0
        frame_info.append({
            "frame": frame_name,
            "index": frame_index,
            "detections": len(annotations),
            "priority": priority,
        })

    if not frame_info:
        print(f"[警告] {video_name} 没有同时存在标注与图片的帧")
        return []

    total = len(frame_info)
    if total <= num_frames:
        selected = [{"frame": info["frame"], "index": info["index"], "detections": info["detections"]} for info in frame_info]
    else:
        segment_size = total / num_frames
        selected = []
        for i in range(num_frames):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size) if i < num_frames - 1 else total
            segment = frame_info[start:end]
            if segment:
                best = max(segment, key=lambda x: (x["priority"], x["detections"]))
                selected.append({
                    "frame": best["frame"],
                    "index": best["index"],
                    "detections": best["detections"],
                })

    return selected


def main():
    print("=" * 60)
    print("下面及捞面：为各 xl 视频选择关键帧")
    print("=" * 60)

    if not labels_base.exists():
        print(f"[错误] 标注根目录不存在: {labels_base}")
        return

    all_videos = sorted(
        [d.name for d in labels_base.iterdir() if d.is_dir() and d.name.startswith("xl")]
    )
    if not all_videos:
        print("未找到 xl 开头的视频目录")
        return

    print(f"\n视频列表: {', '.join(all_videos)}")
    print("每个视频默认选择 10 个关键帧（可改脚本 num_frames）\n")

    num_frames = 10
    all_keyframes = {}

    for video_name in all_videos:
        keyframes = select_keyframes(video_name, num_frames=num_frames)
        if keyframes:
            all_keyframes[video_name] = keyframes
            scores_dir = scores_base / video_name
            scores_dir.mkdir(parents=True, exist_ok=True)
            out_file = scores_dir / "key_frames.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(keyframes, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {video_name}: {len(keyframes)} 帧 -> {out_file}")
            for idx, kf in enumerate(keyframes[:5], 1):
                print(f"      {idx}. {kf['frame']} (检测数: {kf['detections']})")
            if len(keyframes) > 5:
                print(f"      ... 共 {len(keyframes)} 帧")

    summary_file = scores_base / "关键帧列表_下面及捞面.txt"
    scores_base.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("下面及捞面 关键帧列表\n")
        f.write("=" * 60 + "\n\n")
        for video_name, keyframes in sorted(all_keyframes.items()):
            f.write(f"\n视频: {video_name}\n")
            f.write("-" * 40 + "\n")
            for idx, kf in enumerate(keyframes, 1):
                f.write(f"  {idx:2d}. {kf['frame']} (index: {kf['index']}, 检测数: {kf['detections']})\n")
            f.write("\n")

    print("\n" + "=" * 60)
    print("✅ 关键帧选择完成")
    print("=" * 60)
    print(f"汇总: {summary_file}")
    print("\n下一步: 打开下面及捞面评分标注工具，勾选「仅显示关键帧」后进行评分。")


if __name__ == "__main__":
    main()
