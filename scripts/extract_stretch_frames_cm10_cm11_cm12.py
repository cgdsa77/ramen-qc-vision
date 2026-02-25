#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅针对抻面 cm10、cm11、cm12 从 raw 抽帧到 processed，并确保 labels 目录有主 classes.txt。
不读取、不修改 cm1～cm7、cm9 及任何已有标注 .txt，保证原有数据不受影响。

- 视频来源: data/raw/抻面/cm10.mp4（或 .MP4）等
- 输出图片: data/processed/抻面/cm10/cm10_00001.jpg，命名与 cm1～cm7 一致（视频名_%05d.jpg）
- 抽帧规则: 2 fps（与 extract_frames.ps1 / extract_video_frames 一致），可用参数覆盖

用法:
  python scripts/extract_stretch_frames_cm10_cm11_cm12.py
  python scripts/extract_stretch_frames_cm10_cm11_cm12.py --fps 1.5
"""
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw" / "抻面"
processed_base = project_root / "data" / "processed" / "抻面"
labels_base = project_root / "data" / "labels" / "抻面"
main_classes_file = labels_base / "classes.txt"

# 仅处理这三个视频，不碰 cm1～cm7、cm8、cm9
TARGET_VIDEOS = ["cm10", "cm11", "cm12"]


def extract_frames_ffmpeg(video_path: Path, output_dir: Path, video_name: str, fps: float) -> int:
    """用 ffmpeg 抽帧，命名 video_name_%05d.jpg。返回成功提取的帧数。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / f"{video_name}_%05d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2", "-y",
        pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [警告] ffmpeg 执行失败，尝试用 OpenCV 抽帧: {e}")
        return extract_frames_opencv(video_path, output_dir, video_name, fps)
    except FileNotFoundError:
        print("  [提示] 未找到 ffmpeg，使用 OpenCV 抽帧")
        return extract_frames_opencv(video_path, output_dir, video_name, fps)
    return len(list(output_dir.glob("*.jpg")))


def extract_frames_opencv(video_path: Path, output_dir: Path, video_name: str, fps: float) -> int:
    """用 OpenCV 抽帧（无 ffmpeg 时备用）。"""
    try:
        import cv2
    except ImportError:
        print("  [错误] 未安装 opencv-python，请安装: pip install opencv-python")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [错误] 无法打开视频文件")
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(video_fps / fps))  # 每隔 interval 帧取一帧
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            saved += 1
            out_name = output_dir / f"{video_name}_{saved:05d}.jpg"
            cv2.imwrite(str(out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    cap.release()
    return saved


def ensure_labels_dir_and_classes(video_name: str) -> None:
    """仅当该视频在 TARGET_VIDEOS 内时：若 labels 子目录不存在则创建，并写入/覆盖 classes.txt（从主文件复制）。不删除、不修改任何已有 .txt 标注。"""
    if video_name not in TARGET_VIDEOS:
        return
    label_dir = labels_base / video_name
    label_dir.mkdir(parents=True, exist_ok=True)
    if not main_classes_file.exists():
        print(f"  [警告] 主 classes.txt 不存在，跳过同步: {main_classes_file}")
        return
    dest = label_dir / "classes.txt"
    shutil.copy2(main_classes_file, dest)
    print(f"  [OK] 已同步 classes.txt -> {label_dir.name}/classes.txt")


def main():
    parser = argparse.ArgumentParser(description="抻面 cm10/cm11/cm12 抽帧并准备 labels，不修改已有数据")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧帧率（默认 2）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的步骤，不写文件")
    args = parser.parse_args()

    if not raw_dir.exists():
        print(f"错误：raw 目录不存在: {raw_dir}")
        sys.exit(1)
    if not main_classes_file.exists():
        print(f"警告：主 classes.txt 不存在，labels 目录将无法同步类别: {main_classes_file}")

    print("仅处理以下三个视频（不修改 cm1～cm7、cm8、cm9 及任何已有标注）：")
    print("  " + ", ".join(TARGET_VIDEOS))
    print(f"抽帧规则: fps={args.fps}，输出命名: <视频名>_00001.jpg ...")
    if args.dry_run:
        print("[dry-run] 不写入文件")
    print()

    for video_name in TARGET_VIDEOS:
        # 查找视频文件（.mp4 或 .MP4）
        video_path = None
        for ext in (".mp4", ".MP4"):
            p = raw_dir / f"{video_name}{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            print(f"[跳过] {video_name}: raw 中未找到 {video_name}.mp4 / .MP4")
            continue

        out_img_dir = processed_base / video_name
        if args.dry_run:
            print(f"[dry-run] 将抽帧: {video_path.name} -> {out_img_dir}")
            ensure_labels_dir_and_classes(video_name)
            continue

        print(f"正在抽帧: {video_path.name} -> {processed_base.name}/{video_name}/")
        n = extract_frames_ffmpeg(video_path, out_img_dir, video_name, args.fps)
        print(f"  提取 {n} 张图片")
        ensure_labels_dir_and_classes(video_name)

    print()
    print("完成。请使用 labelImg 对 data/processed/抻面/<视频名>/ 下的图片进行标注，")
    print("保存到 data/labels/抻面/<视频名>/，类别顺序保持: hand, noodle_rope, noodle_bundle。")
    print("标注完成后请运行: python scripts/sync_stretch_classes_from_master.py")


if __name__ == "__main__":
    main()
