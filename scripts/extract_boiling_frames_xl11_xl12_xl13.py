#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下面及捞面 xl11、xl12、xl13：从 raw 抽帧到 processed，并同步 labels 子目录的 classes.txt（供 labelImg 使用）。
不修改其他 xl 目录下已有标注。

- 视频来源: data/raw/下面及捞面/xl11.mp4（或 .MP4）等
- 输出图片: data/processed/下面及捞面/xl11/xl11_00001.jpg（命名与既有 xl 一致）
- 抽帧默认 2 fps，与 extract_video_frames / cm 抽帧脚本一致

用法:
  python scripts/extract_boiling_frames_xl11_xl12_xl13.py
  python scripts/extract_boiling_frames_xl11_xl12_xl13.py --fps 1.5
  python scripts/extract_boiling_frames_xl11_xl12_xl13.py --dry-run
"""
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw" / "下面及捞面"
processed_base = project_root / "data" / "processed" / "下面及捞面"
labels_base = project_root / "data" / "labels" / "下面及捞面"
main_classes_file = labels_base / "classes.txt"

TARGET_VIDEOS = ["xl11", "xl12", "xl13"]


def _imwrite_jpg(path: Path, frame_bgr) -> bool:
    """Windows 下路径含中文时 cv2.imwrite 常失败；用 imencode 再二进制写入。"""
    try:
        import cv2
    except ImportError:
        return False
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok or buf is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def extract_frames_ffmpeg(video_path: Path, output_dir: Path, video_name: str, fps: float) -> int:
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
    n = len(list(output_dir.glob("*.jpg")))
    if n == 0:
        print("  [警告] ffmpeg 未写出 jpg（路径含中文时常见），改用 OpenCV 抽帧")
        return extract_frames_opencv(video_path, output_dir, video_name, fps)
    return n


def extract_frames_opencv(video_path: Path, output_dir: Path, video_name: str, fps: float) -> int:
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
    interval = max(1, round(video_fps / fps))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out_name = output_dir / f"{video_name}_{saved + 1:05d}.jpg"
            if _imwrite_jpg(out_name, frame):
                saved += 1
        count += 1
    cap.release()
    if saved == 0:
        print("  [错误] 未写出任何帧：请确认视频可读，或把项目放到仅英文路径下重试")
    return saved


def ensure_labels_dir_and_classes(video_name: str) -> None:
    if video_name not in TARGET_VIDEOS:
        return
    label_dir = labels_base / video_name
    label_dir.mkdir(parents=True, exist_ok=True)
    if not main_classes_file.exists():
        print(f"  [警告] 主 classes.txt 不存在，请创建: {main_classes_file}")
        return
    dest = label_dir / "classes.txt"
    shutil.copy2(main_classes_file, dest)
    print(f"  [OK] 已同步 classes.txt -> {label_dir.name}/classes.txt")


def main():
    parser = argparse.ArgumentParser(description="下面及捞面 xl11/xl12/xl13 抽帧并准备 labelImg")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧帧率（默认 2）")
    parser.add_argument("--dry-run", action="store_true", help="只打印步骤，不写文件")
    args = parser.parse_args()

    if not raw_dir.exists():
        print(f"错误：raw 目录不存在: {raw_dir}")
        sys.exit(1)

    print("处理视频: " + ", ".join(TARGET_VIDEOS))
    print(f"抽帧: fps={args.fps}，命名 <视频名>_00001.jpg …")
    if args.dry_run:
        print("[dry-run] 不写入文件")
    print()

    for video_name in TARGET_VIDEOS:
        video_path = None
        for ext in (".mp4", ".MP4"):
            p = raw_dir / f"{video_name}{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            print(f"[跳过] {video_name}: 未找到 {raw_dir}/{video_name}.mp4")
            continue

        out_img_dir = processed_base / video_name
        if args.dry_run:
            print(f"[dry-run] 将抽帧: {video_path} -> {out_img_dir}")
            print(f"[dry-run] 将同步 classes.txt -> {labels_base / video_name}/")
            continue

        print(f"正在抽帧: {video_path.name} -> processed/下面及捞面/{video_name}/")
        n = extract_frames_ffmpeg(video_path, out_img_dir, video_name, args.fps)
        print(f"  提取 {n} 张图片")
        ensure_labels_dir_and_classes(video_name)

    print()
    print("labelImg 设置：")
    print(f"  图片目录: {processed_base}/xl11|xl12|xl13/")
    print(f"  标注保存: {labels_base}/xl11|xl12|xl13/")
    print("  类别顺序见 data/labels/下面及捞面/classes.txt（noodle_rope, hand, tools_noodle, soup_noodle）。")


if __name__ == "__main__":
    main()
