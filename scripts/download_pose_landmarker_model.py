#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 MediaPipe 身体姿态模型到 weights/mediapipe/，用于实时检测骨架线（手+身体）。
默认下载 lite（速度快）。若需更高身体骨架准确率，可手动下载 heavy 并保存为 pose_landmarker_heavy.task：
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "weights" / "mediapipe"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "pose_landmarker_lite.task"
url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def main():
    if out_file.exists():
        print(f"已存在: {out_file}")
        return 0
    try:
        import urllib.request
        print(f"正在下载: {url}")
        urllib.request.urlretrieve(url, out_file)
        print(f"已保存: {out_file}")
        return 0
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载并放到 weights/mediapipe/ 目录:")
        print(f"  {url}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
