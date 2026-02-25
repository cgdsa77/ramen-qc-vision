#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 MediaPipe 人脸检测模型到 weights/mediapipe/，用于实时监测「头部检测」。"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / "weights" / "mediapipe"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "face_detection_short_range.tflite"
url = "https://storage.googleapis.com/mediapipe-tasks/face_detector/face_detection_short_range.tflite"

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
