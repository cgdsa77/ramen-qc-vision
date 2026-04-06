#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将抻面 / 下面及捞面 YOLO 权重导出为 ONNX，供 RAMEN_PREFER_ONNX=1 时加速推理（Ultralytics 内部走 onnxruntime）。

用法（项目根目录）:
  python scripts/export_yolo_to_onnx.py --stretch
  python scripts/export_yolo_to_onnx.py --boiling
  python scripts/export_yolo_to_onnx.py --weights models/stretch_detection/weights/best.pt

TensorRT（可选，需本机 CUDA + TensorRT 环境）:
  python scripts/export_yolo_to_onnx.py --stretch --format engine

导出完成后重启 Web 服务，并设置环境变量:
  set RAMEN_PREFER_ONNX=1
  python start_web.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def _resolve_weights(which: str) -> Path:
    from src.api.video_detection_api import _latest_stretch_best_pt, _latest_boiling_best_pt

    if which == "stretch":
        p = _latest_stretch_best_pt(project_root)
    else:
        p = _latest_boiling_best_pt(project_root)
    if p is None or not p.is_file():
        raise SystemExit(f"[错误] 未找到 {which} 的 best.pt，请先训练或检查 models/ 目录")
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 YOLO 为 ONNX / engine")
    parser.add_argument("--stretch", action="store_true", help="导出当前最新的抻面 stretch_detection*/weights/best.pt")
    parser.add_argument("--boiling", action="store_true", help="导出当前最新的下面及捞面 boiling_scooping_detection*/weights/best.pt")
    parser.add_argument("--weights", type=str, default=None, help="直接指定 .pt 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入边长，需与检测时 imgsz 一致")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=("onnx", "engine"),
        help="onnx（通用）或 engine（TensorRT，仅 NVIDIA + 已安装 TensorRT 时）",
    )
    parser.add_argument("--half", action="store_true", help="FP16（engine 时常用）")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 需要 ultralytics: pip install ultralytics")
        return 1

    paths: list[Path] = []
    if args.weights:
        w = Path(args.weights)
        if not w.is_file():
            print(f"[错误] 文件不存在: {w}")
            return 1
        paths.append(w)
    else:
        if args.stretch:
            paths.append(_resolve_weights("stretch"))
        if args.boiling:
            paths.append(_resolve_weights("boiling"))
        if not args.stretch and not args.boiling:
            print("请指定 --stretch、--boiling 或 --weights")
            return 1

    for pt in paths:
        print(f"\n导出: {pt}")
        model = YOLO(str(pt))
        # ultralytics: simplify=True 合并算子，opset 默认兼容 onnxruntime
        out = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            simplify=True,
        )
        print(f"  [OK] {out}")

    print("\n说明: 启动服务前设置 RAMEN_PREFER_ONNX=1（onnx）或放置同目录 .engine 并设置 RAMEN_PREFER_ENGINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
