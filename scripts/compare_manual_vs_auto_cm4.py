#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 cm4 的「人工评分 vs 自动评分」分布。

思路：
- 人工评分来自: data/scores/抻面/cm4/*_scores.json
- 检测框来自标注: data/labels/抻面/cm4/*.txt（与 Web 工具 get_frame_detections 一致，conf=1.0）
- 自动评分使用: src/scoring/stretch_scorer.py 的 StretchScorer（基于图像特征）

输出：
- reports/compare_cm4_manual_vs_auto.csv  逐帧汇总（overall + 每类）
- reports/compare_cm4_manual_vs_auto_summary.json  分布统计与差值
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
try:
    # Windows 终端可能是 GBK，强制 UTF-8 避免打印异常
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _read_image_unicode(image_path: Path):
    """OpenCV 兼容中文路径读取。返回 BGR ndarray 或 None"""
    try:
        import cv2

        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _load_class_names(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        return []
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_label_detections(
    label_file: Path, image_w: int, image_h: int, class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    读取 YOLO txt 标注并转为 detection dict（与 start_web.py /api/get_frame_detections 一致）
    """
    detections: List[Dict[str, Any]] = []
    if not label_file.exists():
        return detections

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center = float(parts[1]) * image_w
            y_center = float(parts[2]) * image_h
            width = float(parts[3]) * image_w
            height = float(parts[4]) * image_h

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            detections.append(
                {
                    "class": cls_name,
                    "class_id": cls_id,
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": 1.0,  # 标注数据置信度为 1
                }
            )
    return detections


def _strip_notes(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if k == "notes":
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _calc_overall_from_class_scores(class_scores: Dict[str, float], overall_weights: Dict[str, float]) -> float:
    total, wsum = 0.0, 0.0
    for cls, s in class_scores.items():
        w = float(overall_weights.get(cls, 0.0))
        if w <= 0:
            continue
        total += s * w
        wsum += w
    return total / wsum if wsum > 0 else 0.0


def _dist_1_to_5(values: List[float]) -> Dict[str, int]:
    """
    将分数（可能是小数）round 到 1-5 档做分布统计。
    """
    bins = Counter()
    for v in values:
        if v is None:
            continue
        if v <= 0:
            continue
        b = int(round(v))
        b = max(1, min(5, b))
        bins[str(b)] += 1
    return {str(i): int(bins.get(str(i), 0)) for i in range(1, 6)}


def main():
    from src.scoring.stretch_scorer import StretchScorer

    video = "cm4"
    scores_dir = project_root / "data" / "scores" / "抻面" / video
    labels_dir = project_root / "data" / "labels" / "抻面" / video
    images_dir = project_root / "data" / "processed" / "抻面" / video
    classes_file = labels_dir.parent / "classes.txt"

    if not scores_dir.exists():
        raise FileNotFoundError(f"评分目录不存在: {scores_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    class_names = _load_class_names(classes_file)
    scorer = StretchScorer()
    overall_weights = scorer.rules.get("overall_weights", {})

    score_files = sorted(scores_dir.glob("*_scores.json"))
    if not score_files:
        print("[WARN] 未找到评分文件")
        return

    # 收集逐帧对比数据
    rows: List[Dict[str, Any]] = []
    overall_manual_list: List[float] = []
    overall_auto_list: List[float] = []
    overall_delta_list: List[float] = []

    per_class_manual: Dict[str, List[float]] = defaultdict(list)
    per_class_auto: Dict[str, List[float]] = defaultdict(list)
    per_class_delta: Dict[str, List[float]] = defaultdict(list)

    for sf in score_files:
        with open(sf, "r", encoding="utf-8") as f:
            manual_payload = json.load(f)

        frame = manual_payload.get("frame")
        if not frame:
            continue

        image_path = images_dir / frame
        if not image_path.exists():
            print(f"[跳过] 图片不存在: {image_path}")
            continue

        img = _read_image_unicode(image_path)
        if img is None:
            print(f"[跳过] 读取图片失败: {image_path}")
            continue

        h, w = img.shape[:2]
        label_file = labels_dir / frame.replace(".jpg", ".txt")
        detections = _load_label_detections(label_file, w, h, class_names)
        if not detections:
            print(f"[跳过] 未找到标注检测框: {label_file}")
            continue

        # --- 自动评分（基于图像特征）---
        auto_frame = scorer.score_frame(detections, img)
        auto_class_scores = auto_frame.get("class_scores", {}) or {}
        auto_overall = float(auto_frame.get("overall_score", 0.0) or 0.0)

        # --- 人工评分（按 detection_i 对齐标注行顺序）---
        manual_scores_by_det = manual_payload.get("scores", {}) or {}
        manual_det_weighted: List[Tuple[str, float]] = []

        for i, det in enumerate(detections):
            key = f"detection_{i}"
            ms = manual_scores_by_det.get(key)
            if not isinstance(ms, dict):
                continue
            cls = det.get("class", "")
            if not cls:
                continue
            attrs = _strip_notes(ms)
            if not attrs:
                continue
            wscore = float(scorer.calculate_weighted_score(attrs, cls))
            manual_det_weighted.append((cls, wscore))

        # 聚合到 class_scores（与 scorer.score_frame 逻辑一致：同类取平均）
        manual_class_buckets: Dict[str, List[float]] = defaultdict(list)
        for cls, ws in manual_det_weighted:
            manual_class_buckets[cls].append(ws)
        manual_class_scores: Dict[str, float] = {
            cls: (sum(vs) / len(vs)) for cls, vs in manual_class_buckets.items() if vs
        }
        manual_overall = float(_calc_overall_from_class_scores(manual_class_scores, overall_weights))

        # 收集总体
        overall_manual_list.append(manual_overall)
        overall_auto_list.append(auto_overall)
        overall_delta_list.append(auto_overall - manual_overall)

        # 收集分项
        all_classes = set(manual_class_scores.keys()) | set(auto_class_scores.keys())
        for cls in sorted(all_classes):
            m = float(manual_class_scores.get(cls, 0.0) or 0.0)
            a = float(auto_class_scores.get(cls, 0.0) or 0.0)
            if m > 0:
                per_class_manual[cls].append(m)
            if a > 0:
                per_class_auto[cls].append(a)
            if m > 0 and a > 0:
                per_class_delta[cls].append(a - m)

        rows.append(
            {
                "frame": frame,
                "manual_overall": round(manual_overall, 4),
                "auto_overall": round(auto_overall, 4),
                "delta_overall": round(auto_overall - manual_overall, 4),
                "manual_noodle_rope": round(float(manual_class_scores.get("noodle_rope", 0.0) or 0.0), 4),
                "auto_noodle_rope": round(float(auto_class_scores.get("noodle_rope", 0.0) or 0.0), 4),
                "manual_hand": round(float(manual_class_scores.get("hand", 0.0) or 0.0), 4),
                "auto_hand": round(float(auto_class_scores.get("hand", 0.0) or 0.0), 4),
                "manual_noodle_bundle": round(float(manual_class_scores.get("noodle_bundle", 0.0) or 0.0), 4),
                "auto_noodle_bundle": round(float(auto_class_scores.get("noodle_bundle", 0.0) or 0.0), 4),
            }
        )

    # 输出
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f"compare_{video}_manual_vs_auto_{ts}.csv"
    summary_path = reports_dir / f"compare_{video}_manual_vs_auto_summary_{ts}.json"

    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)

    def _summary(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": round(mean(values), 4),
            "median": round(median(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "dist_1_to_5": _dist_1_to_5(values),
        }

    summary: Dict[str, Any] = {
        "video": video,
        "frames_compared": len(rows),
        "overall": {
            "manual": _summary(overall_manual_list),
            "auto": _summary(overall_auto_list),
            "delta(auto-manual)": _summary(overall_delta_list),
        },
        "per_class": {},
        "artifacts": {
            "csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }

    for cls in sorted(set(per_class_manual.keys()) | set(per_class_auto.keys())):
        summary["per_class"][cls] = {
            "manual": _summary(per_class_manual.get(cls, [])),
            "auto": _summary(per_class_auto.get(cls, [])),
            "delta(auto-manual)": _summary(per_class_delta.get(cls, [])),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"cm4 人工 vs 自动 对比完成（帧数: {len(rows)}）")
    print(f"- CSV: {csv_path}")
    print(f"- Summary: {summary_path}")
    if overall_manual_list and overall_auto_list:
        print(f"- overall manual mean: {mean(overall_manual_list):.3f}")
        print(f"- overall auto   mean: {mean(overall_auto_list):.3f}")
        print(f"- overall delta  mean: {mean(overall_delta_list):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

