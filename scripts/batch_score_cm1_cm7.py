#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量对 cm1~cm7 进行抻面自动评分（使用当前 StretchScorer 与检测模型）。

流程：
- 逐个读取 data/raw/抻面/cmX.mp4
- 使用 VideoDetectionAPI.detect_video 做检测
- 使用 StretchScorer.score_video 做评分（基于图像特征与当前偏置）
- 将每个视频的评分结果保存到 reports/stretch_auto_score_cmX.json
- 额外生成一个总汇总 reports/stretch_auto_score_cm1_cm7_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    from src.api.video_detection_api import VideoDetectionAPI
    from src.scoring.stretch_scorer import StretchScorer

    raw_dir = project_root / "data" / "raw" / "抻面"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    videos = [f"cm{i}" for i in range(1, 8)]

    detector = VideoDetectionAPI()  # 使用默认抻面检测模型
    scorer = StretchScorer()

    summary: Dict[str, Any] = {
        "stage": "stretch",
        "videos": {},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rules_version": scorer.rules.get("version", "unknown"),
    }

    print("=" * 60)
    print("批量自动评分：cm1 ~ cm7（抻面）")
    print("=" * 60)

    for vid in videos:
        video_path = raw_dir / f"{vid}.mp4"
        if not video_path.exists():
            print(f"[跳过] 未找到视频: {video_path}")
            continue

        print(f"\n--- 处理 {vid} ---")
        print(f"视频路径: {video_path}")

        # 1) 检测
        det_result = detector.detect_video(str(video_path), conf_threshold=0.25)
        if not det_result.get("success"):
            print(f"[错误] 检测失败: {det_result}")
            continue

        detections: List[Dict[str, Any]] = det_result.get("detections", [])
        total_frames = det_result.get("total_frames", len(detections))
        print(f"检测完成，帧数: {total_frames}, 有检测结果的帧数: {len(detections)}")

        # 2) 评分
        score_result = scorer.score_video(detections, video_path=str(video_path))

        # 保存单视频结果
        out_path = reports_dir / f"stretch_auto_score_{vid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(score_result, f, ensure_ascii=False, indent=2)

        avg_overall = float(score_result.get("average_overall_score", 0.0) or 0.0)
        class_avg = score_result.get("class_average_scores", {})
        summary["videos"][vid] = {
            "video_path": str(video_path),
            "total_frames": score_result.get("total_frames", total_frames),
            "scored_frames": score_result.get("scored_frames", len(detections)),
            "average_overall_score": avg_overall,
            "class_average_scores": class_avg,
            "report_file": str(out_path),
        }

        print(f"[完成] {vid} 平均总分: {avg_overall:.3f}, 分项: {class_avg}")

    # 保存汇总
    summary_path = reports_dir / "stretch_auto_score_cm1_cm7_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("批量评分完成。汇总信息：")
    print(f"- Summary: {summary_path}")
    for vid, info in summary["videos"].items():
        print(
            f"  {vid}: 总分均值={info['average_overall_score']:.3f}, "
            f"noodle_rope={info['class_average_scores'].get('noodle_rope', 0):.3f}, "
            f"hand={info['class_average_scores'].get('hand', 0):.3f}, "
            f"noodle_bundle={info['class_average_scores'].get('noodle_bundle', 0):.3f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

