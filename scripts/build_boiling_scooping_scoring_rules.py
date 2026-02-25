#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 data/scores/下面及捞面/xl*/*_scores.json 统计各属性，生成 scoring_rules.json。
与抻面规则格式兼容：thresholds（excellent/good/fair/poor/mean/std）、weights、overall_weights。
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    project_root = Path(__file__).resolve().parent.parent
    scores_dir = project_root / "data" / "scores" / "下面及捞面"
    out_path = scores_dir / "scoring_rules.json"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 *_scores.json 中的属性值（跳过 notes）
    attr_values = defaultdict(list)
    for sub in scores_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*_scores.json"):
            try:
                with open(f, "r", encoding="utf-8-sig") as fp:
                    data = json.load(fp)
            except Exception:
                continue
            scores = data.get("scores") or {}
            for det_id, det_s in scores.items():
                if not isinstance(det_s, dict):
                    continue
                for k, v in det_s.items():
                    if k == "notes" or not isinstance(v, (int, float)):
                        continue
                    v = float(v)
                    if 0 <= v <= 5:
                        attr_values[k].append(v)

    if not attr_values:
        print("未找到任何标注分数，请先运行下面及捞面评分标注并保存。")
        return

    # 各属性阈值：mean, std, poor(25%分位), fair(50%), good(75%), excellent(90% 或 4.5)
    thresholds = {}
    statistics = {}
    for attr, vals in attr_values.items():
        arr = np.array(vals)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) if n > 1 else 0.0
        q25 = float(np.percentile(arr, 25))
        q50 = float(np.percentile(arr, 50))
        q75 = float(np.percentile(arr, 75))
        q90 = float(np.percentile(arr, 90))
        poor = min(q25, mean - 0.5 * std) if std > 0 else q25
        fair = q50
        good = q75
        excellent = min(4.5, q90)
        thresholds[attr] = {
            "excellent": round(excellent, 4),
            "good": round(good, 4),
            "fair": round(fair, 4),
            "poor": round(max(1.0, poor), 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
        }
        statistics[attr] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": round(q50, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "count": n,
        }

    # 下面及捞面四类与标注属性的对应（用于帧级评分器）
    # hand / noodle_rope 与标注一致；tools_noodle、soup_noodle 用部分属性或置信度映射
    weights = {
        "hand": {
            "position": 0.25,
            "action": 0.25,
            "angle": 0.2,
            "coordination": 0.15,
            "tool_coordination": 0.15,
        },
        "noodle_rope": {
            "thickness": 0.15,
            "elasticity": 0.15,
            "integrity": 0.15,
            "ripeness": 0.15,
            "soup_adhesion": 0.15,
            "noodle_soup_ratio": 0.15,
            "distribution_state": 0.10,
        },
        "tools_noodle": {
            "operation_standardization": 0.5,
            "tool_coordination": 0.5,
        },
        "soup_noodle": {
            "ripeness": 0.5,
            "distribution_state": 0.5,
        },
    }
    # 为没有标注统计的属性补默认阈值（置信度映射时用）
    default_attrs = ["angle", "tool_coordination", "operation_standardization"]
    for a in default_attrs:
        if a not in thresholds:
            thresholds[a] = {"excellent": 4.5, "good": 3.5, "fair": 2.5, "poor": 2.0, "mean": 3.0, "std": 0.8}
            statistics[a] = {"mean": 3.0, "std": 0.8, "min": 1, "max": 5, "median": 3, "q25": 2.5, "q75": 3.5, "count": 0}

    overall_weights = {
        "noodle_rope": 0.35,
        "hand": 0.35,
        "tools_noodle": 0.15,
        "soup_noodle": 0.15,
    }

    out = {
        "version": "1.0",
        "stage": "下面及捞面",
        "thresholds": thresholds,
        "statistics": statistics,
        "weights": weights,
        "overall_weights": overall_weights,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已生成: {out_path}")
    print(f"  属性数: {len(thresholds)}, 总样本数: {sum(len(v) for v in attr_values.values())}")


if __name__ == "__main__":
    main()
