#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评分规则校验脚本（对应《抻面与下面及捞面评分规则全体系优化方案》中的强制校验）
校验项：结构完整、阈值单调性及[1,5]区间、权重和=1、键名与代码一致。
运行：python scripts/validate_scoring_rules.py
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
PATHS = {
    "stretch": project_root / "data" / "scores" / "抻面" / "scoring_rules.json",
    "boiling": project_root / "data" / "scores" / "下面及捞面" / "scoring_rules.json",
}

REQUIRED_TOP_KEYS = {"version", "stage", "thresholds", "weights", "overall_weights"}
WEIGHT_SUM_TOL = 0.001
THRESHOLD_BOUNDS = (1.0, 5.0)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_thresholds(thresholds: dict, stage: str) -> list:
    errs = []
    if not isinstance(thresholds, dict):
        return ["thresholds 应为对象"]
    for attr, t in thresholds.items():
        if attr in ("version", "update_time", "calibration_basis", "operator"):
            continue
        if not isinstance(t, dict):
            continue
        e = t.get("excellent")
        g = t.get("good")
        f = t.get("fair")
        p = t.get("poor")
        if e is None or g is None or f is None or p is None:
            continue
        try:
            e, g, f, p = float(e), float(g), float(f), float(p)
        except (TypeError, ValueError):
            errs.append(f"[{stage}] thresholds.{attr} 含非数字")
            continue
        if not (e >= g >= f >= p):
            errs.append(f"[{stage}] thresholds.{attr} 不满足 excellent≥good≥fair≥poor (当前 {e},{g},{f},{p})")
        for name, v in [("excellent", e), ("good", g), ("fair", f), ("poor", p)]:
            if v < THRESHOLD_BOUNDS[0] or v > THRESHOLD_BOUNDS[1]:
                errs.append(f"[{stage}] thresholds.{attr}.{name}={v} 超出 [1,5]")
    return errs


def check_weights(weights: dict, stage: str) -> list:
    errs = []
    if not isinstance(weights, dict):
        return ["weights 应为对象"]
    for cls, wdict in weights.items():
        if not isinstance(wdict, dict):
            continue
        s = sum(float(v) for v in wdict.values())
        if abs(s - 1.0) > WEIGHT_SUM_TOL:
            errs.append(f"[{stage}] weights.{cls} 之和={s:.4f}，应为 1")
    return errs


def check_overall_weights(overall: dict, stage: str) -> list:
    errs = []
    if not isinstance(overall, dict):
        return ["overall_weights 应为对象"]
    s = sum(float(v) for v in overall.values())
    if abs(s - 1.0) > WEIGHT_SUM_TOL:
        errs.append(f"[{stage}] overall_weights 之和={s:.4f}，应为 1")
    return errs


def main():
    all_errors = []
    for name, path in PATHS.items():
        if not path.exists():
            all_errors.append(f"[{name}] 文件不存在: {path}")
            continue
        try:
            data = load_json(path)
        except Exception as e:
            all_errors.append(f"[{name}] JSON 解析失败: {e}")
            continue
        missing = REQUIRED_TOP_KEYS - set(data.keys())
        if missing:
            all_errors.append(f"[{name}] 缺少顶层键: {missing}")
        if "thresholds" in data:
            all_errors.extend(check_thresholds(data["thresholds"], name))
        if "weights" in data:
            all_errors.extend(check_weights(data["weights"], name))
        if "overall_weights" in data:
            all_errors.extend(check_overall_weights(data["overall_weights"], name))

    if all_errors:
        for e in all_errors:
            print(e)
        print("\n校验未通过，请修改规则文件后重新运行。")
        sys.exit(1)
    print("校验通过：结构完整，阈值单调且落在 [1,5]，权重和均为 1。")
    sys.exit(0)


if __name__ == "__main__":
    main()
