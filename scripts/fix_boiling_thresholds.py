#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性修正下面及捞面规则中无区分度阈值，保证 excellent > good > fair > poor，并保留2位小数。"""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data" / "scores" / "下面及捞面" / "scoring_rules.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

th = data["thresholds"]
# 修正无区分度：保证 e > g > f > p，且均在 [1,5]
def fix_attr(name, e, g, f, p, mean, std):
    e, g, f, p = float(e), float(g), float(f), float(p)
    # 保证严格单调 e >= g >= f >= p，且均在 [1,5]
    e = round(max(1, min(5, e)), 2)
    g = round(max(1, min(5, g)), 2)
    f = round(max(1, min(5, f)), 2)
    p = round(max(1, min(5, p)), 2)
    if e <= g: g = round(e - 0.01, 2)
    if g <= f: f = round(g - 0.01, 2)
    if f <= p: p = round(f - 0.01, 2)
    p = max(1.0, p)
    th[name].update({"excellent": e, "good": g, "fair": f, "poor": p, "mean": round(float(mean), 2), "std": round(float(std), 2)})

for attr, v in list(th.items()):
    if not isinstance(v, dict) or "excellent" not in v:
        continue
    fix_attr(attr, v["excellent"], v["good"], v["fair"], v["poor"], v.get("mean", 3), v.get("std", 0.5))

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("已修正并写回:", path)
