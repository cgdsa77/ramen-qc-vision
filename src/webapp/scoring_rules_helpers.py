"""评分规则 JSON 路径与百分制/等级换算（供评分 API 与路由共用）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

OVERALL_WEIGHT_LABELS: Dict[str, Dict[str, str]] = {
    "stretch": {"noodle_rope": "面绳", "hand": "手部", "noodle_bundle": "面束"},
    "boiling": {"noodle_rope": "面绳", "hand": "手部", "tools_noodle": "工具面", "soup_noodle": "汤面"},
}


def scoring_rules_paths(project_root: Path) -> Dict[str, Path]:
    return {
        "stretch": project_root / "data" / "scores" / "抻面" / "scoring_rules.json",
        "boiling": project_root / "data" / "scores" / "下面及捞面" / "scoring_rules.json",
    }


def score_100_and_grade(project_root: Path, stage: str, average_overall_score_1_5: float) -> Tuple[float, Optional[str]]:
    """根据规则中的及格线/优秀线，将 1-5 分制转为百分制并得到等级。"""
    path = scoring_rules_paths(project_root).get(stage)
    if not path or not path.exists():
        return round((average_overall_score_1_5 - 1) / 4 * 100, 1), None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return round((average_overall_score_1_5 - 1) / 4 * 100, 1), None
    pass_t = float(data.get("pass_threshold", 60))
    excellent_t = float(data.get("excellent_threshold", 85))
    score_100 = (average_overall_score_1_5 - 1) / 4 * 100
    score_100 = round(max(0, min(100, score_100)), 1)
    if score_100 >= excellent_t:
        grade = "优秀"
    elif score_100 >= pass_t:
        grade = "良好"
    else:
        grade = "不及格"
    return score_100, grade
