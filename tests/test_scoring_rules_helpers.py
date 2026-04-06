import json
from pathlib import Path

from src.webapp.scoring_rules_helpers import score_100_and_grade, scoring_rules_paths


def test_score_100_without_rules_file(tmp_path: Path):
    pr = tmp_path
    s, g = score_100_and_grade(pr, "stretch", 3.0)
    assert g is None
    assert 40 <= s <= 60


def test_score_100_with_rules_file(tmp_path: Path):
    d = tmp_path / "data" / "scores" / "抻面"
    d.mkdir(parents=True)
    p = d / "scoring_rules.json"
    p.write_text(
        json.dumps({"pass_threshold": 50, "excellent_threshold": 90}, ensure_ascii=False),
        encoding="utf-8",
    )
    s, grade = score_100_and_grade(tmp_path, "stretch", 5.0)
    assert s == 100.0
    assert grade == "优秀"


def test_scoring_rules_paths_keys(tmp_path: Path):
    d = scoring_rules_paths(tmp_path)
    assert set(d.keys()) == {"stretch", "boiling"}
