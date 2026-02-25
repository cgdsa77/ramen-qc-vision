"""
拉面成品评分模块
以面条质感为主体，汤型、面条裹辣椒、辅料有无不参与惩罚，仅可选小幅呈现加分。
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# 项目根目录
_project_root = Path(__file__).resolve().parents[2]


class ProductScorer:
    """
    拉面成品评分器。
    核心原则：
    - 总分由面条质感 S_texture 主导，清汤/红汤、辣椒有无不参与扣分。
    - 面条裹辣椒（all/partial/none）仅反映进食状态或汤型，不用于惩罚。
    - 可选 S_presentation 仅对辅料做小幅正向加分，无辅料不扣分。
    """

    DEFAULT_RULES_PATH = _project_root / "data" / "scores" / "拉面成品" / "product_scoring_rules.json"

    def __init__(self, rules_path: Optional[Path] = None):
        if rules_path is None:
            rules_path = self.DEFAULT_RULES_PATH
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"成品评分规则不存在: {self.rules_path}")
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _s_texture(self, noodle_quality: str) -> float:
        """仅由面条质感等级得到核心得分，汤型/辣椒不参与。"""
        mapping = self.rules.get("noodle_quality_mapping", {})
        entry = mapping.get((noodle_quality or "").strip().lower())
        if entry and "score" in entry:
            return float(entry["score"])
        # 未知等级时按 fair 处理，避免极端
        return float(mapping.get("fair", {}).get("score", 58))

    def _s_presentation(self, annotation: Dict[str, Any]) -> float:
        """可选呈现加分 [0, max_bonus]。无辅料得 0，不扣分。"""
        bonus_cfg = self.rules.get("presentation_bonus", {})
        if not bonus_cfg:
            return 0.0
        total = 0.0
        for key, levels in [("beef", bonus_cfg.get("beef", {})),
                            ("scallion", bonus_cfg.get("scallion", {}))]:
            val = (annotation.get(key) or "").strip().lower()
            total += float(levels.get(val, 0))
        egg_cfg = bonus_cfg.get("egg", {})
        egg_val = (annotation.get("egg") or "").strip().lower()
        if egg_val in ("yes", "little", "normal", "more"):
            total += float(egg_cfg.get("yes", egg_cfg.get(egg_val, 0)))
        else:
            total += float(egg_cfg.get("none", 0))
        max_bonus = float(bonus_cfg.get("max_bonus", 5.0))
        return min(max_bonus, max(0.0, total))

    def score_from_annotation(self, annotation: Dict[str, Any],
                              use_presentation: bool = True) -> Dict[str, Any]:
        """
        从单条标注计算成品得分。
        不因 soup_type、noodle_chili 扣分；仅以 noodle_quality 为主，可选加 S_presentation。

        Args:
            annotation: 至少包含 noodle_quality；可含 beef, egg, scallion, soup_type, noodle_chili
            use_presentation: 是否加上 S_presentation 加分

        Returns:
            total_score: 0–100
            s_texture, s_presentation, noodle_quality, formula_used
        """
        noodle_quality = (annotation.get("noodle_quality") or "fair").strip().lower()
        s_texture = self._s_texture(noodle_quality)
        s_presentation = self._s_presentation(annotation) if use_presentation else 0.0
        w = float(self.rules.get("formula", {}).get("w_presentation", 0.04))
        total = s_texture + w * s_presentation
        total = max(0.0, min(100.0, total))
        return {
            "total_score": round(total, 2),
            "s_texture": round(s_texture, 2),
            "s_presentation": round(s_presentation, 2),
            "noodle_quality": noodle_quality,
            "formula_used": "S_texture + w_presentation * S_presentation (texture-dominant, no soup/chili penalty)",
        }

    def score_from_prediction(self, noodle_quality: str,
                              presentation_bonus: float = 0.0) -> Dict[str, Any]:
        """
        从模型预测的 noodle_quality 计算得分（无辅料时 presentation_bonus=0）。
        """
        s_texture = self._s_texture(noodle_quality)
        w = float(self.rules.get("formula", {}).get("w_presentation", 0.04))
        total = s_texture + w * presentation_bonus
        total = max(0.0, min(100.0, total))
        return {
            "total_score": round(total, 2),
            "s_texture": round(s_texture, 2),
            "s_presentation": round(presentation_bonus, 2),
            "noodle_quality": noodle_quality.strip().lower(),
            "formula_used": "S_texture + w_presentation * S_presentation",
        }

    def batch_score_from_annotations(self, items: List[Dict[str, Any]],
                                    use_presentation: bool = True) -> List[Dict[str, Any]]:
        """对多条标注批量评分。"""
        return [self.score_from_annotation(item, use_presentation=use_presentation) for item in items]


def load_annotations(annotations_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载拉面成品标注汇总。"""
    if annotations_path is None:
        annotations_path = _project_root / "data" / "scores" / "拉面成品" / "annotations.json"
    path = Path(annotations_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])
