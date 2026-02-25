"""
下面及捞面帧级评分器（上传视频用）
与抻面一致：检测 + 骨架/图像特征 + 规则（信息多，分数更细）。
- 有视频帧时使用 ImageFeatureExtractor 提取图像特征，经校准后按规则打分；
- 无图像或提取失败时回退为置信度 + 规则。
"""
import json
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

_project_root = Path(__file__).resolve().parents[2]
RULES_PATH = _project_root / "data" / "scores" / "下面及捞面" / "scoring_rules.json"
CLASSES = ["noodle_rope", "hand", "tools_noodle", "soup_noodle"]


class BoilingScoopingScorer:
    """下面及捞面帧级评分：检测 + 图像特征 + 规则（与抻面 StretchScorer 同级）"""

    def __init__(self, rules_path: Optional[Path] = None, use_image_features: bool = True):
        self.rules_path = Path(rules_path or RULES_PATH)
        self.rules = self._load_rules()
        self.use_image_features = use_image_features
        self.feature_extractor = None
        if self.use_image_features:
            try:
                from src.features.image_feature_extractor import ImageFeatureExtractor
                self.feature_extractor = ImageFeatureExtractor()
            except Exception:
                self.use_image_features = False

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"下面及捞面评分规则不存在: {self.rules_path}，请先运行 scripts/build_boiling_scooping_scoring_rules.py"
            )
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _calibrate_features(self, features: Dict[str, float], class_name: str) -> Dict[str, float]:
        """将提取的原始特征（约 0~5）按规则中的 mean/poor 校准到与人工评分一致的范围。"""
        calibrated = {}
        # 下面及捞面各类别在规则中的属性
        class_attrs = {
            "hand": ["position", "action", "angle", "coordination", "tool_coordination"],
            "noodle_rope": ["thickness", "elasticity", "integrity", "ripeness", "soup_adhesion", "noodle_soup_ratio", "distribution_state"],
            "tools_noodle": ["operation_standardization", "tool_coordination"],
            "soup_noodle": ["ripeness", "distribution_state"],
        }
        attrs = class_attrs.get(class_name, [])
        for attr in attrs:
            raw = features.get(attr, 0.0)
            th = self.rules.get("thresholds", {}).get(attr, {})
            mean_val = th.get("mean", 3.0)
            poor_val = th.get("poor", 2.0)
            if raw < 0:
                calibrated[attr] = poor_val
            elif raw <= 2.0:
                ratio = raw / 2.0
                calibrated[attr] = poor_val + (mean_val - poor_val) * ratio
                calibrated[attr] = max(poor_val, min(mean_val, calibrated[attr]))
            elif raw <= 3.5:
                ratio = (raw - 2.0) / 1.5
                calibrated[attr] = mean_val + (4.0 - mean_val) * ratio * 0.5
                calibrated[attr] = max(mean_val, min(4.5, calibrated[attr]))
            elif raw <= 5.0:
                ratio = (raw - 3.5) / 1.5
                calibrated[attr] = 4.0 + 1.0 * ratio
                calibrated[attr] = max(4.0, min(5.0, calibrated[attr]))
            else:
                calibrated[attr] = 5.0
            calibrated[attr] = max(1.0, min(5.0, calibrated[attr]))
        return calibrated

    def score_attribute(self, attribute: str, value: float) -> float:
        """按规则阈值将标量值映射为 1~5 分。"""
        if attribute not in self.rules.get("thresholds", {}):
            return max(1.0, min(5.0, round(value, 2)))
        t = self.rules["thresholds"][attribute]
        v = float(value)
        if v >= t.get("excellent", 4.5):
            return 5.0
        if v >= t.get("good", 3.5):
            return 4.0
        if v >= t.get("fair", 2.5):
            return 3.0
        if v >= t.get("poor", 2.0):
            return 2.0
        return 1.0

    def _score_by_confidence(self, detection: Dict[str, Any], class_name: str) -> Dict[str, float]:
        """仅用置信度生成各类别属性分（回退方案）。"""
        conf = float(detection.get("conf", 0.5))
        base = 1.0 + conf * 4.0
        base = max(1.0, min(5.0, base))
        weights = self.rules.get("weights", {}).get(class_name, {})
        if not weights:
            return {"default": base}
        return {attr: self.score_attribute(attr, base) for attr in weights}

    def _extract_features_for_class(
        self, image: Any, detection: Dict[str, Any], class_name: str
    ) -> Dict[str, float]:
        """根据类别从图像中提取特征（复用抻面 extractor：hand/noodle_rope；tools/soup 用同 ROI 的 hand/noodle 特征代理）。"""
        bbox = detection.get("xyxy", [0, 0, 0, 0])
        if not self.feature_extractor or image is None:
            return {}
        try:
            if class_name == "hand":
                h, w = image.shape[:2]
                return self.feature_extractor.extract_hand_features(image, bbox, image_size=(w, h))
            if class_name == "noodle_rope":
                return self.feature_extractor.extract_noodle_rope_features(image, bbox)
            if class_name == "tools_noodle":
                # 工具+面：用 hand 特征代理 operation_standardization / tool_coordination
                h, w = image.shape[:2]
                raw = self.feature_extractor.extract_hand_features(image, bbox, image_size=(w, h))
                return {
                    "operation_standardization": raw.get("action", 3.0),
                    "tool_coordination": raw.get("coordination", 3.0),
                }
            if class_name == "soup_noodle":
                raw = self.feature_extractor.extract_noodle_rope_features(image, bbox)
                ripeness = (raw.get("elasticity", 0) + raw.get("integrity", 0)) / 2.0
                return {
                    "ripeness": ripeness,
                    "distribution_state": raw.get("integrity", 3.0),
                }
        except Exception:
            pass
        return {}

    def score_detection(
        self,
        detection: Dict[str, Any],
        class_name: str,
        image: Optional[Any] = None,
    ) -> Dict[str, float]:
        """对单个检测框评分：有图像则用图像特征+校准+规则，否则置信度+规则。"""
        scores: Dict[str, float] = {}
        weights = self.rules.get("weights", {}).get(class_name, {})
        if not weights:
            return self._score_by_confidence(detection, class_name)

        if self.use_image_features and image is not None and self.feature_extractor is not None:
            raw_features = self._extract_features_for_class(image, detection, class_name)
            if raw_features:
                # hand 多一个 tool_coordination：用 coordination 与置信度融合
                if class_name == "hand" and "tool_coordination" in weights:
                    conf = float(detection.get("conf", 0.5))
                    base = 1.0 + conf * 4.0
                    raw_features["tool_coordination"] = (raw_features.get("coordination", 3.0) + min(5.0, base)) / 2.0
                # noodle_rope：从 extractor 的 thickness/elasticity/integrity/gloss 映射到规则属性
                if class_name == "noodle_rope":
                    if "ripeness" not in raw_features and "elasticity" in raw_features:
                        raw_features["ripeness"] = (raw_features.get("elasticity", 0) + raw_features.get("integrity", 0)) / 2.0
                    if "soup_adhesion" not in raw_features:
                        raw_features["soup_adhesion"] = raw_features.get("gloss", 3.0)
                    if "noodle_soup_ratio" not in raw_features:
                        raw_features["noodle_soup_ratio"] = raw_features.get("thickness", 3.0)
                    if "distribution_state" not in raw_features:
                        raw_features["distribution_state"] = raw_features.get("integrity", 3.0)
                calibrated = self._calibrate_features(raw_features, class_name)
                for attr in weights:
                    scores[attr] = self.score_attribute(attr, calibrated.get(attr, 3.0))
                if scores:
                    return scores
        # 回退：置信度
        return self._score_by_confidence(detection, class_name)

    def calculate_weighted_score(self, scores: Dict[str, float], class_name: str) -> float:
        """按 rules.weights 加权得到该类 1~5 分。"""
        w = self.rules.get("weights", {}).get(class_name, {})
        if not w or not scores:
            return sum(scores.values()) / len(scores) if scores else 3.0
        total = sum(scores.get(attr, 0) * weight for attr, weight in w.items() if attr in scores)
        weight_sum = sum(w[attr] for attr in scores if attr in w)
        if weight_sum <= 0:
            return 3.0
        return max(1.0, min(5.0, total / weight_sum))

    def score_frame(self, detections: List[Dict[str, Any]], image: Optional[Any] = None) -> Dict[str, Any]:
        """对单帧检测列表评分，返回 class_scores 与 overall_score。"""
        frame_scores: Dict[str, Any] = {
            "detections": [],
            "class_scores": {},
            "overall_score": 0.0,
        }
        overall_weights = self.rules.get("overall_weights", {})
        class_scores: Dict[str, List[float]] = {}

        for det in detections:
            class_name = det.get("class", "")
            if not class_name or class_name not in CLASSES:
                continue
            scores = self.score_detection(det, class_name, image)
            weighted = self.calculate_weighted_score(scores, class_name)
            frame_scores["detections"].append({"class": class_name, "scores": scores, "weighted_score": weighted})
            class_scores.setdefault(class_name, []).append(weighted)

        for cls, lst in class_scores.items():
            frame_scores["class_scores"][cls] = sum(lst) / len(lst)

        total_score = 0.0
        total_weight = 0.0
        for cls, avg in frame_scores["class_scores"].items():
            if cls in overall_weights:
                total_score += avg * overall_weights[cls]
                total_weight += overall_weights[cls]
        if total_weight > 0:
            frame_scores["overall_score"] = total_score / total_weight
        return frame_scores

    def score_video(
        self,
        video_detections: List[Dict[str, Any]],
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """整段视频逐帧评分：有 video_path 则按帧读取图像并做图像特征评分，否则仅用置信度。"""
        frame_scores_list: List[Dict[str, Any]] = []
        video_cap = None
        if self.use_image_features and video_path and Path(video_path).exists():
            try:
                video_cap = cv2.VideoCapture(video_path)
                if not video_cap.isOpened():
                    video_cap = None
            except Exception:
                video_cap = None

        for frame_data in video_detections:
            detections = frame_data.get("detections", [])
            frame_index = frame_data.get("frame_index", len(frame_scores_list))
            frame_image = None
            if video_cap is not None:
                try:
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame_image = video_cap.read()
                    if not ret:
                        frame_image = None
                except Exception:
                    frame_image = None
            fs = self.score_frame(detections, frame_image)
            frame_scores_list.append({"frame_index": frame_index, **fs})

        if video_cap is not None:
            video_cap.release()

        if not frame_scores_list:
            return {
                "total_frames": 0,
                "scored_frames": 0,
                "average_overall_score": 0.0,
                "class_average_scores": {c: 0.0 for c in CLASSES},
                "frame_scores": [],
            }

        overall_scores = [f["overall_score"] for f in frame_scores_list if f.get("overall_score", 0) > 0]
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        class_avg: Dict[str, float] = {}
        for cls in CLASSES:
            vals = [f["class_scores"][cls] for f in frame_scores_list if cls in f.get("class_scores", {})]
            class_avg[cls] = sum(vals) / len(vals) if vals else 0.0

        return {
            "total_frames": len(frame_scores_list),
            "scored_frames": len(overall_scores),
            "average_overall_score": avg_overall,
            "class_average_scores": class_avg,
            "frame_scores": frame_scores_list,
        }
