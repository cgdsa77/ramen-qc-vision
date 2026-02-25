"""
下面及捞面综合评分系统
结合骨架线（手部关键点）与标注数据集进行综合评分，输出格式与抻面一致便于前端复用。
- 权重与类别划分与 data/scores/下面及捞面/scoring_rules.json 统一（与抻面规则体系一致）
- 仅对「有评分文件的抽取帧」评分（与 processed 帧对应）
- 骨架用于手部 position/angle/coordination，action 以标注为主（无 DTW 标准序列）
- 标注维度：手部、面条、工具+汤面，映射为 hand / noodle_rope / noodle_bundle 三类输出
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from .spatial_angle_extractor import SpatialAngleExtractor
except ImportError:
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.scoring.spatial_angle_extractor import SpatialAngleExtractor


def _default_boiling_weights() -> Dict[str, Any]:
    """无规则文件时的默认权重（与 scoring_rules.json 结构对齐的兜底）。"""
    return {
        'class': {'hand': 0.4, 'noodle_rope': 0.35, 'noodle_bundle': 0.25},
        'hand_attributes': {'position': 0.25, 'action': 0.25, 'angle': 0.2, 'coordination': 0.15, 'tool_coordination': 0.15},
        'noodle_rope_attributes': {
            'thickness': 0.15, 'elasticity': 0.15, 'integrity': 0.15, 'ripeness': 0.15,
            'soup_adhesion': 0.15, 'noodle_soup_ratio': 0.15, 'distribution_state': 0.10,
        },
        'noodle_bundle_attributes': {
            'operation_standardization': 0.25, 'tool_coordination': 0.25,
            'ripeness': 0.25, 'distribution_state': 0.25,
        },
    }


class BoilingComprehensiveScorer:
    """下面及捞面综合评分器：骨架 + 标注融合，权重与 scoring_rules.json 统一（与抻面规则一致）"""

    MIN_KEYPOINT_CONFIDENCE = 0.5  # 稍放宽以配合插值帧

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scores_dir = project_root / "data" / "scores" / "下面及捞面"
        self.hand_keypoints_dir = self.scores_dir / "hand_keypoints"
        self.angle_extractor = SpatialAngleExtractor(min_confidence=self.MIN_KEYPOINT_CONFIDENCE)
        self.scoring_rules = self._load_scoring_rules()
        self.weights = self._build_weights_from_rules()

    def _load_scoring_rules(self) -> Dict[str, Any]:
        """加载评分规则（与抻面一致：data/scores/<stage>/scoring_rules.json）。"""
        rules_file = self.scores_dir / "scoring_rules.json"
        if rules_file.exists():
            try:
                with open(rules_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _build_weights_from_rules(self) -> Dict[str, Any]:
        """从 scoring_rules.json 构建权重：overall_weights -> class，weights -> 各类属性；noodle_bundle = tools_noodle + soup_noodle。"""
        if not self.scoring_rules:
            return _default_boiling_weights()

        ow = self.scoring_rules.get("overall_weights", {})
        w = self.scoring_rules.get("weights", {})
        # 类别权重：hand / noodle_rope / noodle_bundle（bundle = tools_noodle + soup_noodle）
        class_w = {
            "hand": float(ow.get("hand", 0.35)),
            "noodle_rope": float(ow.get("noodle_rope", 0.35)),
            "noodle_bundle": float(ow.get("tools_noodle", 0.15)) + float(ow.get("soup_noodle", 0.15)),
        }
        s = sum(class_w.values())
        if s > 0:
            class_w = {k: v / s for k, v in class_w.items()}

        hand_attr = dict(w.get("hand", {}))
        rope_attr = dict(w.get("noodle_rope", {}))
        tools_attr = dict(w.get("tools_noodle", {}))
        soup_attr = dict(w.get("soup_noodle", {}))
        # noodle_bundle = tools_noodle 与 soup_noodle 合并并归一化
        bundle_attr = {**tools_attr, **soup_attr}
        if bundle_attr:
            total = sum(bundle_attr.values())
            if total > 0:
                bundle_attr = {k: v / total for k, v in bundle_attr.items()}

        if not hand_attr:
            hand_attr = _default_boiling_weights()["hand_attributes"]
        if not rope_attr:
            rope_attr = _default_boiling_weights()["noodle_rope_attributes"]
        if not bundle_attr:
            bundle_attr = _default_boiling_weights()["noodle_bundle_attributes"]

        return {
            "class": class_w,
            "hand_attributes": hand_attr,
            "noodle_rope_attributes": rope_attr,
            "noodle_bundle_attributes": bundle_attr,
        }

    def _load_annotation_scores(self, video_name: str) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[int, float]]:
        """加载每帧标注（关键帧+插值），frame_index 为文件名中的帧号（如 1..25）。"""
        annotation_scores: Dict[int, Dict[str, Dict[str, float]]] = {}
        annotation_confidences: Dict[int, float] = {}
        video_scores_dir = self.scores_dir / video_name
        if not video_scores_dir.exists():
            return annotation_scores, annotation_confidences

        for score_file in sorted(video_scores_dir.glob("*_scores.json")):
            try:
                with open(score_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                frame_name = data.get('frame', '')
                if '_' not in frame_name:
                    continue
                try:
                    frame_index = int(frame_name.split('_')[1].split('.')[0])
                except (ValueError, IndexError):
                    continue
                scores = data.get('scores', {})
                frame_scores = {'hand': [], 'noodle_rope': [], 'noodle_bundle': []}

                for det_id, det_s in scores.items():
                    if not isinstance(det_s, dict):
                        continue
                    if 'position' in det_s or 'action' in det_s or 'coordination' in det_s or 'tool_coordination' in det_s:
                        hand_s = {k: float(det_s[k]) for k in ['position', 'action', 'angle', 'coordination', 'tool_coordination'] if k in det_s}
                        if hand_s:
                            frame_scores['hand'].append(hand_s)
                    if any(k in det_s for k in ['thickness', 'elasticity', 'integrity', 'ripeness', 'soup_adhesion', 'noodle_soup_ratio', 'distribution_state']):
                        rope_s = {k: float(det_s[k]) for k in self.weights['noodle_rope_attributes'] if k in det_s}
                        if rope_s:
                            frame_scores['noodle_rope'].append(rope_s)
                    if any(k in det_s for k in self.weights['noodle_bundle_attributes']):
                        bundle_s = {k: float(det_s[k]) for k in self.weights['noodle_bundle_attributes'] if k in det_s}
                        if bundle_s:
                            frame_scores['noodle_bundle'].append(bundle_s)

                final = {}
                for cls, lst in frame_scores.items():
                    if not lst:
                        continue
                    avg = {}
                    for s in lst:
                        for k, v in s.items():
                            avg.setdefault(k, []).append(v)
                    final[cls] = {k: float(np.mean(v)) for k, v in avg.items()}
                if final:
                    annotation_scores[frame_index] = final
                    annotation_confidences[frame_index] = 0.9 if data.get('interpolated') else 0.95
            except Exception:
                continue
        return annotation_scores, annotation_confidences

    def _assess_detection_quality(self, frames_data: List[Dict]) -> Dict[str, float]:
        total = len(frames_data)
        if total == 0:
            return {'detection_rate': 0.0, 'both_hands_rate': 0.0, 'quality_score': 0.0}
        detected = sum(1 for f in frames_data if f.get('hands'))
        both = sum(1 for f in frames_data if len(f.get('hands', [])) >= 2)
        return {
            'detection_rate': detected / total,
            'both_hands_rate': both / total,
            'quality_score': (detected / total) * 0.6 + (both / total) * 0.4,
        }

    def _score_position_from_angles(self, frame_angles: Dict, image_size: Tuple[int, int]) -> float:
        if 'bilateral' in frame_angles and frame_angles['bilateral']:
            a = frame_angles['bilateral'].get('wrist_connection_angle')
            if a is not None and not np.isnan(a):
                dev = min(abs(a), abs(a - 180))
                return float(max(0, min(5, 5.0 * np.exp(-(dev / 45.0) ** 2))))
        return 3.0

    def _score_angle_from_angles(self, frame_angles: Dict) -> float:
        for k, v in frame_angles.items():
            if k.startswith('hand_') and isinstance(v, dict) and 'palm_direction_angle' in v:
                a = v['palm_direction_angle']
                if a is not None and not np.isnan(a):
                    return float(max(0, min(5, 5.0 * np.exp(-(abs(a - 90) / 45.0) ** 2))))
        return 3.0

    def _score_coordination_from_angles(self, frame_angles: Dict) -> float:
        if 'bilateral' in frame_angles and frame_angles['bilateral']:
            d = frame_angles['bilateral'].get('palm_angle_difference')
            if d is not None and not np.isnan(d):
                return float(max(0, min(5, 5.0 * np.exp(-(d / 45.0) ** 2))))
        return 3.0

    def _frame_skeleton_quality(self, frames_data: List[Dict], idx: int) -> float:
        if idx < 0 or idx >= len(frames_data):
            return 0.0
        hands = frames_data[idx].get('hands', [])
        if not hands:
            return 0.0
        q = 0.0
        for h in hands:
            kps = h.get('keypoints', [])
            if len(kps) >= 21:
                c = np.mean([kp.get('confidence', 0) for kp in kps])
                q = max(q, c)
        return float(min(1.0, q))

    def _fuse_hand_scores(self, skeleton_scores: Dict[str, float], annotation_scores: Optional[Dict[str, float]],
                          frame_skeleton_quality: float = 1.0, annotation_confidence: float = 0.9) -> Dict[str, float]:
        S = max(0, min(1, frame_skeleton_quality))
        A = max(0, min(1, annotation_confidence))
        denom = S + A + 0.01
        w_s = max(0.2, min(0.8, S / denom))
        w_a = 1.0 - w_s
        out = {}
        for attr in ['position', 'angle', 'action', 'coordination']:
            sk = skeleton_scores.get(attr, 3.0)
            ann = annotation_scores.get(attr) if annotation_scores else None
            if ann is not None:
                out[attr] = float(max(0, min(5, sk * w_s + ann * w_a)))
            else:
                out[attr] = float(max(0, min(5, sk * S)))
        # 规则中 hand 含 tool_coordination 等仅来自标注的属性
        for attr, val in (annotation_scores or {}).items():
            if attr not in out and isinstance(val, (int, float)):
                out[attr] = float(max(0, min(5, val)))
        return out

    def score_video(self, video_name: str) -> Dict[str, Any]:
        """对单个视频评分（仅对存在 *_scores.json 的抽取帧）。"""
        annotation_scores, annotation_confidences = self._load_annotation_scores(video_name)
        if not annotation_scores:
            return {'error': f'无评分数据: {video_name}', 'video': video_name}

        keypoints_file = self.hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            return {'error': f'骨架线数据不存在: {keypoints_file}', 'video': video_name}

        with open(keypoints_file, 'r', encoding='utf-8') as f:
            kp_data = json.load(f)
        all_frames = kp_data.get('frames', [])
        total_video_frames = len(all_frames)
        width = kp_data.get('width', 1280)
        height = kp_data.get('height', 720)
        image_size = (width, height)

        sorted_sample_indices = sorted(annotation_scores.keys())
        n_score_frames = len(sorted_sample_indices)
        if n_score_frames == 0:
            return {'error': '无有效评分帧', 'video': video_name}

        frames_data: List[Dict] = []
        step = (total_video_frames - 1) / max(1, n_score_frames - 1) if n_score_frames > 1 else 0
        for i, sample_idx in enumerate(sorted_sample_indices):
            if n_score_frames == 1:
                video_idx = 0
            else:
                video_idx = int(round((sample_idx - 1) * step))
            video_idx = max(0, min(video_idx, total_video_frames - 1))
            frames_data.append(all_frames[video_idx])

        detection_quality = self._assess_detection_quality(frames_data)
        test_angle_sequence = self.angle_extractor.extract_angle_sequence(frames_data)
        if len(test_angle_sequence) < len(frames_data):
            test_angle_sequence = test_angle_sequence + [{}] * (len(frames_data) - len(test_angle_sequence))

        frame_quality_list = [self._frame_skeleton_quality(frames_data, i) for i in range(len(frames_data))]
        frame_scores_list: List[Dict] = []
        hand_scores_list: List[Dict] = []
        rope_scores_list: List[Dict] = []
        bundle_scores_list: List[Dict] = []

        for i, sample_idx in enumerate(sorted_sample_indices):
            frame_angles = test_angle_sequence[i] if i < len(test_angle_sequence) else {}
            S_quality = frame_quality_list[i] if i < len(frame_quality_list) else 0.0
            A_conf = annotation_confidences.get(sample_idx, 0.9)

            pos_s = self._score_position_from_angles(frame_angles, image_size)
            ang_s = self._score_angle_from_angles(frame_angles)
            coo_s = self._score_coordination_from_angles(frame_angles)
            ann = annotation_scores.get(sample_idx, {})
            hand_ann = ann.get('hand') or {}
            action_s = hand_ann.get('action')
            if action_s is None:
                action_s = hand_ann.get('tool_coordination', 3.0)
            if action_s is None:
                action_s = 3.0

            skeleton_hand = {'position': pos_s, 'angle': ang_s, 'action': float(action_s), 'coordination': coo_s}
            hand_scores = self._fuse_hand_scores(skeleton_hand, hand_ann, S_quality, A_conf)
            hand_scores_list.append(hand_scores)

            rope_ann = ann.get('noodle_rope') or {}
            rope_scores = {}
            for attr in self.weights['noodle_rope_attributes']:
                if attr in rope_ann:
                    rope_scores[attr] = float(max(0, min(5, rope_ann[attr])))
            if not rope_scores:
                rope_scores = {'default': 3.0}
            rope_scores_list.append(rope_scores)

            bundle_ann = ann.get('noodle_bundle') or {}
            bundle_scores = {}
            for attr in self.weights['noodle_bundle_attributes']:
                if attr in bundle_ann:
                    bundle_scores[attr] = float(max(0, min(5, bundle_ann[attr])))
            if not bundle_scores:
                bundle_scores = {'default': 3.0}
            bundle_scores_list.append(bundle_scores)

            frame_scores_list.append({
                'frame_index': sample_idx,
                'hand': hand_scores,
                'noodle_rope': rope_scores,
                'noodle_bundle': bundle_scores,
            })

        def weighted_avg(scores_list: List[Dict], attrs: Dict[str, float]) -> float:
            if not scores_list:
                return 0.0
            total, w_sum = 0.0, 0.0
            for s in scores_list:
                for attr, w in attrs.items():
                    val = s.get(attr, s.get('default'))
                    if val is not None:
                        total += float(val) * w
                        w_sum += w
            return total / w_sum if w_sum > 0 else 0.0

        hand_avg = weighted_avg(hand_scores_list, self.weights['hand_attributes'])
        rope_avg = weighted_avg(rope_scores_list, self.weights['noodle_rope_attributes'])
        bundle_avg = weighted_avg(bundle_scores_list, self.weights['noodle_bundle_attributes'])
        if rope_avg == 0 and rope_scores_list:
            rope_avg = float(np.mean([s.get('default', 3.0) for s in rope_scores_list]))
        if bundle_avg == 0 and bundle_scores_list:
            bundle_avg = float(np.mean([s.get('default', 3.0) for s in bundle_scores_list]))
        class_weights = self.weights['class']
        total_score = (
            hand_avg * class_weights['hand'] +
            rope_avg * class_weights['noodle_rope'] +
            bundle_avg * class_weights['noodle_bundle']
        )

        return {
            'video': video_name,
            'total_frames': total_video_frames,
            'scored_frames': n_score_frames,
            'detection_quality': detection_quality,
            'dtw_result': None,
            'frame_scores': frame_scores_list[:100],
            'class_average_scores': {'hand': hand_avg, 'noodle_rope': rope_avg, 'noodle_bundle': bundle_avg},
            'hand_score': hand_avg,
            'noodle_rope_score': rope_avg,
            'noodle_bundle_score': bundle_avg,
            'total_score': float(max(0, min(5, total_score))),
            'total_confidence': 0.9,
        }
