"""
增强的综合评分系统
基于论文《基于深度学习的乒乓球姿态动作评分方法》优化
集成空间角度特征、改进DTW算法、Mean Shift权重估计
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

try:
    from .spatial_angle_extractor import SpatialAngleExtractor
    from .improved_dtw import ImprovedDTW
    from .keypoint_weight_estimator import KeypointWeightEstimator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.scoring.spatial_angle_extractor import SpatialAngleExtractor
    from src.scoring.improved_dtw import ImprovedDTW
    from src.scoring.keypoint_weight_estimator import KeypointWeightEstimator


class EnhancedComprehensiveScorer:
    """增强的综合评分器（优化版：滑窗DTW、双置信度融合、类别置信度调权）"""
    
    # 滑窗DTW 参数
    DTW_WINDOW_SIZE = 50
    DTW_STEP_SIZE = 10
    # 关键点置信度阈值
    MIN_KEYPOINT_CONFIDENCE = 0.7
    # 关键帧间隔超过此秒数则插值分×0.8（需结合 fps 换算为帧数，此处用帧数近似：5秒≈150帧@30fps）
    KEYFRAME_GAP_FRAMES_SPARSE = 150
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scores_dir = project_root / "data" / "scores" / "抻面"
        self.hand_keypoints_dir = self.scores_dir / "hand_keypoints"
        
        # 初始化核心组件（置信度≥0.7 才参与角度计算）
        self.angle_extractor = SpatialAngleExtractor(min_confidence=self.MIN_KEYPOINT_CONFIDENCE)
        self.dtw = ImprovedDTW(window_size=None, distance_metric='euclidean', max_sequence_length=500)
        self.weight_estimator = KeypointWeightEstimator(bandwidth=0.5)
        
        # 标准视频：默认含 cm1–cm5 与新增示范 cm13–cm17；可由 data/scores/抻面/dtw_standard_videos.json 覆盖
        self.standard_video_names = self._load_dtw_standard_video_names()
        self.standard_angle_sequences = self._load_standard_angle_sequences()
        self.keypoint_weights = self._load_keypoint_weights()
        
        # DTW 距离校准：标准序列两两距离的 95 分位数作为 max_expected_distance
        self._dtw_distance_calibration = self._calibrate_dtw_distance()
        
        self.scoring_rules = self._load_scoring_rules()
        
        # 权重配置（优化版：手部 action 0.35、position 0.20）
        self.weights = {
            'class': {
                'hand': 0.4,
                'noodle_rope': 0.5,
                'noodle_bundle': 0.1
            },
            'hand_attributes': {
                'position': 0.20,
                'action': 0.35,
                'angle': 0.20,
                'coordination': 0.25
            },
            'noodle_rope_attributes': {
                'thickness': 0.30,
                'elasticity': 0.25,
                'gloss': 0.20,
                'integrity': 0.25
            }
        }
        # 类别置信度调权：最低权重
        self.class_weight_min = {'hand': 0.3, 'noodle_rope': 0.3, 'noodle_bundle': 0.05}
        self.class_confidence_threshold = {'hand': 0.5, 'noodle_rope': 0.5, 'noodle_bundle': 0.3}
        self.class_confidence_penalty = {'hand': 0.8, 'noodle_rope': 0.8, 'noodle_bundle': 0.5}
    
    def _load_dtw_standard_video_names(self) -> List[str]:
        """DTW 参照序列对应的示范视频名列表（需存在 hand_keypoints_*.json 才会载入角度序列）。"""
        default = [
            "cm1", "cm2", "cm3", "cm4", "cm5",
            "cm13", "cm14", "cm15", "cm16", "cm17",
        ]
        cfg = self.scores_dir / "dtw_standard_videos.json"
        if cfg.exists():
            try:
                with open(cfg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                names = data.get("standard_video_names")
                if isinstance(names, list) and len(names) > 0:
                    return [str(x).strip() for x in names if str(x).strip()]
            except Exception:
                pass
        return default

    def _load_scoring_rules(self) -> Dict:
        """加载评分规则"""
        rules_file = self.scores_dir / "scoring_rules.json"
        if rules_file.exists():
            with open(rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_standard_angle_sequences(self) -> List[List[Dict[str, float]]]:
        """
        加载标准动作序列的角度数据（预处理过滤置信度＜0.7 的帧：由角度提取器内部过滤）
        从标准视频列表（dtw_standard_videos.json，含 cm1–cm5 与 cm13–cm17 等）提取角度序列
        """
        sequences = []
        for video_name in self.standard_video_names:
            keypoints_file = self.hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
            if not keypoints_file.exists():
                continue
            with open(keypoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            frames_data = data.get('frames', [])
            angle_sequence = self.angle_extractor.extract_angle_sequence(frames_data)
            if angle_sequence:
                sequences.append(angle_sequence)
        return sequences

    def _calibrate_dtw_distance(self) -> Dict[str, float]:
        """
        预计算标准序列两两 DTW 距离分布，取 95 分位数作为 max_expected_distance，
        用于线性归一化分数映射。返回 {'min': d_min, 'max': d_max_95, 'percentile_95': d_max_95}。
        """
        out = {'min': 0.0, 'max': 50.0, 'percentile_95': 50.0}
        seqs = self.standard_angle_sequences
        if len(seqs) < 2 or not self.keypoint_weights:
            return out
        distances = []
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                d = self.dtw.compute_angle_sequence_distance(seqs[i], seqs[j], self.keypoint_weights)
                n = max(len(seqs[i]), len(seqs[j]))
                if n > 0:
                    d_norm = self.dtw.normalize_distance(d, n)
                    distances.append(d_norm)
        if distances:
            arr = np.array(distances)
            out['min'] = float(np.min(arr))
            out['max'] = float(np.max(arr))
            out['percentile_95'] = float(np.percentile(arr, 95))
        return out
    
    def _load_keypoint_weights(self) -> Dict[str, float]:
        """
        加载关键关节点权重
        
        如果权重文件不存在，从标准视频计算
        """
        weights_file = self.scores_dir / "keypoint_weights.json"
        
        if weights_file.exists():
            with open(weights_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 从标准视频计算权重
            if self.standard_angle_sequences:
                weights = self.weight_estimator.estimate_weights_from_standard_videos(
                    self.standard_angle_sequences
                )
                # 保存权重
                weights_file.parent.mkdir(parents=True, exist_ok=True)
                with open(weights_file, 'w', encoding='utf-8') as f:
                    json.dump(weights, f, indent=2, ensure_ascii=False)
                return weights
        
        return {}
    
    def extract_angle_sequence_from_video(self, video_name: str) -> List[Dict[str, float]]:
        """从视频提取角度序列"""
        keypoints_file = self.hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            return []
        
        with open(keypoints_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        frames_data = data.get('frames', [])
        return self.angle_extractor.extract_angle_sequence(frames_data)
    
    def calculate_dtw_score(self, test_sequence: List[Dict[str, float]],
                           standard_sequences: List[List[Dict[str, float]]]) -> Dict[str, Any]:
        """
        基于DTW距离计算评分
        
        Args:
            test_sequence: 测试动作序列的角度数据
            standard_sequences: 标准动作序列列表
        
        Returns:
            评分结果字典
        """
        if len(test_sequence) == 0 or len(standard_sequences) == 0:
            return {
                'score': 0.0,
                'distance': float('inf'),
                'matched_sequence': None
            }
        
        # 计算与所有标准序列的DTW距离
        distances = []
        for std_sequence in standard_sequences:
            distance = self.dtw.compute_angle_sequence_distance(
                test_sequence,
                std_sequence,
                self.keypoint_weights
            )
            # 归一化距离
            normalized_distance = self.dtw.normalize_distance(
                distance,
                max(len(test_sequence), len(std_sequence))
            )
            distances.append(normalized_distance)
        
        # 找到最小距离（最匹配的标准序列）
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        
        # 线性归一化分数映射（基于校准的 d_min / d_max）
        d_min = self._dtw_distance_calibration.get('min', 0.0)
        d_max = self._dtw_distance_calibration.get('percentile_95', 50.0)
        if d_max <= d_min:
            d_max = d_min + 1e-6
        score = 5.0 * (1.0 - min(1.0, (min_distance - d_min) / (d_max - d_min)))
        score = max(0.0, min(5.0, score))
        
        return {
            'score': float(score),
            'distance': float(min_distance),
            'matched_sequence': min_index,
            'all_distances': [float(d) for d in distances]
        }
    
    def _sliding_window_dtw_scores(self, test_angle_sequence: List[Dict[str, float]],
                                   num_frames: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        滑窗 DTW：窗口 50 帧、步长 10 帧，计算每个窗口与标准序列的最小距离并转为 0-5 分，
        再插值得到每帧的 action 得分。
        Returns:
            frame_action_scores: 长度 num_frames 的列表，每帧对应一个 action 分
            window_results: 每个窗口的 {start, end, score, distance, matched_sequence}
        """
        w, step = self.DTW_WINDOW_SIZE, self.DTW_STEP_SIZE
        n = len(test_angle_sequence)
        if n == 0 or num_frames <= 0:
            return [0.0] * num_frames, []
        d_min = self._dtw_distance_calibration.get('min', 0.0)
        d_max = self._dtw_distance_calibration.get('percentile_95', 50.0)
        if d_max <= d_min:
            d_max = d_min + 1e-6
        
        window_results = []
        for start in range(0, max(1, n - w + 1), step):
            end = min(start + w, n)
            window_seq = test_angle_sequence[start:end]
            if len(window_seq) < 2:
                continue
            distances = []
            for std_seq in self.standard_angle_sequences:
                d = self.dtw.compute_angle_sequence_distance(window_seq, std_seq, self.keypoint_weights)
                norm_len = max(len(window_seq), len(std_seq))
                d_norm = self.dtw.normalize_distance(d, norm_len) if norm_len else d
                distances.append(d_norm)
            if not distances:
                continue
            min_d = min(distances)
            min_idx = distances.index(min_d)
            score = 5.0 * (1.0 - min(1.0, (min_d - d_min) / (d_max - d_min)))
            score = max(0.0, min(5.0, score))
            window_results.append({
                'start': start, 'end': end,
                'score': float(score), 'distance': float(min_d), 'matched_sequence': min_idx
            })
        
        # 插值到每帧：每帧取覆盖该帧的窗口得分的平均，若无则用最近窗口
        frame_action_scores = [0.0] * num_frames
        if not window_results:
            return frame_action_scores, []
        for i in range(num_frames):
            frame_idx = int(i * (n - 1) / (num_frames - 1)) if num_frames > 1 else 0
            scores_at_i = []
            for wr in window_results:
                if wr['start'] <= frame_idx < wr['end']:
                    scores_at_i.append(wr['score'])
            if scores_at_i:
                frame_action_scores[i] = float(np.mean(scores_at_i))
            else:
                # 最近窗口
                best_dist = float('inf')
                best_score = 0.0
                for wr in window_results:
                    mid = (wr['start'] + wr['end']) / 2
                    d = abs(frame_idx - mid)
                    if d < best_dist:
                        best_dist = d
                        best_score = wr['score']
                frame_action_scores[i] = best_score
        return frame_action_scores, window_results
    
    def _frame_skeleton_quality(self, frames_data: List[Dict[str, Any]], frame_idx: int) -> float:
        """单帧骨架质量分 0~1：双手 1.0，单手约 0.6，无手 0。"""
        if frame_idx < 0 or frame_idx >= len(frames_data):
            return 0.0
        hands = frames_data[frame_idx].get('hands', [])
        if not hands:
            return 0.0
        if len(hands) >= 2:
            return 1.0
        return 0.6

    def _get_annotation_confidence_for_frame(self, frame_idx: int,
                                              annotation_scores: Optional[Dict],
                                              annotation_confidences: Optional[Dict[int, float]]) -> float:
        """获取当前帧的标注置信度 0~1；无则插值前后关键帧或默认 0.8。"""
        if not annotation_confidences:
            return 0.8
        if frame_idx in annotation_confidences:
            return float(max(0.0, min(1.0, annotation_confidences[frame_idx])))
        key_frames = sorted(annotation_confidences.keys())
        prev = next((k for k in reversed(key_frames) if k <= frame_idx), None)
        nex = next((k for k in key_frames if k > frame_idx), None)
        if prev is not None and nex is not None:
            alpha = (frame_idx - prev) / (nex - prev) if nex != prev else 0
            return float((1 - alpha) * annotation_confidences[prev] + alpha * annotation_confidences[nex])
        if prev is not None:
            return float(annotation_confidences[prev])
        if nex is not None:
            return float(annotation_confidences[nex])
        return 0.8

    def _assess_detection_quality(self, frames_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估骨架线检测质量
        
        Returns:
            包含检测质量指标的字典
        """
        total_frames = len(frames_data)
        if total_frames == 0:
            return {'detection_rate': 0.0, 'both_hands_rate': 0.0, 'quality_score': 0.0}
        
        detected_frames = 0
        both_hands_frames = 0
        
        for frame_data in frames_data:
            hands = frame_data.get('hands', [])
            if hands:
                detected_frames += 1
                if len(hands) >= 2:
                    both_hands_frames += 1
        
        detection_rate = detected_frames / total_frames
        both_hands_rate = both_hands_frames / total_frames
        
        # 质量评分：综合考虑检测率和双手检测率
        quality_score = detection_rate * 0.6 + both_hands_rate * 0.4
        
        return {
            'detection_rate': detection_rate,
            'both_hands_rate': both_hands_rate,
            'quality_score': quality_score
        }
    
    def calculate_comprehensive_score(self, video_name: str,
                                     annotation_scores: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
                                     annotation_confidences: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        计算综合评分（逐帧计算，滑窗 DTW 帧级 action、双置信度融合、类别置信度二次调权）
        
        Args:
            video_name: 视频名称
            annotation_scores: 关键帧标注评分 {frame_index: {class: {attribute: score}}}
            annotation_confidences: 关键帧标注置信度 0~1，{frame_index: confidence}，缺省时按 0.8
        Returns:
            综合评分结果（含 total_score, total_confidence, scoring_log 等）
        """
        if annotation_confidences is None:
            annotation_confidences = {}
        # 加载骨架线数据
        keypoints_file = self.hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            return {
                'error': f'骨架线数据文件不存在: {keypoints_file}',
                'video': video_name
            }
        
        with open(keypoints_file, 'r', encoding='utf-8') as f:
            keypoints_data = json.load(f)
        
        frames_data = keypoints_data.get('frames', [])
        width = keypoints_data.get('width', 1280)
        height = keypoints_data.get('height', 720)
        image_size = (width, height)
        
        # 评估检测质量
        detection_quality = self._assess_detection_quality(frames_data)
        
        # 提取角度序列（置信度过滤由 angle_extractor 内部完成）
        test_angle_sequence = self.angle_extractor.extract_angle_sequence(frames_data)
        
        if len(test_angle_sequence) == 0:
            return {
                'error': f'无法提取角度序列: {video_name}',
                'video': video_name
            }
        
        # 滑窗 DTW 得到帧级 action 得分
        num_frames = len(frames_data)
        frame_action_scores, dtw_window_results = self._sliding_window_dtw_scores(
            test_angle_sequence, num_frames
        )
        # 整段 DTW（用于兼容输出）
        dtw_result = self.calculate_dtw_score(test_angle_sequence, self.standard_angle_sequences)
        
        angle_changes = self.angle_extractor.calculate_angle_changes(test_angle_sequence)
        
        # 预计算每帧骨架质量分（用于手部加权平均与融合）
        frame_quality_list = [
            self._frame_skeleton_quality(frames_data, i) for i in range(num_frames)
        ]
        
        frame_scores_list = []
        hand_scores_list = []
        rope_scores_list = []
        bundle_scores_list = []
        scoring_log = []  # 评分日志：每帧/每类数据源、质量分、融合权重
        
        for frame_idx, frame_data in enumerate(frames_data):
            hands = frame_data.get('hands', [])
            frame_angles = test_angle_sequence[frame_idx] if frame_idx < len(test_angle_sequence) else {}
            S_quality = frame_quality_list[frame_idx] if frame_idx < len(frame_quality_list) else 0.0
            
            # 手部单帧骨架分（高斯衰减）
            position_score = self._score_position_from_angles_single_frame(frame_angles, image_size)
            angle_score = self._score_angle_from_angles_single_frame(frame_angles)
            coordination_score = self._score_coordination_from_angles_single_frame(frame_angles)
            action_score = frame_action_scores[frame_idx] if frame_idx < len(frame_action_scores) else 0.0
            
            hand_annotation = None
            rope_annotation = None
            bundle_annotation = None
            A_confidence = self._get_annotation_confidence_for_frame(
                frame_idx, annotation_scores, annotation_confidences
            )
            
            if annotation_scores:
                if frame_idx in annotation_scores:
                    frame_ann = annotation_scores[frame_idx]
                    hand_annotation = frame_ann.get('hand')
                    rope_annotation = frame_ann.get('noodle_rope')
                    bundle_annotation = frame_ann.get('noodle_bundle')
                else:
                    hand_annotation = {}
                    rope_annotation = {}
                    bundle_annotation = {}
                    for attr in ['position', 'angle', 'action', 'coordination']:
                        s = self._interpolate_annotation_score(
                            frame_idx, annotation_scores, 'hand', attr,
                            keyframe_gap_penalty_frames=self.KEYFRAME_GAP_FRAMES_SPARSE
                        )
                        if s is not None:
                            hand_annotation[attr] = s
                    for attr in ['thickness', 'elasticity', 'gloss', 'integrity']:
                        s = self._interpolate_annotation_score(
                            frame_idx, annotation_scores, 'noodle_rope', attr,
                            keyframe_gap_penalty_frames=self.KEYFRAME_GAP_FRAMES_SPARSE
                        )
                        if s is not None:
                            rope_annotation[attr] = s
                    for attr in ['tightness', 'uniformity']:
                        s = self._interpolate_annotation_score(
                            frame_idx, annotation_scores, 'noodle_bundle', attr,
                            keyframe_gap_penalty_frames=self.KEYFRAME_GAP_FRAMES_SPARSE
                        )
                        if s is not None:
                            bundle_annotation[attr] = s
            
            hand_scores = self._fuse_hand_scores(
                {
                    'position': position_score,
                    'angle': angle_score,
                    'action': action_score,
                    'coordination': coordination_score
                },
                hand_annotation,
                detection_quality,
                frame_skeleton_quality=S_quality,
                annotation_confidence=A_confidence
            )
            
            rope_scores = self._calculate_noodle_rope_scores(
                rope_annotation, frame_angles, coordination_skeleton=coordination_score
            )
            bundle_scores = self._calculate_noodle_bundle_scores(
                bundle_annotation, rope_scores
            )
            
            frame_scores_list.append({
                'frame_index': frame_idx,
                'hand': hand_scores,
                'noodle_rope': rope_scores,
                'noodle_bundle': bundle_scores
            })
            if hand_scores:
                hand_scores_list.append((hand_scores, S_quality))
            if rope_scores:
                rope_scores_list.append(rope_scores)
            if bundle_scores:
                bundle_scores_list.append(bundle_scores)
            
            if len(scoring_log) < 200:
                scoring_log.append({
                    'frame_index': frame_idx,
                    'skeleton_quality': S_quality,
                    'annotation_confidence': A_confidence,
                    'hand': hand_scores,
                    'noodle_rope': rope_scores,
                    'noodle_bundle': bundle_scores
                })
        
        def calculate_class_average(scores_list, attribute_weights):
            if not scores_list:
                return 0.0
            avg_scores = {}
            for attr in attribute_weights.keys():
                values = [s.get(attr, 0) for s in scores_list if attr in s]
                if values:
                    avg_scores[attr] = float(np.mean(values))
            if not avg_scores:
                return 0.0
            return sum(avg_scores[attr] * attribute_weights[attr]
                      for attr in avg_scores.keys() if attr in attribute_weights)
        
        # 手部：按帧级骨架质量加权平均
        if hand_scores_list:
            weighted_sum = 0.0
            weight_total = 0.0
            for scores_dict, q in hand_scores_list:
                frame_hand = sum(scores_dict.get(a, 0) * self.weights['hand_attributes'].get(a, 0)
                                 for a in self.weights['hand_attributes'])
                w = max(0.01, q)
                weighted_sum += frame_hand * w
                weight_total += w
            hand_avg_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        else:
            hand_avg_score = 0.0
        
        hand_scores_flat = [s for s, _ in hand_scores_list]
        rope_avg_score = calculate_class_average(rope_scores_list, self.weights['noodle_rope_attributes'])
        bundle_avg_score = calculate_class_average(bundle_scores_list, {'tightness': 0.5, 'uniformity': 0.5})
        
        # 类别置信度：hand = 有效帧占比 × 平均骨架质量；rope/bundle = 有效帧占比 × 平均标注置信度
        n_frames = len(frames_data)
        hand_valid_ratio = len(hand_scores_flat) / n_frames if n_frames else 0
        rope_valid_ratio = len(rope_scores_list) / n_frames if n_frames else 0
        bundle_valid_ratio = len(bundle_scores_list) / n_frames if n_frames else 0
        avg_skeleton_quality = float(np.mean(frame_quality_list)) if frame_quality_list else 0
        key_frames_with_ann = sorted(annotation_scores.keys()) if annotation_scores else []
        avg_ann_conf = (np.mean([annotation_confidences.get(k, 0.8) for k in key_frames_with_ann])
                        if key_frames_with_ann else 0.8)
        
        class_confidence = {
            'hand': hand_valid_ratio * (avg_skeleton_quality if hand_valid_ratio > 0 else 0),
            'noodle_rope': rope_valid_ratio * (avg_ann_conf if rope_valid_ratio > 0 else 0),
            'noodle_bundle': bundle_valid_ratio * (avg_ann_conf if bundle_valid_ratio > 0 else 0)
        }
        
        available_classes = {}
        if hand_scores_list:
            available_classes['hand'] = (hand_avg_score, class_confidence['hand'])
        if rope_scores_list:
            available_classes['noodle_rope'] = (rope_avg_score, class_confidence['noodle_rope'])
        if bundle_scores_list:
            available_classes['noodle_bundle'] = (bundle_avg_score, class_confidence['noodle_bundle'])
        
        # 动态权重：先按基础权重归一化，再按类别置信度二次调权（低置信仅降权不剔除）
        base_weights = self.weights['class']
        total_weight_base = sum(base_weights[cls] for cls in available_classes.keys())
        normalized_base = {cls: base_weights[cls] / total_weight_base for cls in available_classes.keys()} if total_weight_base > 0 else {}
        
        adjusted_weights = {}
        for cls in available_classes.keys():
            w = normalized_base.get(cls, 0)
            conf = class_confidence[cls]
            th = self.class_confidence_threshold.get(cls, 0.5)
            penalty = self.class_confidence_penalty.get(cls, 0.8)
            if conf < th:
                w = w * penalty
            min_w = self.class_weight_min.get(cls, 0)
            adjusted_weights[cls] = max(min_w, w)
        
        total_adj = sum(adjusted_weights.values())
        if total_adj > 0:
            final_weights = {cls: adjusted_weights[cls] / total_adj for cls in available_classes.keys()}
        else:
            final_weights = normalized_base
        
        total_score = sum(available_classes[cls][0] * final_weights[cls] for cls in available_classes.keys())
        total_score = float(max(0.0, min(5.0, total_score)))
        total_confidence = sum(class_confidence[cls] * final_weights[cls] for cls in available_classes.keys())
        total_confidence = float(max(0.0, min(1.0, total_confidence)))
        
        # 将 inf/nan 转为 None，避免 JSON 序列化出非法 Infinity/NaN
        def _sanitize_json(obj):
            if isinstance(obj, dict):
                return {k: _sanitize_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize_json(x) for x in obj]
            if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
                return None
            return obj
        
        dtw_windows_safe = _sanitize_json(dtw_window_results[:50])
        dtw_result_safe = _sanitize_json(dtw_result)
        
        return {
            'video': video_name,
            'total_frames': len(frames_data),
            'scored_frames': len([s for s in frame_scores_list if any(s.values())]),
            'detection_quality': detection_quality,
            'dtw_result': dtw_result_safe,
            'dtw_window_results': dtw_windows_safe,
            'angle_changes': {k: {
                'mean': float(np.mean(v)) if v else 0.0,
                'std': float(np.std(v)) if v else 0.0,
                'count': len(v)
            } for k, v in angle_changes.items()},
            'class_average_scores': {
                'hand': hand_avg_score,
                'noodle_rope': rope_avg_score,
                'noodle_bundle': bundle_avg_score
            },
            'class_confidence': class_confidence,
            'available_classes': list(available_classes.keys()),
            'dynamic_weights': final_weights,
            'hand_score': hand_avg_score,
            'noodle_rope_score': rope_avg_score,
            'noodle_bundle_score': bundle_avg_score,
            'total_score': total_score,
            'total_confidence': total_confidence,
            'keypoint_weights': self.keypoint_weights,
            'frame_scores': frame_scores_list[:100],
            'scoring_log': scoring_log
        }
    
    def _score_position_from_angles_single_frame(self, frame_angles: Dict[str, Any],
                                                image_size: Tuple[int, int]) -> float:
        """基于角度特征计算单帧位置评分（高斯衰减：score = 5*exp(-(deviation/45)^2)）"""
        if 'bilateral' in frame_angles:
            bilateral = frame_angles['bilateral']
            if 'wrist_connection_angle' in bilateral:
                angle = bilateral['wrist_connection_angle']
                deviation = min(abs(angle - 0.0), abs(angle - 180.0))
                score = 5.0 * math.exp(-(deviation / 45.0) ** 2)
                return float(max(0.0, min(5.0, score)))
        return 0.0

    def _score_angle_from_angles_single_frame(self, frame_angles: Dict[str, Any]) -> float:
        """基于角度特征计算单帧角度评分（高斯衰减，理想 90°）"""
        for hand_key in frame_angles.keys():
            if hand_key.startswith('hand_') and isinstance(frame_angles[hand_key], dict):
                if 'palm_direction_angle' in frame_angles[hand_key]:
                    angle = frame_angles[hand_key]['palm_direction_angle']
                    deviation = abs(angle - 90.0)
                    score = 5.0 * math.exp(-(deviation / 45.0) ** 2)
                    return float(max(0.0, min(5.0, score)))
        return 0.0

    def _score_coordination_from_angles_single_frame(self, frame_angles: Dict[str, Any]) -> float:
        """基于角度特征计算单帧协调性评分（高斯衰减）；单手基础分 3.0，无手 1.0"""
        if 'bilateral' in frame_angles:
            bilateral = frame_angles['bilateral']
            if 'palm_angle_difference' in bilateral:
                diff = bilateral['palm_angle_difference']
                score = 5.0 * math.exp(-(diff / 45.0) ** 2)
                return float(max(0.0, min(5.0, score)))
        hand_keys = [k for k in frame_angles.keys() if k.startswith('hand_')]
        if hand_keys:
            for hand_key in hand_keys:
                hand_data = frame_angles.get(hand_key, {})
                if isinstance(hand_data, dict) and 'palm_direction_angle' in hand_data:
                    angle = hand_data['palm_direction_angle']
                    deviation = abs(angle - 90.0)
                    score = 3.0 + 1.0 * (1.0 - min(1.0, deviation / 90.0))
                    return float(max(2.0, min(4.0, score)))
        return 1.0
    
    def _fuse_hand_scores(self, skeleton_scores: Dict[str, float],
                          annotation_scores: Optional[Dict[str, float]],
                          detection_quality: Optional[Dict[str, float]] = None,
                          frame_skeleton_quality: float = 1.0,
                          annotation_confidence: float = 0.8) -> Dict[str, float]:
        """双置信度融合：w_skeleton = S_quality/(S_quality+A_confidence+0.01)，约束在 [0.2,0.8]"""
        S_quality = max(0.0, min(1.0, frame_skeleton_quality))
        A_confidence = max(0.0, min(1.0, annotation_confidence))
        denom = S_quality + A_confidence + 0.01
        w_skeleton = S_quality / denom
        w_skeleton = max(0.2, min(0.8, w_skeleton))
        w_annotation = 1.0 - w_skeleton
        
        fused_scores = {}
        for attr in ['position', 'angle', 'action', 'coordination']:
            skeleton_score = skeleton_scores.get(attr, 0.0)
            annotation_score = annotation_scores.get(attr) if annotation_scores else None
            if annotation_score is not None:
                fused_score = skeleton_score * w_skeleton + annotation_score * w_annotation
            else:
                fused_score = skeleton_score * S_quality
            fused_scores[attr] = float(max(0.0, min(5.0, fused_score)))
        return fused_scores
    
    def _noodle_rope_skeleton_auxiliary(self, frame_angles: Dict[str, Any],
                                        coordination_skeleton: float) -> Dict[str, float]:
        """面条绳骨架辅助分：thickness/elasticity 关联双手腕（有则 3.0），integrity 关联协调性。"""
        thick_elastic = 3.0
        if 'bilateral' in frame_angles and frame_angles['bilateral']:
            thick_elastic = 3.0
        return {
            'thickness': thick_elastic,
            'elasticity': thick_elastic,
            'integrity': coordination_skeleton,
            'gloss': 2.5
        }

    def _calculate_noodle_rope_scores(self, annotation_scores: Optional[Dict[str, float]],
                                      frame_angles: Dict[str, Any],
                                      coordination_skeleton: float = 3.0) -> Dict[str, float]:
        """面条绳评分：有标注时按比例融合骨架；无标注时 thickness/elasticity/integrity=0.9*骨架+0.5，gloss=2.5"""
        aux = self._noodle_rope_skeleton_auxiliary(frame_angles, coordination_skeleton)
        scores = {}
        if annotation_scores:
            if 'thickness' in annotation_scores:
                scores['thickness'] = 0.8 * annotation_scores['thickness'] + 0.2 * aux['thickness']
            if 'elasticity' in annotation_scores:
                scores['elasticity'] = 0.7 * annotation_scores['elasticity'] + 0.3 * aux['elasticity']
            if 'integrity' in annotation_scores:
                scores['integrity'] = 0.75 * annotation_scores['integrity'] + 0.25 * aux['integrity']
            if 'gloss' in annotation_scores:
                scores['gloss'] = float(annotation_scores['gloss'])
            for k in scores:
                scores[k] = float(max(0.0, min(5.0, scores[k])))
        else:
            scores['thickness'] = float(max(0.0, min(5.0, 0.9 * aux['thickness'] + 0.5)))
            scores['elasticity'] = float(max(0.0, min(5.0, 0.9 * aux['elasticity'] + 0.5)))
            scores['integrity'] = float(max(0.0, min(5.0, 0.9 * aux['integrity'] + 0.5)))
            scores['gloss'] = 2.5
        return scores

    def _calculate_noodle_bundle_scores(self, annotation_scores: Optional[Dict[str, float]],
                                        rope_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """面条束评分：有标注用标注；无标注若有面条绳则 bundle = 面条绳总分×0.8，否则返回空（由调用方决定是否用 rope 推断）"""
        scores = {}
        if annotation_scores:
            for attr in ['tightness', 'uniformity']:
                if attr in annotation_scores:
                    scores[attr] = float(annotation_scores[attr])
            return scores
        if rope_scores:
            rope_total = sum(rope_scores.get(a, 0) * self.weights['noodle_rope_attributes'].get(a, 0)
                            for a in self.weights['noodle_rope_attributes'])
            rope_total = rope_total / sum(self.weights['noodle_rope_attributes'].values()) if self.weights['noodle_rope_attributes'] else 0
            bundle_val = rope_total * 0.8
            scores['tightness'] = scores['uniformity'] = float(max(0.0, min(5.0, bundle_val)))
        return scores
    
    
    def score_video(self, video_name: str) -> Dict[str, Any]:
        """对整个视频进行评分（含标注置信度加载）"""
        annotation_scores, annotation_confidences = self._load_annotation_scores(video_name)
        return self.calculate_comprehensive_score(
            video_name, annotation_scores, annotation_confidences
        )

    def _load_annotation_scores(self, video_name: str) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[int, float]]:
        """
        加载关键帧标注评分及置信度（1-5 转为 0~1）
        Returns:
            (annotation_scores, annotation_confidences)
        """
        annotation_scores = {}
        annotation_confidences: Dict[int, float] = {}
        video_scores_dir = self.scores_dir / video_name
        
        if not video_scores_dir.exists():
            return annotation_scores, annotation_confidences
        
        # 遍历所有评分文件
        for score_file in video_scores_dir.glob("*_scores.json"):
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                frame_name = data.get('frame', '')
                # 从帧名提取帧索引（如 cm1_00013.jpg -> 13）
                if '_' in frame_name:
                    try:
                        frame_index = int(frame_name.split('_')[1].split('.')[0])
                    except (ValueError, IndexError):
                        continue
                else:
                    continue
                
                scores = data.get('scores', {})
                frame_scores = {
                    'hand': [],
                    'noodle_rope': [],
                    'noodle_bundle': []
                }
                
                # 根据检测框的属性判断类别
                for detection_id, detection_scores in scores.items():
                    # 判断类别
                    if 'position' in detection_scores or 'action' in detection_scores:
                        # hand类别
                        hand_scores = {}
                        for attr in ['position', 'action', 'angle', 'coordination']:
                            if attr in detection_scores:
                                hand_scores[attr] = float(detection_scores[attr])
                        if hand_scores:
                            frame_scores['hand'].append(hand_scores)
                    
                    elif 'thickness' in detection_scores or 'elasticity' in detection_scores:
                        # noodle_rope类别
                        rope_scores = {}
                        for attr in ['thickness', 'elasticity', 'gloss', 'integrity']:
                            if attr in detection_scores:
                                rope_scores[attr] = float(detection_scores[attr])
                        if rope_scores:
                            frame_scores['noodle_rope'].append(rope_scores)
                    
                    elif 'tightness' in detection_scores or 'uniformity' in detection_scores:
                        # noodle_bundle类别
                        bundle_scores = {}
                        for attr in ['tightness', 'uniformity']:
                            if attr in detection_scores:
                                bundle_scores[attr] = float(detection_scores[attr])
                        if bundle_scores:
                            frame_scores['noodle_bundle'].append(bundle_scores)
                
                # 对每个类别取平均（如果有多个检测框）
                final_frame_scores = {}
                for class_name, class_scores_list in frame_scores.items():
                    if class_scores_list:
                        # 计算平均值
                        avg_scores = {}
                        for attr in ['position', 'action', 'angle', 'coordination', 
                                    'thickness', 'elasticity', 'gloss', 'integrity',
                                    'tightness', 'uniformity']:
                            values = [s.get(attr) for s in class_scores_list if attr in s]
                            if values:
                                avg_scores[attr] = float(np.mean(values))
                        
                        if avg_scores:
                            final_frame_scores[class_name] = avg_scores
                
                if final_frame_scores:
                    annotation_scores[frame_index] = final_frame_scores
                    # 标注置信度 1-5 转为 0~1，缺省 4/5=0.8
                    annotation_confidences[frame_index] = float(max(0.0, min(1.0, data.get('confidence', 4) / 5.0)))
            
            except Exception as e:
                continue
        
        return annotation_scores, annotation_confidences
    
    def _get_annotation_score_for_frame(self, frame_index: int, 
                                       annotation_scores: Dict[int, Dict[str, Dict[str, float]]],
                                       class_name: str, attribute: str) -> Optional[float]:
        """获取指定帧的标注评分"""
        if frame_index not in annotation_scores:
            return None
        
        frame_scores = annotation_scores[frame_index]
        if class_name not in frame_scores:
            return None
        
        return frame_scores[class_name].get(attribute)
    
    def _interpolate_annotation_score(self, frame_index: int,
                                     annotation_scores: Dict[int, Dict[str, Dict[str, float]]],
                                     class_name: str, attribute: str,
                                     keyframe_gap_penalty_frames: Optional[int] = None) -> Optional[float]:
        """对非关键帧进行标注评分插值；关键帧间隔>keyframe_gap_penalty_frames 时插值分×0.8"""
        if not annotation_scores:
            return None
        
        key_frames = sorted(annotation_scores.keys())
        prev_frame = next_frame = None
        for kf in key_frames:
            if kf <= frame_index:
                prev_frame = kf
            elif kf > frame_index:
                next_frame = kf
                break
        
        prev_score = self._get_annotation_score_for_frame(
            prev_frame, annotation_scores, class_name, attribute
        ) if prev_frame is not None else None
        next_score = self._get_annotation_score_for_frame(
            next_frame, annotation_scores, class_name, attribute
        ) if next_frame is not None else None
        
        interpolated = None
        gap_penalty = 1.0
        if prev_score is not None and next_score is not None:
            alpha = (frame_index - prev_frame) / (next_frame - prev_frame) if next_frame != prev_frame else 0
            interpolated = prev_score * (1 - alpha) + next_score * alpha
            if keyframe_gap_penalty_frames is not None and (next_frame - prev_frame) > keyframe_gap_penalty_frames:
                gap_penalty = 0.8
        elif prev_score is not None:
            interpolated = prev_score * 0.9
            if keyframe_gap_penalty_frames is not None and prev_frame is not None and (frame_index - prev_frame) > keyframe_gap_penalty_frames:
                gap_penalty = 0.8
        elif next_score is not None:
            interpolated = next_score * 0.9
            if keyframe_gap_penalty_frames is not None and next_frame is not None and (next_frame - frame_index) > keyframe_gap_penalty_frames:
                gap_penalty = 0.8
        
        if interpolated is not None:
            return float(max(0.0, min(5.0, interpolated * gap_penalty)))
        return None


def main():
    """测试主函数"""
    project_root = Path(__file__).parent.parent.parent
    scorer = EnhancedComprehensiveScorer(project_root)
    
    # 测试单个视频
    result = scorer.score_video('cm1')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
