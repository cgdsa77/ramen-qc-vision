"""
抻面综合评分系统
结合关键帧标注评分和骨架线数据，进行综合评分
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math


class ComprehensiveScorer:
    """综合评分器：结合标注评分和骨架线数据"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scores_dir = project_root / "data" / "scores" / "抻面"
        self.hand_keypoints_dir = self.scores_dir / "hand_keypoints"
        
        # 加载评分规则
        self.scoring_rules = self._load_scoring_rules()
        
        # 标准骨架线统计（需要从标准视频提取）
        self.standard_skeleton_stats = self._load_standard_skeleton_stats()
        
        # 权重配置
        self.weights = {
            'class': {
                'hand': 0.4,
                'noodle_rope': 0.5,
                'noodle_bundle': 0.1
            },
            'hand_attributes': {
                'position': 0.25,
                'action': 0.30,
                'angle': 0.20,
                'coordination': 0.25
            },
            'noodle_rope_attributes': {
                'thickness': 0.30,
                'elasticity': 0.25,
                'gloss': 0.20,
                'integrity': 0.25
            },
            'data_source': {
                'position': {'skeleton': 0.4, 'annotation': 0.6},
                'angle': {'skeleton': 0.5, 'annotation': 0.5},
                'action': {'skeleton': 0.6, 'annotation': 0.4},
                'coordination': {'skeleton': 0.5, 'annotation': 0.5},
                'thickness': {'skeleton': 0.3, 'annotation': 0.7},
                'elasticity': {'skeleton': 0.4, 'annotation': 0.6},
                'gloss': {'skeleton': 0.0, 'annotation': 1.0},
                'integrity': {'skeleton': 0.2, 'annotation': 0.8}
            }
        }
    
    def _load_scoring_rules(self) -> Dict:
        """加载评分规则"""
        rules_file = self.scores_dir / "scoring_rules.json"
        if rules_file.exists():
            with open(rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_standard_skeleton_stats(self) -> Dict:
        """加载标准骨架线统计信息"""
        # TODO: 从标准视频提取统计信息
        # 目前使用默认值
        return {
            'hand_distance': {
                'mean': 450.5,
                'std': 85.3,
                'min': 280.0,
                'max': 680.0
            },
            'palm_angle': {
                'mean': 92.5,
                'std': 15.2,
                'min': 60.0,
                'max': 135.0
            },
            'stretch_velocity': {
                'mean': 2.5,
                'std': 1.2,
                'min': 0.5,
                'max': 5.0
            }
        }
    
    def extract_skeleton_features(self, hands: List[Dict], 
                                  prev_hands: Optional[List[Dict]] = None,
                                  frame_interval: float = 1.0/30.0) -> Dict[str, Any]:
        """
        从骨架线数据提取特征
        
        Args:
            hands: 当前帧的手部关键点数据
            prev_hands: 前一帧的手部关键点数据（用于计算时序特征）
            frame_interval: 帧间隔（秒）
        
        Returns:
            特征字典
        """
        features = {}
        
        if not hands or len(hands) == 0:
            return features
        
        # 提取单 hand 特征
        if len(hands) >= 1:
            hand0 = hands[0]
            kps0 = hand0.get('keypoints', [])
            
            if len(kps0) >= 21:
                # 手腕位置
                wrist = kps0[0]
                features['wrist_x'] = wrist.get('x', 0)
                features['wrist_y'] = wrist.get('y', 0)
                features['wrist_confidence'] = wrist.get('confidence', 0)
                
                # 手掌方向
                pinky_base = kps0[17]
                index_base = kps0[5]
                palm_dir_x = index_base.get('x', 0) - pinky_base.get('x', 0)
                palm_dir_y = index_base.get('y', 0) - pinky_base.get('y', 0)
                palm_angle = math.atan2(palm_dir_y, palm_dir_x) * 180 / math.pi
                features['palm_angle'] = palm_angle
                
                # 手指张开度
                finger_distances = []
                for tip_idx in [4, 8, 12, 16, 20]:
                    tip = kps0[tip_idx]
                    dist = math.sqrt(
                        (tip.get('x', 0) - wrist.get('x', 0))**2 +
                        (tip.get('y', 0) - wrist.get('y', 0))**2
                    )
                    finger_distances.append(dist)
                
                if finger_distances:
                    features['finger_spread'] = float(np.std(finger_distances))
                    features['finger_mean_distance'] = float(np.mean(finger_distances))
        
        # 提取双手特征
        if len(hands) >= 2:
            hand0 = hands[0]
            hand1 = hands[1]
            kps0 = hand0.get('keypoints', [])
            kps1 = hand1.get('keypoints', [])
            
            if len(kps0) >= 21 and len(kps1) >= 21:
                wrist0 = kps0[0]
                wrist1 = kps1[0]
                
                # 双手距离
                hand_distance = math.sqrt(
                    (wrist0.get('x', 0) - wrist1.get('x', 0))**2 +
                    (wrist0.get('y', 0) - wrist1.get('y', 0))**2
                )
                features['hand_distance'] = hand_distance
                
                # 双手对称性（Y坐标差）
                symmetry_y_diff = abs(wrist0.get('y', 0) - wrist1.get('y', 0))
                features['symmetry_y_diff'] = symmetry_y_diff
                
                # 双手角度一致性
                palm_angle0 = features.get('palm_angle', 0)
                palm_angle1 = self._calculate_palm_angle(kps1)
                angle_diff = abs(palm_angle0 - palm_angle1)
                angle_consistency = 180 - min(angle_diff, 360 - angle_diff)
                features['angle_consistency'] = angle_consistency
        
        # 提取时序特征
        if prev_hands and len(prev_hands) > 0 and len(hands) > 0:
            prev_wrist = prev_hands[0].get('keypoints', [{}])[0]
            curr_wrist = hands[0].get('keypoints', [{}])[0]
            
            if prev_wrist and curr_wrist:
                velocity_x = (curr_wrist.get('x', 0) - prev_wrist.get('x', 0)) / frame_interval
                velocity_y = (curr_wrist.get('y', 0) - prev_wrist.get('y', 0)) / frame_interval
                velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
                
                features['velocity_x'] = velocity_x
                features['velocity_y'] = velocity_y
                features['velocity_magnitude'] = velocity_magnitude
                
                # 如果有双手，计算拉伸速度
                if len(hands) >= 2 and len(prev_hands) >= 2:
                    prev_distance = self._calculate_hand_distance(prev_hands[0], prev_hands[1])
                    curr_distance = features.get('hand_distance', 0)
                    if prev_distance > 0:
                        stretch_velocity = (curr_distance - prev_distance) / frame_interval
                        features['stretch_velocity'] = stretch_velocity
        
        return features
    
    def _calculate_palm_angle(self, kps: List[Dict]) -> float:
        """计算手掌角度"""
        if len(kps) < 21:
            return 0.0
        pinky_base = kps[17]
        index_base = kps[5]
        palm_dir_x = index_base.get('x', 0) - pinky_base.get('x', 0)
        palm_dir_y = index_base.get('y', 0) - pinky_base.get('y', 0)
        return math.atan2(palm_dir_y, palm_dir_x) * 180 / math.pi
    
    def _calculate_hand_distance(self, hand0: Dict, hand1: Dict) -> float:
        """计算双手距离"""
        kps0 = hand0.get('keypoints', [])
        kps1 = hand1.get('keypoints', [])
        if len(kps0) < 21 or len(kps1) < 21:
            return 0.0
        wrist0 = kps0[0]
        wrist1 = kps1[0]
        return math.sqrt(
            (wrist0.get('x', 0) - wrist1.get('x', 0))**2 +
            (wrist0.get('y', 0) - wrist1.get('y', 0))**2
        )
    
    def calculate_skeleton_score(self, attribute: str, features: Dict, 
                                 image_size: Tuple[int, int] = None) -> float:
        """
        基于骨架线特征计算评分
        
        Args:
            attribute: 评分属性（position, angle, action, coordination等）
            features: 骨架线特征字典
            image_size: 图像尺寸 (width, height)
        
        Returns:
            评分（0-5分）
        """
        if attribute == 'position':
            return self._score_position_skeleton(features, image_size)
        elif attribute == 'angle':
            return self._score_angle_skeleton(features)
        elif attribute == 'action':
            return self._score_action_skeleton(features)
        elif attribute == 'coordination':
            return self._score_coordination_skeleton(features)
        elif attribute == 'thickness':
            return self._score_thickness_skeleton(features)
        elif attribute == 'elasticity':
            return self._score_elasticity_skeleton(features)
        elif attribute == 'integrity':
            return self._score_integrity_skeleton(features)
        else:
            return 0.0
    
    def _score_position_skeleton(self, features: Dict, image_size: Tuple[int, int]) -> float:
        """基于骨架线计算位置评分"""
        if 'wrist_x' not in features or image_size is None:
            return 0.0
        
        width, height = image_size
        wrist_x = features['wrist_x']
        wrist_y = features['wrist_y']
        
        # 归一化位置
        norm_x = wrist_x / width if width > 0 else 0.5
        norm_y = wrist_y / height if height > 0 else 0.5
        
        # 理想位置：画面中心偏下
        ideal_x, ideal_y = 0.5, 0.6
        position_deviation = math.sqrt((norm_x - ideal_x)**2 + (norm_y - ideal_y)**2)
        
        # 最大可能偏差
        max_deviation = math.sqrt(width**2 + height**2) / (2 * max(width, height))
        
        # 评分：偏差越小，分数越高
        score = 5.0 * (1.0 - min(1.0, position_deviation / max_deviation))
        return max(0.0, min(5.0, score))
    
    def _score_angle_skeleton(self, features: Dict) -> float:
        """基于骨架线计算角度评分"""
        if 'palm_angle' not in features:
            return 0.0
        
        palm_angle = features['palm_angle']
        ideal_angle = 90.0  # 标准角度（水平）
        
        # 角度偏差
        angle_deviation = abs(palm_angle - ideal_angle)
        # 处理周期性（0-360度）
        angle_deviation = min(angle_deviation, 360 - angle_deviation)
        
        # 评分：偏差越小，分数越高
        score = 5.0 * (1.0 - min(1.0, angle_deviation / 90.0))
        return max(0.0, min(5.0, score))
    
    def _score_action_skeleton(self, features: Dict) -> float:
        """基于骨架线计算动作评分"""
        scores = []
        
        # 手指张开度评分
        if 'finger_spread' in features:
            finger_spread = features['finger_spread']
            max_spread = 50.0  # 假设最大张开度
            spread_score = 5.0 * (1.0 - min(1.0, finger_spread / max_spread))
            scores.append(spread_score * 0.3)
        
        # 速度评分
        if 'velocity_magnitude' in features:
            velocity = features['velocity_magnitude']
            ideal_velocity = 100.0  # 像素/秒（需要根据标准数据集调整）
            velocity_score = 5.0 * min(1.0, velocity / ideal_velocity)
            scores.append(velocity_score * 0.4)
        
        # 稳定性评分（如果有速度数据）
        if 'velocity_magnitude' in features:
            # 简化：假设速度适中表示稳定
            velocity = features['velocity_magnitude']
            ideal_velocity = 100.0
            stability = 1.0 / (1.0 + abs(velocity - ideal_velocity) / ideal_velocity)
            stability_score = 5.0 * stability
            scores.append(stability_score * 0.3)
        
        if scores:
            return max(0.0, min(5.0, sum(scores)))
        return 0.0
    
    def _score_coordination_skeleton(self, features: Dict) -> float:
        """基于骨架线计算协调性评分"""
        if 'hand_distance' not in features:
            return 0.0
        
        scores = []
        
        # 双手距离评分
        if 'hand_distance' in features:
            hand_distance = features['hand_distance']
            ideal_distance = self.standard_skeleton_stats['hand_distance']['mean']
            distance_deviation = abs(hand_distance - ideal_distance) / ideal_distance
            distance_score = 5.0 * (1.0 - min(1.0, distance_deviation))
            scores.append(distance_score * 0.3)
        
        # 对称性评分
        if 'symmetry_y_diff' in features:
            symmetry_diff = features['symmetry_y_diff']
            max_diff = 100.0  # 像素（需要根据标准数据集调整）
            symmetry_score = 5.0 * (1.0 - min(1.0, symmetry_diff / max_diff))
            scores.append(symmetry_score * 0.3)
        
        # 角度一致性评分
        if 'angle_consistency' in features:
            angle_consistency = features['angle_consistency']
            consistency_score = 5.0 * (angle_consistency / 180.0)
            scores.append(consistency_score * 0.4)
        
        if scores:
            return max(0.0, min(5.0, sum(scores)))
        return 0.0
    
    def _score_thickness_skeleton(self, features: Dict) -> float:
        """基于骨架线计算粗细评分（间接推断）"""
        if 'hand_distance' not in features:
            return 0.0
        
        hand_distance = features['hand_distance']
        ideal_distance = self.standard_skeleton_stats['hand_distance']['mean']
        
        # 假设：距离越大，拉伸越充分，面条越细
        stretch_ratio = hand_distance / ideal_distance if ideal_distance > 0 else 1.0
        
        # 根据拉伸比例推断粗细（需要根据实际数据调整）
        # 简化：假设拉伸比例在0.8-1.2之间为理想范围
        if 0.8 <= stretch_ratio <= 1.2:
            return 5.0
        elif 0.6 <= stretch_ratio < 0.8 or 1.2 < stretch_ratio <= 1.4:
            return 4.0
        elif 0.4 <= stretch_ratio < 0.6 or 1.4 < stretch_ratio <= 1.6:
            return 3.0
        else:
            return 2.0
    
    def _score_elasticity_skeleton(self, features: Dict) -> float:
        """基于骨架线计算弹性评分"""
        if 'stretch_velocity' not in features:
            return 0.0
        
        stretch_velocity = features['stretch_velocity']
        ideal_velocity = self.standard_skeleton_stats['stretch_velocity']['mean']
        
        # 速度接近理想值，弹性越好
        velocity_deviation = abs(stretch_velocity - ideal_velocity) / ideal_velocity if ideal_velocity > 0 else 1.0
        score = 5.0 * (1.0 - min(1.0, velocity_deviation))
        return max(0.0, min(5.0, score))
    
    def _score_integrity_skeleton(self, features: Dict) -> float:
        """基于骨架线计算完整程度评分（间接推断）"""
        # 简化：如果动作连贯，假设面条完整
        if 'velocity_magnitude' in features:
            velocity = features['velocity_magnitude']
            # 速度适中表示动作连贯
            if 50.0 <= velocity <= 150.0:
                return 5.0
            else:
                return 3.0
        return 0.0
    
    def load_annotation_score(self, video_name: str, frame_name: str, 
                             detection_id: str, attribute: str) -> Optional[float]:
        """加载关键帧标注评分"""
        score_file = self.scores_dir / video_name / f"{frame_name}_scores.json"
        if not score_file.exists():
            return None
        
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                scores = data.get('scores', {})
                detection_scores = scores.get(detection_id, {})
                return detection_scores.get(attribute)
        except Exception:
            return None
    
    def calculate_comprehensive_score(self, video_name: str, frame_index: int,
                                     hands: List[Dict],
                                     prev_hands: Optional[List[Dict]] = None,
                                     image_size: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        计算综合评分
        
        Args:
            video_name: 视频名称
            frame_index: 帧索引
            hands: 当前帧的手部关键点数据
            prev_hands: 前一帧的手部关键点数据
            image_size: 图像尺寸 (width, height)
        
        Returns:
            综合评分结果字典
        """
        # 提取骨架线特征
        features = self.extract_skeleton_features(hands, prev_hands)
        
        # 构建帧名称（用于查找标注评分）
        frame_name = f"{video_name}_{frame_index:05d}.jpg"
        
        # 计算各类别评分
        scores = {
            'hand': {},
            'noodle_rope': {},
            'noodle_bundle': {}
        }
        
        # 手部评分
        hand_attributes = ['position', 'angle', 'action', 'coordination']
        for attr in hand_attributes:
            # 骨架线评分
            skeleton_score = self.calculate_skeleton_score(attr, features, image_size)
            
            # 标注评分（如果有）
            annotation_score = None
            # TODO: 需要根据检测框ID查找对应的标注评分
            # 目前简化处理
            
            # 综合评分
            weights = self.weights['data_source'].get(attr, {'skeleton': 0.5, 'annotation': 0.5})
            if annotation_score is not None:
                comprehensive_score = (skeleton_score * weights['skeleton'] + 
                                     annotation_score * weights['annotation'])
            else:
                # 只有骨架线评分，降低权重
                comprehensive_score = skeleton_score * 0.8
            
            scores['hand'][attr] = {
                'skeleton': skeleton_score,
                'annotation': annotation_score,
                'comprehensive': comprehensive_score
            }
        
        # 面条评分（需要检测框数据，这里简化处理）
        # TODO: 需要结合检测框数据进行评分
        
        # 计算类别平均分
        hand_score = sum(scores['hand'][attr]['comprehensive'] * 
                        self.weights['hand_attributes'][attr]
                        for attr in hand_attributes)
        
        # 计算总分
        total_score = hand_score * self.weights['class']['hand']
        
        return {
            'frame_index': frame_index,
            'scores': scores,
            'hand_score': hand_score,
            'total_score': total_score,
            'features': features
        }
    
    def score_video(self, video_name: str) -> Dict[str, Any]:
        """
        对整个视频进行评分
        
        Args:
            video_name: 视频名称
        
        Returns:
            视频评分结果
        """
        # 加载骨架线数据
        keypoints_file = self.hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            return {'error': f'Keypoints file not found: {keypoints_file}'}
        
        with open(keypoints_file, 'r', encoding='utf-8') as f:
            keypoints_data = json.load(f)
        
        frames_data = keypoints_data.get('frames', [])
        width = keypoints_data.get('width', 1280)
        height = keypoints_data.get('height', 720)
        image_size = (width, height)
        
        # 逐帧评分
        frame_scores = []
        prev_hands = None
        
        for frame_data in frames_data:
            frame_index = frame_data.get('frame_index', 0)
            hands = frame_data.get('hands', [])
            
            # 计算综合评分
            score_result = self.calculate_comprehensive_score(
                video_name, frame_index, hands, prev_hands, image_size
            )
            frame_scores.append(score_result)
            
            prev_hands = hands
        
        # 计算统计信息
        total_scores = [s['total_score'] for s in frame_scores if s['total_score'] > 0]
        hand_scores = [s['hand_score'] for s in frame_scores if s['hand_score'] > 0]
        
        return {
            'video': video_name,
            'total_frames': len(frames_data),
            'scored_frames': len(total_scores),
            'average_total_score': float(np.mean(total_scores)) if total_scores else 0.0,
            'average_hand_score': float(np.mean(hand_scores)) if hand_scores else 0.0,
            'frame_scores': frame_scores[:100]  # 只返回前100帧（避免数据过大）
        }


def main():
    """测试主函数"""
    project_root = Path(__file__).parent.parent.parent
    scorer = ComprehensiveScorer(project_root)
    
    # 测试单个视频
    result = scorer.score_video('cm1')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
