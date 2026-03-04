"""
抻面动作自动评分模块
基于评分规则和阈值对检测结果进行自动评分
"""
import json
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class StretchScorer:
    """抻面动作评分器"""
    
    def __init__(self, rules_path: Optional[str] = None, use_image_features: bool = True):
        """
        初始化评分器
        
        Args:
            rules_path: 评分规则文件路径，如果为None则使用默认路径
            use_image_features: 是否使用图像特征提取（True）或仅使用置信度（False）
        """
        if rules_path is None:
            rules_path = project_root / "data" / "scores" / "抻面" / "scoring_rules.json"
        
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()
        self.use_image_features = use_image_features
        
        # 如果使用图像特征，初始化特征提取器
        if self.use_image_features:
            try:
                from src.features.image_feature_extractor import ImageFeatureExtractor
                self.feature_extractor = ImageFeatureExtractor()
            except ImportError:
                print("[警告] 无法导入图像特征提取器，将使用置信度评分")
                self.use_image_features = False
                self.feature_extractor = None
        else:
            self.feature_extractor = None
    
    def _load_rules(self) -> Dict[str, Any]:
        """加载评分规则"""
        if not self.rules_path.exists():
            raise FileNotFoundError(f"评分规则文件不存在: {self.rules_path}")
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        return rules
    
    def _calibrate_features(self, features: Dict[str, float], class_name: str) -> Dict[str, float]:
        """
        校准特征值：将特征提取的原始值映射到与手动评分数据相同的范围（1-5分）
        
        改进策略：
        1. 使用评分规则中的统计信息（mean, poor阈值）进行映射
        2. 将原始值的范围（假设0-5）映射到手动评分数据的实际范围（poor阈值到mean）
        3. 低值（< poor阈值）映射到poor阈值附近，高值映射到mean附近
        
        Args:
            features: 原始特征值字典
            class_name: 类别名称
            
        Returns:
            校准后的特征值字典
        """
        calibrated = {}
        
        # 定义各类别对应的属性
        class_attributes = {
            'noodle_rope': ['thickness', 'elasticity', 'gloss', 'integrity'],
            'hand': ['position', 'action', 'angle', 'coordination'],
            'noodle_bundle': ['tightness', 'uniformity']
        }
        
        attributes = class_attributes.get(class_name, [])
        
        for attr in attributes:
            raw_value = features.get(attr, 0.0)
            
            # 使用评分规则中的统计信息进行校准
            if attr in self.rules.get('thresholds', {}):
                threshold_info = self.rules['thresholds'][attr]
                mean_value = threshold_info.get('mean', 3.0)
                poor_threshold = threshold_info.get('poor', 2.0)
                
                # 优化的映射策略：
                # 将特征提取的原始值（假设范围0-5）映射到手动评分范围
                # 目标：让低值（如0-2）映射到poor阈值以上，确保能获得合理评分
                
                if raw_value < 0:
                    calibrated[attr] = poor_threshold  # 最小值映射到poor阈值
                elif raw_value <= 2.0:
                    # 原始值 0-2 映射到 poor阈值 - mean值
                    # 这是最重要的区间，确保低值能获得合理评分（至少2分）
                    ratio = raw_value / 2.0  # 归一化到0-1
                    calibrated[attr] = poor_threshold + (mean_value - poor_threshold) * ratio
                    calibrated[attr] = max(poor_threshold, min(mean_value, calibrated[attr]))
                elif raw_value <= 3.5:
                    # 原始值 2-3.5 映射到 mean值附近
                    ratio = (raw_value - 2.0) / 1.5  # 归一化到0-1
                    calibrated[attr] = mean_value + (4.0 - mean_value) * ratio * 0.5
                    calibrated[attr] = max(mean_value, min(4.5, calibrated[attr]))
                elif raw_value <= 5.0:
                    # 原始值 3.5-5 映射到 4.0 - 5.0（优秀区间）
                    ratio = (raw_value - 3.5) / 1.5  # 归一化到0-1
                    calibrated[attr] = 4.0 + 1.0 * ratio
                    calibrated[attr] = max(4.0, min(5.0, calibrated[attr]))
                else:
                    # 原始值 > 5，截断到5.0
                    calibrated[attr] = 5.0
            else:
                # 如果没有阈值信息，使用简单的线性映射
                if raw_value < 0:
                    calibrated[attr] = 1.0
                elif raw_value < 1.0:
                    calibrated[attr] = 1.0 + 2.0 * raw_value  # 0 -> 1, 1 -> 3
                elif raw_value <= 5.0:
                    calibrated[attr] = raw_value
                else:
                    calibrated[attr] = 5.0
                
                calibrated[attr] = min(5.0, max(1.0, calibrated[attr]))
        
        return calibrated
    
    def score_attribute(self, attribute: str, value: float) -> int:
        """
        根据阈值对单个属性进行评分
        
        Args:
            attribute: 属性名称（如'thickness', 'position'等）
            value: 属性值（1-5分）
            
        Returns:
            评分结果（1-5分）
        """
        if attribute not in self.rules['thresholds']:
            # 如果没有对应的阈值，直接返回原值（四舍五入）
            return int(round(value))
        
        thresholds = self.rules['thresholds'][attribute]
        
        # 根据阈值确定评分等级
        if value >= thresholds['excellent']:
            return 5
        elif value >= thresholds['good']:
            return 4
        elif value >= thresholds['fair']:
            return 3
        elif value >= thresholds['poor']:
            return 2
        else:
            return 1
    
    def score_detection(self, detection: Dict[str, Any], class_name: str, 
                       image: Optional[Any] = None) -> Dict[str, Any]:
        """
        对单个检测框进行评分
        
        Args:
            detection: 检测结果字典，包含class, conf, xyxy等信息
            class_name: 类别名称（'hand', 'noodle_rope', 'noodle_bundle'）
            image: 原始图像（numpy数组，BGR格式），如果为None则使用置信度评分
            
        Returns:
            评分结果字典
        """
        scores = {}
        
        # 如果使用图像特征且提供了图像
        if self.use_image_features and image is not None and self.feature_extractor is not None:
            try:
                # 提取图像特征
                features = self.feature_extractor.extract_features(image, detection)
                
                # 校准特征值：将特征提取的原始值映射到与手动评分数据相同的范围
                # 使用评分规则中的统计信息（mean, std）进行校准
                calibrated_features = self._calibrate_features(features, class_name)
                
                # 根据类别使用不同的特征
                if class_name == 'noodle_rope':
                    # 使用校准后的特征值进行评分
                    scores['thickness'] = self.score_attribute('thickness', calibrated_features.get('thickness', 3.0))
                    scores['elasticity'] = self.score_attribute('elasticity', calibrated_features.get('elasticity', 3.0))
                    scores['gloss'] = self.score_attribute('gloss', calibrated_features.get('gloss', 3.0))
                    scores['integrity'] = self.score_attribute('integrity', calibrated_features.get('integrity', 3.0))
                    
                elif class_name == 'hand':
                    scores['position'] = self.score_attribute('position', calibrated_features.get('position', 3.0))
                    scores['action'] = self.score_attribute('action', calibrated_features.get('action', 3.0))
                    scores['angle'] = self.score_attribute('angle', calibrated_features.get('angle', 3.0))
                    scores['coordination'] = self.score_attribute('coordination', calibrated_features.get('coordination', 3.0))
                    
                elif class_name == 'noodle_bundle':
                    scores['tightness'] = self.score_attribute('tightness', calibrated_features.get('tightness', 3.0))
                    scores['uniformity'] = self.score_attribute('uniformity', calibrated_features.get('uniformity', 3.0))
            except Exception as e:
                print(f"[警告] 特征提取失败，回退到置信度评分: {e}")
                # 回退到置信度评分
                self._score_by_confidence(detection, class_name, scores)
        else:
            # 使用置信度评分（回退方案）
            self._score_by_confidence(detection, class_name, scores)
        
        return scores
    
    def _score_by_confidence(self, detection: Dict[str, Any], class_name: str, scores: Dict[str, float]):
        """基于置信度的评分（回退方案）"""
        conf = detection.get('conf', 0.5)
        base_score = min(5, max(1, conf * 5))
        
        if class_name == 'noodle_rope':
            scores['thickness'] = self.score_attribute('thickness', base_score)
            scores['elasticity'] = self.score_attribute('elasticity', base_score)
            scores['gloss'] = self.score_attribute('gloss', base_score)
            scores['integrity'] = self.score_attribute('integrity', base_score)
        elif class_name == 'hand':
            scores['position'] = self.score_attribute('position', base_score)
            scores['action'] = self.score_attribute('action', base_score)
            scores['angle'] = self.score_attribute('angle', base_score)
            scores['coordination'] = self.score_attribute('coordination', base_score)
        elif class_name == 'noodle_bundle':
            scores['tightness'] = self.score_attribute('tightness', base_score)
            scores['uniformity'] = self.score_attribute('uniformity', base_score)
    
    def calculate_weighted_score(self, scores: Dict[str, float], class_name: str) -> float:
        """
        计算加权总分
        
        Args:
            scores: 各属性评分字典
            class_name: 类别名称
            
        Returns:
            加权总分（0-5分）
        """
        if class_name not in self.rules['weights']:
            # 如果没有权重配置，返回平均值
            if scores:
                base = sum(scores.values()) / len(scores)
            else:
                base = 0.0
        else:
            weights = self.rules['weights'][class_name]
            total_score = 0.0
            total_weight = 0.0
            
            for attr, score in scores.items():
                if attr in weights:
                    total_score += score * weights[attr]
                    total_weight += weights[attr]
            
            base = total_score / total_weight if total_weight > 0 else 0.0
        
        # 动态偏置校准（方案 4.1）：仅对中间区间分数做修正，避免极端值失真
        # 仅当 1.2 < base < 4.8 时施加偏置，且越接近两端偏置越小
        if class_name in ('hand', 'noodle_rope') and 1.2 < base < 4.8:
            bias = 0.7 if class_name == 'hand' else 0.8
            center = 3.0
            distance = abs(base - center) / 2.0  # 归一化距离
            dynamic_bias = bias * max(0.0, 1.0 - distance / 3.0)
            base += dynamic_bias

        # 保证在 1-5 范围内
        base = max(1.0, min(5.0, base))
        return base
    
    def score_frame(self, detections: List[Dict[str, Any]], image: Optional[Any] = None) -> Dict[str, Any]:
        """
        对单帧的所有检测结果进行评分
        
        Args:
            detections: 检测结果列表
            image: 原始图像（numpy数组，BGR格式），如果为None则使用置信度评分
            
        Returns:
            评分结果字典
        """
        frame_scores = {
            'detections': [],
            'class_scores': {},
            'overall_score': 0.0
        }
        
        class_scores = {}
        overall_weights = self.rules.get('overall_weights', {})
        min_conf = float(self.rules.get('min_confidence', 0.3))
        for det in detections:
            class_name = det.get('class', '')
            if not class_name:
                continue
            conf = float(det.get('conf', 0))
            if conf < min_conf:
                continue

            # 对单个检测框评分（传入图像以提取特征）
            scores = self.score_detection(det, class_name, image)
            weighted_score = self.calculate_weighted_score(scores, class_name)
            
            frame_scores['detections'].append({
                'class': class_name,
                'scores': scores,
                'weighted_score': weighted_score
            })
            
            # 累计类别分数
            if class_name not in class_scores:
                class_scores[class_name] = []
            class_scores[class_name].append(weighted_score)
        
        # 计算各类别平均分
        for class_name, scores_list in class_scores.items():
            if scores_list:
                frame_scores['class_scores'][class_name] = sum(scores_list) / len(scores_list)
        
        # 计算总体分数（加权平均）
        total_score = 0.0
        total_weight = 0.0
        
        for class_name, avg_score in frame_scores['class_scores'].items():
            if class_name in overall_weights:
                total_score += avg_score * overall_weights[class_name]
                total_weight += overall_weights[class_name]
        
        if total_weight > 0:
            frame_scores['overall_score'] = total_score / total_weight
        
        return frame_scores
    
    def score_video(self, video_detections: List[Dict[str, Any]], 
                   video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        对整个视频进行评分
        
        Args:
            video_detections: 视频检测结果列表，每个元素是一帧的检测结果
            video_path: 视频文件路径，如果提供且使用图像特征，将读取视频帧
            
        Returns:
            视频评分结果
        """
        frame_scores_list = []
        
        # 如果需要使用图像特征，尝试读取视频
        video_cap = None
        if self.use_image_features and video_path and Path(video_path).exists():
            try:
                video_cap = cv2.VideoCapture(video_path)
                if not video_cap.isOpened():
                    print(f"[警告] 无法打开视频文件: {video_path}，将使用置信度评分")
                    video_cap = None
            except Exception as e:
                print(f"[警告] 读取视频失败: {e}，将使用置信度评分")
                video_cap = None
        
        for frame_data in video_detections:
            detections = frame_data.get('detections', [])
            frame_index = frame_data.get('frame_index', 0)
            
            # 如果使用图像特征，尝试读取对应帧
            frame_image = None
            if video_cap is not None:
                try:
                    # 设置到对应帧
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame_image = video_cap.read()
                    if not ret:
                        frame_image = None
                except Exception as e:
                    print(f"[警告] 读取第{frame_index}帧失败: {e}")
                    frame_image = None
            
            # 对帧进行评分
            frame_score = self.score_frame(detections, frame_image)
            frame_scores_list.append({
                'frame_index': frame_index,
                **frame_score
            })
        
        # 关闭视频
        if video_cap is not None:
            video_cap.release()
        
        # 计算视频总体评分
        if frame_scores_list:
            overall_scores = [fs['overall_score'] for fs in frame_scores_list if fs['overall_score'] > 0]
            avg_overall_fallback = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            # 计算各类别平均分（仅对出现过的类别有值）
            stretch_classes = ['noodle_rope', 'hand', 'noodle_bundle']
            class_avg_scores = {}
            for class_name in stretch_classes:
                class_scores = []
                for fs in frame_scores_list:
                    if class_name in fs['class_scores']:
                        class_scores.append(fs['class_scores'][class_name])
                if class_scores:
                    class_avg_scores[class_name] = sum(class_scores) / len(class_scores)
                else:
                    class_avg_scores[class_name] = 0.0
            
            total_frames = len(frame_scores_list)
            min_frame_ratio = float(self.rules.get('min_frame_ratio', 0.10))
            present_classes = []
            for c in stretch_classes:
                appear_count = sum(1 for fs in frame_scores_list if fs.get('class_scores', {}).get(c, 0) > 0)
                if appear_count >= max(1, total_frames * min_frame_ratio):
                    present_classes.append(c)
                else:
                    class_avg_scores[c] = 0.0  # 未达比例视为不参与

            overall_weights = self.rules.get('overall_weights', {})
            total_weight = sum(overall_weights.get(c, 0) for c in present_classes)
            if total_weight > 0:
                total_score_sum = sum(class_avg_scores[c] * overall_weights.get(c, 0) for c in present_classes)
                avg_overall = round(total_score_sum / total_weight, 2)
            else:
                avg_overall = avg_overall_fallback

            # 归一化明细，便于追溯（方案 4.3）
            normalized_weights = {}
            if total_weight > 0:
                for c in present_classes:
                    normalized_weights[c] = round(overall_weights.get(c, 0) / total_weight, 4)

            scored_frames = len(overall_scores)
            warning = None
            if total_frames > 0 and scored_frames < total_frames * 0.5:
                warning = "有效评分帧数不足总帧数50%，建议检查视频或检测质量"

            return {
                'total_frames': total_frames,
                'scored_frames': scored_frames,
                'average_overall_score': avg_overall,
                'class_average_scores': class_avg_scores,
                'frame_scores': frame_scores_list,
                'normalization_detail': {
                    'all_categories': stretch_classes,
                    'valid_categories': present_classes,
                    'original_overall_weights': overall_weights,
                    'normalized_weights': normalized_weights,
                    'category_avg_scores': {k: round(v, 2) for k, v in class_avg_scores.items()},
                },
                'warning': warning,
            }
        
        return {
            'total_frames': 0,
            'scored_frames': 0,
            'average_overall_score': 0.0,
            'class_average_scores': {},
            'frame_scores': []
        }

