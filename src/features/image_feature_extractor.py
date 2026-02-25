"""
图像特征提取模块
从检测框区域提取真实的图像特征，用于评分
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class ImageFeatureExtractor:
    """图像特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        pass
    
    def extract_roi(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        从图像中提取检测框区域（ROI）
        
        Args:
            image: 原始图像 (BGR格式)
            bbox: 检测框坐标 [x1, y1, x2, y2]
            
        Returns:
            ROI图像
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = image.shape[:2]
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        
        roi = image[y1:y2, x1:x2]
        return roi
    
    def extract_noodle_rope_features(self, image: np.ndarray, bbox: List[float]) -> Dict[str, float]:
        """
        提取面条（noodle_rope）的特征
        
        Args:
            image: 原始图像
            bbox: 检测框坐标 [x1, y1, x2, y2]
            
        Returns:
            特征字典，包含 thickness, gloss, integrity, elasticity 的原始值（需要映射到1-5分）
        """
        roi = self.extract_roi(image, bbox)
        if roi.size == 0:
            return {
                'thickness': 0.0,
                'gloss': 0.0,
                'integrity': 0.0,
                'elasticity': 0.0
            }
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        features = {}
        
        # 1. 厚度（thickness）：计算面条的平均宽度
        # 使用边缘检测找到面条边缘，计算平均宽度
        edges = cv2.Canny(gray, 50, 150)
        
        # 计算垂直方向的边缘密度（假设面条大致水平）
        # 简化方法：计算检测框的宽高比，结合边缘信息
        if h > 0 and w > 0:
            aspect_ratio = w / h
            
            # 计算边缘像素占比
            edge_ratio = np.sum(edges > 0) / (h * w) if (h * w) > 0 else 0
            
            # 厚度估算：基于检测框高度和边缘信息
            # 检测框高度越小，面条越细（假设检测框紧贴面条）
            # 归一化到合理范围
            thickness_raw = 1.0 / (aspect_ratio + 0.1) * (1 + edge_ratio * 2)
            features['thickness'] = min(5.0, max(0.0, thickness_raw * 2.5))  # 粗略映射到0-5
        else:
            features['thickness'] = 0.0
        
        # 2. 光泽度（gloss）：基于亮度和对比度
        # 转换为HSV，使用V通道（亮度）
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # 高光泽 = 高平均亮度 + 低亮度方差（均匀明亮）
        brightness_mean = np.mean(v_channel) / 255.0
        brightness_std = np.std(v_channel) / 255.0
        
        # 光泽度：亮度越高越好，方差越小越好（更均匀）
        gloss_raw = brightness_mean * (1.0 - min(1.0, brightness_std))
        features['gloss'] = min(5.0, max(0.0, gloss_raw * 5.0))
        
        # 3. 完整性（integrity）：检测是否有断裂
        # 使用轮廓分析：完整的面条应该有连续的轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 计算最大轮廓的面积占比
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)
            total_area = h * w
            
            # 完整性：最大轮廓面积占比越高，越完整
            integrity_ratio = contour_area / total_area if total_area > 0 else 0
            
            # 如果轮廓数量少且面积占比高，说明完整性好
            contour_count_factor = 1.0 / (1.0 + len(contours) * 0.1)
            integrity_raw = integrity_ratio * contour_count_factor
            features['integrity'] = min(5.0, max(0.0, integrity_raw * 5.0))
        else:
            features['integrity'] = 0.0
        
        # 4. 弹性（elasticity）：需要序列帧分析，这里使用简化代理
        # 基于检测框的稳定性：如果检测框形状稳定，说明弹性好（不易变形）
        # 当前帧无法单独判断，返回基于其他特征的估算值
        # 暂时使用完整性和光泽度的组合作为代理
        elasticity_raw = (features['integrity'] + features['gloss']) / 2.0
        features['elasticity'] = elasticity_raw
        
        return features
    
    def extract_hand_features(self, image: np.ndarray, bbox: List[float], 
                             image_size: Tuple[int, int] = None) -> Dict[str, float]:
        """
        提取手部（hand）的特征
        
        Args:
            image: 原始图像
            bbox: 检测框坐标 [x1, y1, x2, y2]
            image_size: 图像尺寸 (width, height)，用于归一化位置
            
        Returns:
            特征字典，包含 position, action, angle, coordination
        """
        roi = self.extract_roi(image, bbox)
        if roi.size == 0:
            return {
                'position': 0.0,
                'action': 0.0,
                'angle': 0.0,
                'coordination': 0.0
            }
        
        features = {}
        h, w = image.shape[:2]
        
        # 1. 位置（position）：检测框中心相对于图像的位置
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # 归一化到0-1
        norm_x = center_x / w if w > 0 else 0.5
        norm_y = center_y / h if h > 0 else 0.5
        
        # 理想位置：画面中心偏下（操作区域）
        ideal_x, ideal_y = 0.5, 0.6
        position_distance = np.sqrt((norm_x - ideal_x)**2 + (norm_y - ideal_y)**2)
        
        # 位置评分：距离理想位置越近，分数越高
        position_raw = 1.0 - min(1.0, position_distance * 2.0)
        features['position'] = min(5.0, max(0.0, position_raw * 5.0))
        
        # 2. 角度（angle）：计算检测框的主轴方向
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 使用最大轮廓计算最小外接矩形
            max_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(max_contour)
            angle = rect[2]  # 角度（-90到90度）
            
            # 归一化角度到0-1（假设理想角度在-30到30度之间）
            angle_normalized = abs(angle) / 90.0
            angle_score = 1.0 - min(1.0, angle_normalized * 2.0)
            features['angle'] = min(5.0, max(0.0, angle_score * 5.0))
        else:
            # 如果没有轮廓，使用检测框的宽高比估算角度
            roi_h, roi_w = roi.shape[:2]
            if roi_w > 0:
                aspect_ratio = roi_h / roi_w
                # 假设理想的手部姿态是接近正方形的检测框
                angle_deviation = abs(aspect_ratio - 1.0)
                angle_score = 1.0 - min(1.0, angle_deviation)
                features['angle'] = min(5.0, max(0.0, angle_score * 5.0))
            else:
                features['angle'] = 3.0  # 默认中等分数
        
        # 3. 动作（action）：需要序列帧分析，这里使用简化代理
        # 基于检测框的置信度和稳定性（当前帧无法判断，使用默认值）
        # 暂时使用位置和角度的组合作为代理
        action_raw = (features['position'] + features['angle']) / 2.0
        features['action'] = action_raw
        
        # 4. 协调性（coordination）：需要多手检测，这里使用简化代理
        # 当前帧无法单独判断，返回基于其他特征的估算值
        coordination_raw = (features['position'] + features['angle'] + features['action']) / 3.0
        features['coordination'] = coordination_raw
        
        return features
    
    def extract_noodle_bundle_features(self, image: np.ndarray, bbox: List[float]) -> Dict[str, float]:
        """
        提取面束（noodle_bundle）的特征
        
        Args:
            image: 原始图像
            bbox: 检测框坐标 [x1, y1, x2, y2]
            
        Returns:
            特征字典，包含 tightness, uniformity
        """
        roi = self.extract_roi(image, bbox)
        if roi.size == 0:
            return {
                'tightness': 0.0,
                'uniformity': 0.0
            }
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        features = {}
        
        # 1. 紧实度（tightness）：计算面束区域内的密度
        # 使用边缘检测计算边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w) if (h * w) > 0 else 0
        
        # 计算轮廓的紧凑度
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 计算所有轮廓的总面积
            total_contour_area = sum(cv2.contourArea(c) for c in contours)
            total_area = h * w
            
            # 紧实度：边缘密度高 + 轮廓面积占比高 = 紧实
            area_ratio = total_contour_area / total_area if total_area > 0 else 0
            tightness_raw = (edge_density * 0.5 + area_ratio * 0.5)
            features['tightness'] = min(5.0, max(0.0, tightness_raw * 5.0))
        else:
            features['tightness'] = 0.0
        
        # 2. 均匀度（uniformity）：计算面束内各部分的粗细一致性
        # 将ROI分成多个区域，计算每个区域的边缘密度，然后计算方差
        if h > 4 and w > 4:
            # 分成3x3网格
            grid_h, grid_w = h // 3, w // 3
            densities = []
            
            for i in range(3):
                for j in range(3):
                    y1 = i * grid_h
                    y2 = min((i + 1) * grid_h, h)
                    x1 = j * grid_w
                    x2 = min((j + 1) * grid_w, w)
                    
                    grid_roi = gray[y1:y2, x1:x2]
                    if grid_roi.size > 0:
                        grid_edges = cv2.Canny(grid_roi, 50, 150)
                        grid_density = np.sum(grid_edges > 0) / grid_roi.size if grid_roi.size > 0 else 0
                        densities.append(grid_density)
            
            if len(densities) > 1:
                # 均匀度：密度方差越小，越均匀
                density_std = np.std(densities)
                density_mean = np.mean(densities)
                
                # 归一化：标准差相对于均值的比例
                cv_coefficient = density_std / density_mean if density_mean > 0 else 1.0
                uniformity_raw = 1.0 - min(1.0, cv_coefficient)
                features['uniformity'] = min(5.0, max(0.0, uniformity_raw * 5.0))
            else:
                features['uniformity'] = 3.0  # 默认中等分数
        else:
            features['uniformity'] = 3.0
        
        return features
    
    def extract_features(self, image: np.ndarray, detection: Dict[str, Any]) -> Dict[str, float]:
        """
        根据检测结果提取特征
        
        Args:
            image: 原始图像 (BGR格式)
            detection: 检测结果字典，包含 class, xyxy, conf 等
            
        Returns:
            特征字典
        """
        class_name = detection.get('class', '')
        bbox = detection.get('xyxy', [0, 0, 0, 0])
        
        if class_name == 'noodle_rope':
            return self.extract_noodle_rope_features(image, bbox)
        elif class_name == 'hand':
            h, w = image.shape[:2]
            return self.extract_hand_features(image, bbox, image_size=(w, h))
        elif class_name == 'noodle_bundle':
            return self.extract_noodle_bundle_features(image, bbox)
        else:
            # 未知类别，返回空特征
            return {}

