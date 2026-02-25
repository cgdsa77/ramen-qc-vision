"""
改进的动态时间规整（DTW）算法
基于论文《基于深度学习的乒乓球姿态动作评分方法》
用于匹配两个动作序列的关节点数据
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math


class ImprovedDTW:
    """改进的DTW算法实现"""
    
    def __init__(self, window_size: Optional[int] = None, 
                 distance_metric: str = 'euclidean',
                 max_sequence_length: int = 500):
        """
        初始化DTW算法
        
        Args:
            window_size: 窗口大小（Sakoe-Chiba band），None表示自动计算
            distance_metric: 距离度量方法 ('euclidean', 'manhattan', 'cosine')
            max_sequence_length: 最大序列长度，超过此长度会进行采样
        """
        self.window_size = window_size
        self.distance_metric = distance_metric
        self.max_sequence_length = max_sequence_length
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两个向量的距离"""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x - y)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            if norm_x == 0 or norm_y == 0:
                return 1.0
            return 1.0 - (dot_product / (norm_x * norm_y))
        else:
            return np.linalg.norm(x - y)
    
    def _downsample_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """对序列进行下采样"""
        if len(sequence) <= target_length:
            return sequence
        
        indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
        return sequence[indices]
    
    def compute(self, sequence1: np.ndarray, sequence2: np.ndarray,
                weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        计算两个序列的DTW距离
        
        Args:
            sequence1: 第一个序列，形状为 (n_frames, n_features)
            sequence2: 第二个序列，形状为 (m_frames, n_features)
            weights: 特征权重，形状为 (n_features,)，如果为None则所有特征权重相等
        
        Returns:
            (DTW距离, 对齐路径)
        """
        n = len(sequence1)
        m = len(sequence2)
        
        if n == 0 or m == 0:
            return float('inf'), np.array([])
        
        # 如果序列太长，进行下采样
        if n > self.max_sequence_length:
            sequence1 = self._downsample_sequence(sequence1, self.max_sequence_length)
            n = len(sequence1)
        
        if m > self.max_sequence_length:
            sequence2 = self._downsample_sequence(sequence2, self.max_sequence_length)
            m = len(sequence2)
        
        # 初始化权重
        if weights is None:
            weights = np.ones(sequence1.shape[1])
        else:
            weights = np.array(weights)
            if len(weights) != sequence1.shape[1]:
                weights = np.ones(sequence1.shape[1])
        
        # 归一化权重
        weights = weights / (np.sum(weights) + 1e-8)
        
        # 初始化DTW矩阵
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0.0
        
        # 计算窗口范围（Sakoe-Chiba band）
        if self.window_size is not None:
            window = self.window_size
        else:
            # 自动计算窗口大小：序列长度的10%
            window = max(int(max(n, m) * 0.1), 10)
        
        # 填充DTW矩阵
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                # 计算加权距离
                feature_distances = []
                for k in range(sequence1.shape[1]):
                    dist = self._distance(
                        np.array([sequence1[i-1, k]]),
                        np.array([sequence2[j-1, k]])
                    )
                    feature_distances.append(dist * weights[k])
                
                cost = sum(feature_distances)
                
                # 找到最小路径
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # 插入
                    dtw_matrix[i, j-1],      # 删除
                    dtw_matrix[i-1, j-1]     # 匹配
                )
        
        # 回溯路径
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                min_val = min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
                if dtw_matrix[i-1, j-1] == min_val:
                    i -= 1
                    j -= 1
                elif dtw_matrix[i-1, j] == min_val:
                    i -= 1
                else:
                    j -= 1
        
        path.reverse()
        
        return dtw_matrix[n, m], np.array(path)
    
    def compute_angle_sequence_distance(self, angles1: List[Dict[str, float]],
                                       angles2: List[Dict[str, float]],
                                       keypoint_weights: Optional[Dict[str, float]] = None) -> float:
        """
        计算两个角度序列的DTW距离
        
        Args:
            angles1: 第一个角度序列，每帧是一个角度字典
            angles2: 第二个角度序列，每帧是一个角度字典
            keypoint_weights: 关键点权重字典
        
        Returns:
            DTW距离
        """
        # 提取所有角度名称
        all_angle_names = set()
        for frame_angles in angles1 + angles2:
            for hand_key in frame_angles.keys():
                if isinstance(frame_angles[hand_key], dict):
                    all_angle_names.update(frame_angles[hand_key].keys())
        
        all_angle_names = sorted(list(all_angle_names))
        
        if len(all_angle_names) == 0:
            return float('inf')
        
        # 构建特征矩阵
        def build_feature_matrix(angle_sequence):
            features = []
            for frame_angles in angle_sequence:
                frame_features = []
                for angle_name in all_angle_names:
                    value = None
                    # 查找角度值
                    for hand_key in frame_angles.keys():
                        if isinstance(frame_angles[hand_key], dict):
                            if angle_name in frame_angles[hand_key]:
                                value = frame_angles[hand_key][angle_name]
                                break
                    
                    if value is None or math.isnan(value):
                        # 使用前一个有效值或0
                        value = features[-1][len(frame_features)] if features else 0.0
                    
                    frame_features.append(value)
                features.append(frame_features)
            
            return np.array(features)
        
        seq1_matrix = build_feature_matrix(angles1)
        seq2_matrix = build_feature_matrix(angles2)
        
        # 构建权重向量
        weights = None
        if keypoint_weights:
            weights = np.array([
                keypoint_weights.get(angle_name, 1.0) 
                for angle_name in all_angle_names
            ])
        
        # 计算DTW距离
        distance, path = self.compute(seq1_matrix, seq2_matrix, weights)
        
        return distance
    
    def normalize_distance(self, distance: float, sequence_length: int) -> float:
        """
        归一化DTW距离（除以序列长度）
        
        Args:
            distance: DTW距离
            sequence_length: 序列长度
        
        Returns:
            归一化距离
        """
        if sequence_length == 0:
            return float('inf')
        return distance / sequence_length
