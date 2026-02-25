"""
关键关节点权重估计器
基于Mean Shift算法确定关键关节点的权重系数
基于论文《基于深度学习的乒乓球姿态动作评分方法》
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math


class KeypointWeightEstimator:
    """关键关节点权重估计器"""
    
    def __init__(self, bandwidth: float = 0.5):
        """
        初始化权重估计器
        
        Args:
            bandwidth: Mean Shift算法的带宽参数
        """
        self.bandwidth = bandwidth
    
    def mean_shift_clustering(self, data: np.ndarray, bandwidth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean Shift聚类算法
        
        Args:
            data: 数据点，形状为 (n_samples, n_features)
            bandwidth: 带宽参数
        
        Returns:
            (聚类中心, 聚类标签)
        """
        n_samples, n_features = data.shape
        
        if n_samples == 0:
            return np.array([]), np.array([])
        
        # 初始化：每个点作为一个聚类中心
        centers = data.copy()
        labels = np.arange(n_samples)
        
        # Mean Shift迭代
        max_iter = 100
        convergence_threshold = 1e-3
        
        for iteration in range(max_iter):
            new_centers = np.zeros_like(centers)
            
            for i in range(n_samples):
                # 计算当前点周围的数据点
                distances = np.linalg.norm(data - centers[i], axis=1)
                weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
                
                # 计算加权平均（Mean Shift）
                if np.sum(weights) > 0:
                    new_centers[i] = np.average(data, axis=0, weights=weights)
                else:
                    new_centers[i] = centers[i]
            
            # 检查收敛
            center_shift = np.linalg.norm(new_centers - centers, axis=1)
            if np.max(center_shift) < convergence_threshold:
                break
            
            centers = new_centers
        
        # 合并相近的聚类中心
        merged_centers = []
        merged_labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0
        
        for i in range(n_samples):
            assigned = False
            for j, center in enumerate(merged_centers):
                if np.linalg.norm(centers[i] - center) < bandwidth:
                    merged_labels[i] = j
                    assigned = True
                    break
            
            if not assigned:
                merged_centers.append(centers[i])
                merged_labels[i] = cluster_id
                cluster_id += 1
        
        return np.array(merged_centers), merged_labels
    
    def calculate_keypoint_importance(self, angle_sequences: List[List[Dict[str, float]]]) -> Dict[str, float]:
        """
        计算关键关节点的权重系数
        
        Args:
            angle_sequences: 多个动作序列的角度数据列表
        
        Returns:
            关键点权重字典 {角度名称: 权重值}
        """
        # 收集所有角度名称
        all_angle_names = set()
        for sequence in angle_sequences:
            for frame_angles in sequence:
                for hand_key in frame_angles.keys():
                    if isinstance(frame_angles[hand_key], dict):
                        all_angle_names.update(frame_angles[hand_key].keys())
        
        all_angle_names = sorted(list(all_angle_names))
        
        if len(all_angle_names) == 0:
            return {}
        
        # 提取每个角度的变化特征
        angle_features = defaultdict(list)
        
        for sequence in angle_sequences:
            # 计算角度变化值
            for i in range(1, len(sequence)):
                prev_angles = sequence[i-1]
                curr_angles = sequence[i]
                
                for angle_name in all_angle_names:
                    prev_value = None
                    curr_value = None
                    
                    # 查找角度值
                    for hand_key in prev_angles.keys():
                        if isinstance(prev_angles[hand_key], dict):
                            if angle_name in prev_angles[hand_key]:
                                prev_value = prev_angles[hand_key][angle_name]
                                break
                    
                    for hand_key in curr_angles.keys():
                        if isinstance(curr_angles[hand_key], dict):
                            if angle_name in curr_angles[hand_key]:
                                curr_value = curr_angles[hand_key][angle_name]
                                break
                    
                    if prev_value is not None and curr_value is not None:
                        if not (math.isnan(prev_value) or math.isnan(curr_value)):
                            change = abs(curr_value - prev_value)
                            # 处理角度周期性
                            if change > 180:
                                change = 360 - change
                            angle_features[angle_name].append(change)
        
        # 对每个角度计算特征向量（变化幅度、变化频率、变化稳定性）
        feature_vectors = []
        angle_names_list = []
        
        for angle_name in all_angle_names:
            if angle_name not in angle_features or len(angle_features[angle_name]) == 0:
                continue
            
            changes = np.array(angle_features[angle_name])
            
            # 特征1：平均变化幅度
            mean_change = np.mean(changes)
            
            # 特征2：变化频率（非零变化的比例）
            change_frequency = np.sum(changes > 1.0) / len(changes) if len(changes) > 0 else 0
            
            # 特征3：变化稳定性（标准差）
            change_std = np.std(changes) if len(changes) > 1 else 0
            
            # 特征4：最大变化幅度
            max_change = np.max(changes) if len(changes) > 0 else 0
            
            # 构建特征向量
            feature_vector = np.array([mean_change, change_frequency, change_std, max_change])
            feature_vectors.append(feature_vector)
            angle_names_list.append(angle_name)
        
        if len(feature_vectors) == 0:
            return {}
        
        feature_matrix = np.array(feature_vectors)
        
        # 归一化特征
        feature_matrix_normalized = (feature_matrix - np.mean(feature_matrix, axis=0)) / (
            np.std(feature_matrix, axis=0) + 1e-8
        )
        
        # Mean Shift聚类
        centers, labels = self.mean_shift_clustering(
            feature_matrix_normalized, 
            self.bandwidth
        )
        
        # 计算每个聚类的样本数
        cluster_counts = defaultdict(int)
        for label in labels:
            cluster_counts[label] += 1
        
        # 根据聚类数量确定权重
        # 聚类样本数越多，说明该角度变化模式越常见，权重越高
        weights = {}
        for i, angle_name in enumerate(angle_names_list):
            cluster_id = labels[i]
            cluster_size = cluster_counts[cluster_id]
            
            # 权重 = 聚类大小 / 总样本数
            # 同时考虑变化幅度（变化大的角度更重要）
            importance_score = cluster_size / len(angle_names_list)
            change_magnitude = feature_vectors[i][0]  # 平均变化幅度
            
            # 综合权重
            weight = importance_score * (1.0 + change_magnitude / 180.0)
            weights[angle_name] = float(weight)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight * len(weights) for k, v in weights.items()}
        
        return weights
    
    def estimate_weights_from_standard_videos(self, 
                                             angle_sequences: List[List[Dict[str, float]]]) -> Dict[str, float]:
        """
        从标准视频序列估计权重
        
        Args:
            angle_sequences: 标准视频的角度序列列表
        
        Returns:
            关键点权重字典
        """
        return self.calculate_keypoint_importance(angle_sequences)
