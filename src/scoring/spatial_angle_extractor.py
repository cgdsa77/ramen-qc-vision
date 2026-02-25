"""
空间角度特征提取模块
基于论文《基于深度学习的乒乓球姿态动作评分方法》
从骨骼关节点数据提取空间角度变化值
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import math


class SpatialAngleExtractor:
    """空间角度特征提取器（关键点置信度≥min_confidence 才参与计算）"""
    
    def __init__(self, min_confidence: float = 0.7):
        """
        初始化角度提取器
        Args:
            min_confidence: 关键点置信度阈值，≥此值才参与角度计算，默认 0.7
        """
        self.min_confidence = min_confidence
        # MediaPipe Hand 21个关键点的索引定义
        self.keypoint_indices = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # 定义关键角度（三点构成的角度）
        self.angle_definitions = [
            # 拇指角度
            ('thumb_angle', [1, 2, 3]),  # CMC-MCP-IP
            ('thumb_tip_angle', [2, 3, 4]),  # MCP-IP-Tip
            
            # 食指角度
            ('index_angle', [5, 6, 7]),  # MCP-PIP-DIP
            ('index_tip_angle', [6, 7, 8]),  # PIP-DIP-Tip
            
            # 中指角度
            ('middle_angle', [9, 10, 11]),  # MCP-PIP-DIP
            ('middle_tip_angle', [10, 11, 12]),  # PIP-DIP-Tip
            
            # 无名指角度
            ('ring_angle', [13, 14, 15]),  # MCP-PIP-DIP
            ('ring_tip_angle', [14, 15, 16]),  # PIP-DIP-Tip
            
            # 小指角度
            ('pinky_angle', [17, 18, 19]),  # MCP-PIP-DIP
            ('pinky_tip_angle', [18, 19, 20]),  # PIP-DIP-Tip
            
            # 手掌关键角度
            ('palm_base_angle', [0, 5, 17]),  # 手腕-食指根部-小指根部
            ('palm_direction_angle', [5, 0, 17]),  # 食指根部-手腕-小指根部
            
            # 手指间角度
            ('finger_spread_index_middle', [5, 0, 9]),  # 食指-手腕-中指
            ('finger_spread_middle_ring', [9, 0, 13]),  # 中指-手腕-无名指
            ('finger_spread_ring_pinky', [13, 0, 17]),  # 无名指-手腕-小指
        ]
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """
        计算三点之间的角度（p2为顶点）
        
        Args:
            p1: 第一个点 {'x': float, 'y': float, 'z': float, 'confidence': float}
            p2: 顶点 {'x': float, 'y': float, 'z': float, 'confidence': float}
            p3: 第三个点 {'x': float, 'y': float, 'z': float, 'confidence': float}
        
        Returns:
            角度值（度），如果点不存在或置信度不足返回NaN
        """
        # 检查置信度（≥min_confidence 才参与计算）
        if (p1.get('confidence', 0) < self.min_confidence or
            p2.get('confidence', 0) < self.min_confidence or
            p3.get('confidence', 0) < self.min_confidence):
            return float('nan')
        
        # 计算向量
        v1 = np.array([
            p1.get('x', 0) - p2.get('x', 0),
            p1.get('y', 0) - p2.get('y', 0),
            p1.get('z', 0) - p2.get('z', 0)
        ])
        v2 = np.array([
            p3.get('x', 0) - p2.get('x', 0),
            p3.get('y', 0) - p2.get('y', 0),
            p3.get('z', 0) - p2.get('z', 0)
        ])
        
        # 计算角度
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return float('nan')
        
        cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return float(angle)
    
    def extract_angles_from_hand(self, hand: Dict) -> Dict[str, float]:
        """
        从单只手提取所有空间角度
        
        Args:
            hand: 手部关键点数据 {'keypoints': List[Dict]}
        
        Returns:
            角度字典 {角度名称: 角度值}
        """
        kps = hand.get('keypoints', [])
        if len(kps) < 21:
            return {}
        
        angles = {}
        
        for angle_name, indices in self.angle_definitions:
            if len(indices) != 3:
                continue
            
            idx1, idx2, idx3 = indices
            if idx1 < len(kps) and idx2 < len(kps) and idx3 < len(kps):
                p1 = kps[idx1]
                p2 = kps[idx2]
                p3 = kps[idx3]
                
                angle = self.calculate_angle(p1, p2, p3)
                if not math.isnan(angle):
                    angles[angle_name] = angle
        
        return angles
    
    def extract_angles_from_frame(self, hands: List[Dict]) -> Dict[str, Any]:
        """
        从一帧提取所有空间角度
        
        Args:
            hands: 手部关键点数据列表
        
        Returns:
            角度特征字典
        """
        frame_angles = {}
        
        for hand_idx, hand in enumerate(hands):
            hand_angles = self.extract_angles_from_hand(hand)
            if hand_angles:
                frame_angles[f'hand_{hand_idx}'] = hand_angles
        
        # 如果是双手，计算双手间的角度关系
        if len(hands) >= 2:
            bilateral_angles = self._extract_bilateral_angles(hands[0], hands[1])
            frame_angles['bilateral'] = bilateral_angles
        
        return frame_angles
    
    def _extract_bilateral_angles(self, hand0: Dict, hand1: Dict) -> Dict[str, float]:
        """提取双手间的角度关系"""
        kps0 = hand0.get('keypoints', [])
        kps1 = hand1.get('keypoints', [])
        
        if len(kps0) < 21 or len(kps1) < 21:
            return {}
        
        bilateral_angles = {}
        
        # 双手手腕连线角度
        wrist0 = kps0[0]
        wrist1 = kps1[0]
        
        if wrist0.get('confidence', 0) >= self.min_confidence and wrist1.get('confidence', 0) >= self.min_confidence:
            dx = wrist1.get('x', 0) - wrist0.get('x', 0)
            dy = wrist1.get('y', 0) - wrist0.get('y', 0)
            angle = math.atan2(dy, dx) * 180 / math.pi
            bilateral_angles['wrist_connection_angle'] = angle
        
        # 双手手掌方向夹角
        palm_angle0 = self._calculate_palm_direction_angle(kps0)
        palm_angle1 = self._calculate_palm_direction_angle(kps1)
        
        if not math.isnan(palm_angle0) and not math.isnan(palm_angle1):
            angle_diff = abs(palm_angle0 - palm_angle1)
            angle_diff = min(angle_diff, 360 - angle_diff)  # 处理周期性
            bilateral_angles['palm_angle_difference'] = angle_diff
        
        return bilateral_angles
    
    def _calculate_palm_direction_angle(self, kps: List[Dict]) -> float:
        """计算手掌方向角度"""
        if len(kps) < 21:
            return float('nan')
        
        index_base = kps[5]
        pinky_base = kps[17]
        
        if (index_base.get('confidence', 0) < self.min_confidence or
            pinky_base.get('confidence', 0) < self.min_confidence):
            return float('nan')
        
        dx = index_base.get('x', 0) - pinky_base.get('x', 0)
        dy = index_base.get('y', 0) - pinky_base.get('y', 0)
        return math.atan2(dy, dx) * 180 / math.pi
    
    def extract_angle_sequence(self, frames_data: List[Dict]) -> List[Dict[str, float]]:
        """
        从动作序列提取角度序列
        
        Args:
            frames_data: 帧数据列表，每帧包含 'hands' 字段
        
        Returns:
            角度序列列表，每帧对应一个角度字典
        """
        angle_sequence = []
        
        for frame_data in frames_data:
            hands = frame_data.get('hands', [])
            frame_angles = self.extract_angles_from_frame(hands)
            angle_sequence.append(frame_angles)
        
        return angle_sequence
    
    def calculate_angle_changes(self, angle_sequence: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """
        计算角度变化值（相邻帧之间的角度差）
        
        Args:
            angle_sequence: 角度序列
        
        Returns:
            角度变化值字典 {角度名称: [变化值列表]}
        """
        angle_changes = {}
        
        for i in range(1, len(angle_sequence)):
            prev_angles = angle_sequence[i-1]
            curr_angles = angle_sequence[i]
            
            # 遍历所有角度
            all_angle_names = set()
            for hand_key in prev_angles.keys():
                if isinstance(prev_angles[hand_key], dict):
                    all_angle_names.update(prev_angles[hand_key].keys())
            
            for angle_name in all_angle_names:
                # 查找角度值（可能在hand_0, hand_1或bilateral中）
                prev_value = None
                curr_value = None
                
                for key in prev_angles.keys():
                    if isinstance(prev_angles[key], dict) and angle_name in prev_angles[key]:
                        prev_value = prev_angles[key][angle_name]
                        break
                
                for key in curr_angles.keys():
                    if isinstance(curr_angles[key], dict) and angle_name in curr_angles[key]:
                        curr_value = curr_angles[key][angle_name]
                        break
                
                if prev_value is not None and curr_value is not None:
                    if not (math.isnan(prev_value) or math.isnan(curr_value)):
                        change = curr_value - prev_value
                        # 处理角度周期性（-180到180度）
                        if change > 180:
                            change -= 360
                        elif change < -180:
                            change += 360
                        
                        if angle_name not in angle_changes:
                            angle_changes[angle_name] = []
                        angle_changes[angle_name].append(change)
        
        return angle_changes
