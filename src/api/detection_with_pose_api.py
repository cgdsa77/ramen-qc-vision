"""
检测和姿态估计API（可选功能）
在检测基础上添加姿态估计功能，但不影响检测结果
"""
from typing import Dict, Any, List
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.models.pose import PoseEstimator


class DetectionWithPoseAPI:
    """
    带姿态估计的检测API
    可以在检测结果基础上添加姿态估计，但两者独立运行
    """
    
    def __init__(self, detector, pose_estimator=None, cfg=None):
        """
        Args:
            detector: 检测器实例
            pose_estimator: 姿态估计器实例（可选）
            cfg: 配置字典
        """
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.cfg = cfg or {}
    
    def detect_and_pose(self, frame: np.ndarray, draw_pose: bool = False) -> Dict[str, Any]:
        """
        对单帧进行检测和姿态估计
        
        Args:
            frame: 图像帧
            draw_pose: 是否在帧上绘制姿态
            
        Returns:
            包含检测和姿态估计结果的字典
        """
        # 执行检测
        detections = self.detector.run_frame(frame)
        
        result = {
            "detections": detections,
            "pose": None,
            "frame_with_pose": None
        }
        
        # 如果启用了姿态估计，执行姿态估计
        if self.pose_estimator and self.pose_estimator.pose is not None:
            try:
                keypoints = self.pose_estimator.run_frame(frame)
                result["pose"] = keypoints
                
                # 如果需要绘制姿态
                if draw_pose and CV2_AVAILABLE:
                    result["frame_with_pose"] = self.pose_estimator.draw_pose(frame.copy(), keypoints)
            except Exception as e:
                print(f"[WARN] 姿态估计失败: {e}")
        
        return result
    
    def process_video_frames(self, frames: List[np.ndarray], 
                            draw_pose: bool = False) -> List[Dict[str, Any]]:
        """
        处理视频帧序列
        
        Args:
            frames: 帧列表
            draw_pose: 是否绘制姿态
            
        Returns:
            每帧的检测和姿态估计结果
        """
        results = []
        for frame in frames:
            result = self.detect_and_pose(frame, draw_pose)
            results.append(result)
        return results

