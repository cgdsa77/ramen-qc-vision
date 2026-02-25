from typing import Any, Dict
from data.video_reader import VideoReader
from models.pose import PoseEstimator
from models.detector import ObjectDetector
from models.thickness import ThicknessEstimator
from scoring.scorer import Scorer
from utils.video_ops import Segmenter


class Pipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        self.pose = PoseEstimator(cfg)
        self.detector = ObjectDetector(cfg)
        self.thickness = ThicknessEstimator(cfg)
        self.scorer = Scorer(cfg)

    def run(self, input_path: str) -> Dict[str, Any]:
        """
        运行完整的分析流程
        
        注意：当前阶段主要关注检测功能，其他模块为预留接口
        """
        video = VideoReader(input_path, fps=self.cfg["input"].get("frame_rate", 15))
        segments = self.segmenter.split(video)
        events = []
        
        for seg in segments:
            # 检测（主要功能）
            dets = self.detector.run(seg)
            
            # 姿态估计（可选，独立运行，不影响检测）
            pose_seq = self.pose.run(seg) if self.pose else []
            
            # 粗细分析（可选，基于检测结果）
            thickness = self.thickness.run(seg, dets) if self.thickness else []
            
            # 评分（预留接口，当前可能不完整）
            # score = self.scorer.score(seg, pose_seq, dets, thickness) if self.scorer else None
            
            events.append({
                "segment_id": getattr(seg, "id", None),
                "detections": dets,  # 检测结果（主要输出）
                "pose": pose_seq,  # 姿态估计结果（可选）
                "thickness": thickness,  # 粗细分析结果（可选）
                # "score": score,  # 评分结果（预留，后续添加）
            })
        
        return {
            "events": events,
            # "summary": self.scorer.aggregate(events) if self.scorer else {},  # 预留
        }


def build_pipeline(cfg: Dict[str, Any]) -> Pipeline:
    return Pipeline(cfg)
