"""
从「最佳抻面模型」检测结果生成综合评分所需的 annotation_scores / annotation_confidences，
供与骨架线、DTW 等融合。
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 项目根
_project_root = Path(__file__).parent.parent.parent


def resolve_stretch_video_path(video_name: str, project_root: Optional[Path] = None) -> Optional[Path]:
    """解析抻面视频路径（cm1~cm12 等）。"""
    root = project_root or _project_root
    candidates = [
        root / "data" / "raw" / "抻面" / f"{video_name}.mp4",
        root / "data" / "raw" / "抻面" / f"{video_name}.MP4",
        root / "data" / "videos" / "抻面" / f"{video_name}.mp4",
        root / "data" / "processed_videos" / "抻面" / f"{video_name}.mp4",
        root / "data" / "raw" / f"{video_name}.mp4",
        root / "data" / "videos" / f"{video_name}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_annotation_from_detection(
    video_path: str,
    project_root: Optional[Path] = None,
    max_frames: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[int, float], str]:
    """
    使用当前最佳抻面模型对视频做检测，并用 StretchScorer 得到逐帧属性分，
    转为 EnhancedComprehensiveScorer 所需的 annotation_scores / annotation_confidences。

    Args:
        video_path: 视频文件路径
        project_root: 项目根目录
        max_frames: 若给出（例如骨架线帧数），只使用前 max_frames 帧，便于与骨架对齐

    Returns:
        (annotation_scores, annotation_confidences, model_source)
        model_source 为当前使用的权重路径说明，用于写入报告。
    """
    root = project_root or _project_root
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.api.video_detection_api import get_detector, _resolve_stretch_model_path
    from src.scoring.stretch_scorer import StretchScorer

    detector = get_detector(model_type="cpu")
    model_source = _resolve_stretch_model_path("cpu") or getattr(detector, "model_path", None) or "当前最佳抻面模型(latest best.pt)"
    if not detector.model:
        raise RuntimeError("抻面检测模型未加载，无法生成检测标注分")

    result = detector.detect_video(str(video_path), conf_threshold=0.20)
    if not result or not result.get("success", True):
        raise RuntimeError(result.get("error", "检测失败"))

    detections = result.get("detections", [])
    total = result.get("total_frames", len(detections))
    if total == 0:
        raise RuntimeError("视频无有效帧")
    print(f"  检测完成（{total} 帧），正在逐帧评分（可能较久）...")
    sys.stdout.flush()

    classes = ["hand", "noodle_rope", "noodle_bundle"]
    video_detections = []
    for frame_data in detections:
        frame_index = frame_data.get("frame_index", 0)
        frame_dets = []
        for det in frame_data.get("detections", []):
            cls_name = det.get("class")
            if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                cls_name = classes[int(cls_name)]
            if cls_name not in classes:
                continue
            xyxy = det.get("xyxy") or [0, 0, 0, 0]
            if len(xyxy) >= 4:
                w_px = xyxy[2] - xyxy[0]
                h_px = xyxy[3] - xyxy[1]
            else:
                w_px = det.get("width", 0)
                h_px = det.get("height", 0)
            frame_dets.append({
                "class": cls_name,
                "conf": det.get("conf", 0.5),
                "xyxy": xyxy,
                "width": w_px,
                "height": h_px,
            })
        video_detections.append({"frame_index": frame_index, "detections": frame_dets})

    scorer = StretchScorer()
    video_score_result = scorer.score_video(video_detections, video_path=video_path)
    frame_scores_list = video_score_result.get("frame_scores", [])
    print("  逐帧评分完成，正在转换为融合输入...")
    if hasattr(sys, 'stdout') and sys.stdout:
        sys.stdout.flush()
    if not frame_scores_list:
        raise RuntimeError("未得到逐帧评分")

    if max_frames is not None:
        frame_scores_list = frame_scores_list[: max_frames]

    annotation_scores: Dict[int, Dict[str, Dict[str, float]]] = {}
    annotation_confidences: Dict[int, float] = {}

    hand_attrs = ["position", "action", "angle", "coordination"]
    rope_attrs = ["thickness", "elasticity", "gloss", "integrity"]
    bundle_attrs = ["tightness", "uniformity"]

    for fs in frame_scores_list:
        frame_idx = fs.get("frame_index", len(annotation_scores))
        frame_ann: Dict[str, Dict[str, float]] = {}
        conf_sum = 0.0
        conf_count = 0

        by_class: Dict[str, List[Dict[str, float]]] = {"hand": [], "noodle_rope": [], "noodle_bundle": []}
        for det in fs.get("detections", []):
            cls_name = det.get("class", "")
            if cls_name not in by_class:
                continue
            s = det.get("scores", {})
            if s:
                by_class[cls_name].append(s)
            w = det.get("weighted_score", 0)
            if w > 0:
                conf_sum += min(1.0, w / 5.0)
                conf_count += 1

        for cls_name, attr_list in [("hand", hand_attrs), ("noodle_rope", rope_attrs), ("noodle_bundle", bundle_attrs)]:
            list_of_scores = by_class.get(cls_name, [])
            if not list_of_scores:
                continue
            agg: Dict[str, float] = {}
            for attr in attr_list:
                vals = [s.get(attr) for s in list_of_scores if s.get(attr) is not None]
                if vals:
                    agg[attr] = float(sum(vals) / len(vals))
            if agg:
                frame_ann[cls_name] = agg

        if frame_ann:
            annotation_scores[frame_idx] = frame_ann
        if conf_count > 0:
            annotation_confidences[frame_idx] = min(1.0, max(0.0, conf_sum / conf_count))
        else:
            annotation_confidences[frame_idx] = 0.85

    return annotation_scores, annotation_confidences, str(model_source)
