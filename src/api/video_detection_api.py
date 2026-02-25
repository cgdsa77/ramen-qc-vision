"""
视频检测API
处理视频上传和检测
"""
import os
import cv2
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
import numpy as np

# PyTorch 2.6+ 默认 weights_only=True，会拒绝加载含 ultralytics 等自定义类的 checkpoint。
# 在导入 ultralytics 之前全局 patch，使本进程内所有 torch.load 默认 weights_only=False（仅对本项目可信权重）.
try:
    import torch
    _orig_torch_load = torch.load
    def _torch_load_weights_only_false(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs = {**kwargs, "weights_only": False}
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_weights_only_false
    if hasattr(torch, "serialization"):
        torch.serialization.load = _torch_load_weights_only_false
except Exception:
    pass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# 部分环境（如 Anaconda 下较新 ultralytics）中 ultralytics.utils.loss 无 DFLoss，导致旧版训练的
# 下面及捞面 best.pt 反序列化报错。若缺失则注入兼容的 DFLoss 类，仅用于加载 checkpoint。
try:
    import ultralytics.utils.loss as _loss_mod
    if not hasattr(_loss_mod, "DFLoss"):
        import torch.nn as nn
        import torch.nn.functional as F
        class DFLoss(nn.Module):
            def __init__(self, reg_max: int = 16) -> None:
                super().__init__()
                self.reg_max = reg_max
            def __call__(self, pred_dist, target):
                target = target.clamp_(0, self.reg_max - 1 - 0.01)
                tl, tr = target.long(), target.long() + 1
                wl = tr.float() - target
                wr = 1.0 - wl
                return (
                    F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                    + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
                ).mean(-1, keepdim=True)
        _loss_mod.DFLoss = DFLoss
except Exception:
    pass

# MediaPipe Hands：外源手部检测。支持 0.10+ tasks API 与旧版 solutions API
_mp_hands = None
_mp_hands_mode = None  # "tasks" | "solutions"

def _get_mediapipe_hands():
    global _mp_hands, _mp_hands_mode
    if _mp_hands is not None:
        return _mp_hands, _mp_hands_mode
    project_root = Path(__file__).parent.parent.parent
    task_file = project_root / "weights" / "mediapipe" / "hand_landmarker.task"
    # 1) 尝试 0.10+ tasks API（需 hand_landmarker.task）
    if task_file.exists():
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(task_file)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
            _mp_hands = HandLandmarker.create_from_options(options)
            _mp_hands_mode = "tasks"
            print("[OK] MediaPipe Hands 已加载（tasks API，手部外源模型）")
            return _mp_hands, _mp_hands_mode
        except Exception as e:
            pass
    # 2) 尝试旧版 solutions API（mediapipe<0.10）
    try:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        _mp_hands_mode = "solutions"
        print("[OK] MediaPipe Hands 已加载（solutions API）")
        return _mp_hands, _mp_hands_mode
    except Exception as e:
        print(f"[WARN] MediaPipe Hands 未可用: {e}，手部将仅用 YOLO。可安装 mediapipe 并放置 weights/mediapipe/hand_landmarker.task")
        return None, None

def _hand_landmarks_to_boxes(landmarks_list, h: int, w: int) -> List[Dict]:
    """从 21 个关键点（归一化坐标）得到 xyxy 框。"""
    out = []
    for hand_landmarks in landmarks_list:
        if hasattr(hand_landmarks, "landmark"):
            points = hand_landmarks.landmark
        else:
            points = hand_landmarks
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        x_min = max(0, int(min(xs) * w) - 10)
        x_max = min(w, int(max(xs) * w) + 10)
        y_min = max(0, int(min(ys) * h) - 10)
        y_max = min(h, int(max(ys) * h) + 10)
        if x_max <= x_min or y_max <= y_min:
            continue
        out.append({
            "class": "hand",
            "class_id": -1,
            "conf": 0.95,
            "xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)],
        })
    return out

def get_mediapipe_hand_boxes(frame_bgr: np.ndarray) -> List[Dict]:
    """
    用 MediaPipe Hands 检测手部框，返回与 detect_frame 相同格式的列表。
    用于替代 YOLO 的 hand 检测，减少脸/背景误判、提高手部召回与稳定框。
    """
    hands, mode = _get_mediapipe_hands()
    if hands is None:
        return []
    h, w = frame_bgr.shape[:2]
    try:
        if mode == "tasks":
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            import time
            ts_ms = int(time.time() * 1000) % (2 ** 31)
            result = hands.detect_for_video(mp_image, ts_ms)
            if not result.hand_landmarks:
                return []
            return _hand_landmarks_to_boxes(result.hand_landmarks, h, w)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                return []
            return _hand_landmarks_to_boxes(results.multi_hand_landmarks, h, w)
    except Exception:
        return []


# MediaPipe Face Detector：头部/人脸检测（外源），用于实时监测
_mp_face_detector = None

def _get_mediapipe_face_detector():
    global _mp_face_detector
    if _mp_face_detector is not None:
        return _mp_face_detector
    project_root = Path(__file__).parent.parent.parent
    for name in ("face_detection_short_range.tflite", "face_detector.task"):
        task_file = project_root / "weights" / "mediapipe" / name
        if not task_file.exists():
            continue
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            FaceDetector = mp.tasks.vision.FaceDetector
            FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(task_file)),
                running_mode=VisionRunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            _mp_face_detector = FaceDetector.create_from_options(options)
            print("[OK] MediaPipe Face 已加载（头部检测）")
            return _mp_face_detector
        except Exception:
            continue
    return None

def get_mediapipe_face_boxes(frame_bgr: np.ndarray) -> List[Dict]:
    """MediaPipe 人脸检测，返回「head」框，用于实时监测头部显示与负样本抑制。"""
    detector = _get_mediapipe_face_detector()
    if detector is None:
        return []
    try:
        import mediapipe as mp
        import time
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000) % (2 ** 31)
        result = detector.detect_for_video(mp_image, ts_ms)
        out = []
        for det in (result.detections or []):
            box = det.bounding_box
            x1 = max(0, box.origin_x)
            y1 = max(0, box.origin_y)
            x2 = min(w, box.origin_x + box.width)
            y2 = min(h, box.origin_y + box.height)
            if x2 <= x1 or y2 <= y1:
                continue
            conf = 0.9
            if det.categories:
                conf = float(det.categories[0].score)
            out.append({
                "class": "head",
                "class_id": -2,
                "conf": conf,
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })
        return out
    except Exception:
        return []


class VideoDetectionAPI:
    """视频检测API类"""
    
    def __init__(self, model_path: str = None, model_type: str = "cpu"):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，如果为None则尝试加载默认模型
            model_type: 'cpu' 时强制在 CPU 上加载与推理，避免 CUDA 报错；'gpu' 时使用 CUDA（若有）
        """
        self.model = None
        self.model_path = model_path
        self.model_type = model_type if model_type in ("cpu", "gpu") else "cpu"
        self._load_error = None  # 加载失败时的错误信息，便于接口返回给前端
        
        if YOLO_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """加载模型。model_type=='cpu' 时强制 CPU，避免选 CPU 模型仍用 GPU 导致 CUDA error。"""
        project_root = Path(__file__).parent.parent.parent
        
        # 尝试加载自定义训练的模型；否则使用最新训练的 best.pt（按修改时间，含 stretch_detection14 等）
        if self.model_path and os.path.exists(self.model_path):
            model_file = str(self.model_path)
        else:
            latest_best = _latest_stretch_best_pt(project_root)
            if latest_best is not None:
                model_file = str(latest_best)
                print(f"[INFO] 抻面模型: {model_file} (最新训练权重)")
            else:
                alt_model = project_root / "models" / "stretch_detection_model.pt"
                if alt_model.exists():
                    model_file = str(alt_model)
                else:
                    print("[WARN] 未找到抻面训练权重，将使用 YOLOv8 预训练模型（非 hand/rope/bundle），检测会异常。请运行训练或复制 best.pt 到 models/stretch_detection*/weights/")
                    model_file = "yolov8n.pt"
        
        try:
            import torch
            if self.model_type == "cpu":
                device = "cpu"
            elif torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"  # Apple M1/M2
            else:
                device = "cpu"
            self.device = device
            self.model = YOLO(model_file)
            if hasattr(self.model, "to"):
                self.model.to(device)
            n_cls = len(getattr(self.model, 'names', [])) if hasattr(self.model, 'names') else 0
            if n_cls != 3:
                print(f"[WARN] 抻面模型应为 3 类 (hand/noodle_rope/noodle_bundle)，当前模型有 {n_cls} 类，hand/面条束可能为 0。请使用 datasets/stretch_detection 训练的 best.pt")
            print(f"[OK] 模型加载成功: {model_file} (使用 {device})")
        except Exception as e:
            err = str(e)
            self._load_error = err
            if "Conv" in err and "bn" in err:
                self._load_error = (
                    "模型与当前 ultralytics 版本不兼容（Conv.bn 结构变更）。"
                    "请尝试：pip install ultralytics==8.0.200 后重启服务；"
                    "或使用当前环境重新训练得到新的 best.pt。详见 docs/模型与ultralytics版本兼容说明.md"
                )
            print(f"[ERROR] 模型加载失败: {e}")
            self.model = None
    
    def detect_video(self, video_path: str, conf_threshold: float = 0.20, progress_callback=None) -> Dict[str, Any]:
        """
        检测视频中的所有帧
        
        Args:
            video_path: 视频文件路径
            conf_threshold: 置信度阈值，略提高可减少新视频上的重叠框（默认 0.20）
            progress_callback: 可选，每处理一定帧数调用 callback(frame_index, total_frames) 便于前端展示进度
            
        Returns:
            包含检测结果的字典
        """
        if not self.model:
            raise ValueError("检测模型未加载")
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        
        detections = []
        frame_index = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"  FPS: {fps}, 总帧数: {total_frames}")
        
        # 固定推理尺寸，避免取消后换视频（如 cm1→cm10）因分辨率不同导致张量 5040/4620 不匹配
        INFER_SIZE = 640
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            # 先缩放到固定尺寸再推理，保证模型输入始终一致
            frame_infer = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
            results = self.model.predict(
                source=frame_infer,
                verbose=False,
                conf=conf_threshold,
                device='cpu',
                imgsz=INFER_SIZE
            )
            
            frame_detections = []
            scale_x, scale_y = w / INFER_SIZE, h / INFER_SIZE
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 从 640x640 映射回原图坐标
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    if not isinstance(cls_name, str):
                        cls_name = str(cls_name)
                    
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    frame_detections.append({
                        'class': cls_name,
                        'class_id': cls_id,
                        'conf': conf,
                        'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
            
            # 按类别 NMS，减轻同一目标多框、重叠框（新视频上更明显）
            if frame_detections:
                frame_detections = self._nms_per_class(frame_detections, iou_threshold=0.45)
            
            # MediaPipe 头部检测：得到 head 框，过滤与 head 重叠的 hand，并追加 head 到本帧
            head_boxes = get_mediapipe_face_boxes(frame)
            hand_class_names = ('hand', '0')
            def is_hand_overlapping_head(det):
                if det.get('class') not in hand_class_names and det.get('class_id') != 0:
                    return False
                da = det.get('xyxy') or []
                if len(da) < 4:
                    return False
                for h in head_boxes:
                    hb = h.get('xyxy') or []
                    if len(hb) < 4:
                        continue
                    if self._iou_xyxy(da, hb) > 0.25:
                        return True
                return False
            frame_detections = [d for d in frame_detections if not is_hand_overlapping_head(d)]
            for head_box in head_boxes:
                xy = head_box.get('xyxy', [])
                if len(xy) < 4:
                    continue
                x1, y1, x2, y2 = xy[0], xy[1], xy[2], xy[3]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                frame_detections.append({
                    'class': 'head',
                    'class_id': -2,
                    'conf': head_box.get('conf', 0.9),
                    'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            
            detections.append({
                'frame_index': frame_index,
                'detections': frame_detections
            })
            
            frame_index += 1
            
            # 进度（终端 + 回调）；每 10 帧检查一次取消，回调中若抛出异常则向上抛出以真正停止
            if progress_callback and frame_index % 10 == 0:
                progress_callback(frame_index, total_frames)
            if frame_index % 30 == 0:
                progress = (frame_index / total_frames) * 100
                print(f"  进度: {progress:.1f}% ({frame_index}/{total_frames})")
        
        # 确保释放视频资源
        cap.release()
        
        if progress_callback:
            progress_callback(frame_index, total_frames)
        
        # 强制释放OpenCV资源
        import gc
        gc.collect()
        
        print(f"检测完成！共处理 {frame_index} 帧")
        
        return {
            'success': True,
            'fps': fps,
            'total_frames': frame_index,
            'detections': detections
        }

    def _iou_xyxy(self, a: List[float], b: List[float]) -> float:
        """计算两个 xyxy 框的 IoU。"""
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        sa = (ax2 - ax1) * (ay2 - ay1)
        sb = (bx2 - bx1) * (by2 - by1)
        return inter / (sa + sb - inter) if (sa + sb - inter) > 0 else 0.0

    def _nms_per_class(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """按类别做 NMS：同类别内 IoU 超过阈值的保留置信度最高的框，抑制其余。减轻一只手多个 hand 框、重叠框。"""
        from collections import defaultdict
        by_class = defaultdict(list)
        for d in detections:
            by_class[d["class"]].append(d)
        out = []
        for _cls, lst in by_class.items():
            lst = sorted(lst, key=lambda x: -x["conf"])
            kept = []
            for d in lst:
                xyxy = d["xyxy"]
                skip = False
                for k in kept:
                    if self._iou_xyxy(xyxy, k["xyxy"]) >= iou_threshold:
                        skip = True
                        break
                if not skip:
                    kept.append(d)
            out.extend(kept)
        return out

    # 按类别固定颜色，避免帧间顺序变化导致红绿蓝闪烁
    _CLASS_COLORS = {
        "hand": (255, 128, 0),       # BGR 橙，手部
        "head": (200, 0, 200),      # 紫，头部
        "noodle_rope": (0, 200, 0), # 绿，面条
        "noodle": (0, 200, 0),
        "noodle_bundle": (0, 180, 0),
        "face": (200, 0, 200),
    }
    _FALLBACK_COLORS = [
        (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 200, 255),
        (128, 255, 128), (255, 128, 128), (128, 128, 255), (200, 200, 0),
    ]

    def _color_for_class(self, cls_name: str) -> tuple:
        """按类别名返回固定 BGR 颜色，同一类别每帧一致。"""
        key = (cls_name or "").strip().lower()
        if key in self._CLASS_COLORS:
            return self._CLASS_COLORS[key]
        idx = sum(ord(c) for c in key) % len(self._FALLBACK_COLORS)
        return self._FALLBACK_COLORS[idx]

    def _draw_detections(self, frame_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在 BGR 图上绘制检测框与标签（按类别固定颜色，不闪烁）。"""
        img = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(round(x)) for x in d["xyxy"]]
            cls_name = d.get("class", "")
            conf = d.get("conf", 0)
            color = self._color_for_class(cls_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def detect_frame(self, frame_bgr: np.ndarray, conf_threshold: float = 0.25, draw_boxes: bool = True,
                     iou_threshold: float = 0.45, nms_per_class: bool = True,
                     use_mediapipe_hands: bool = False, use_mediapipe_face: bool = False,
                     stage: str = "stretch") -> tuple:
        """
        对单帧图像做检测，用于实时流。
        use_mediapipe_face: 若 True，增加 MediaPipe 人脸框作为 head，并用于抑制与头部重叠的 noodle 误检。
        stage: stretch 时对 noodle_rope/noodle_bundle 提高置信度要求以减轻窗帘/褶皱误判。
        """
        if self.model is None:
            return (frame_bgr.copy(), [])
        INFER_SIZE = 640
        h, w = frame_bgr.shape[:2]
        frame_infer = cv2.resize(frame_bgr, (INFER_SIZE, INFER_SIZE))
        scale_x, scale_y = w / INFER_SIZE, h / INFER_SIZE
        try:
            results = self.model.predict(
                source=frame_infer,
                verbose=False,
                conf=conf_threshold,
                iou=iou_threshold,
                device='cpu',
                imgsz=INFER_SIZE
            )
        except Exception:
            return (frame_bgr.copy(), [])
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'conf': conf,
                    'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                })
        # 抻面阶段：提高 noodle_rope / noodle_bundle 置信度门槛，减少窗帘/褶皱误判
        if stage == "stretch":
            def is_noodle_low_conf(d):
                c = (d.get("class") or "").lower()
                return (c in ("noodle_rope", "noodle_bundle") and (d.get("conf") or 0) < 0.7)
            detections = [d for d in detections if not is_noodle_low_conf(d)]
        # 头部检测（外源）并抑制与头部重叠的 noodle 框
        head_boxes = []
        if use_mediapipe_face:
            head_boxes = get_mediapipe_face_boxes(frame_bgr)
            detections = detections + head_boxes
            # 去掉与头部重叠较大的 noodle_rope/noodle_bundle，避免脸被标成面条
            def overlaps_head(d):
                c = (d.get("class") or "").lower()
                if c not in ("noodle_rope", "noodle_bundle"):
                    return False
                for h in head_boxes:
                    if self._iou_xyxy(d["xyxy"], h["xyxy"]) > 0.25:
                        return True
                return False
            detections = [d for d in detections if not overlaps_head(d)]
        # 手部外源模型
        if use_mediapipe_hands:
            hand_boxes = get_mediapipe_hand_boxes(frame_bgr)
            detections = [d for d in detections if "hand" not in (d.get("class") or "").lower()]
            detections = detections + hand_boxes
        if nms_per_class and detections:
            detections = self._nms_per_class(detections, iou_threshold=0.5)
        if draw_boxes:
            return (self._draw_detections(frame_bgr, detections), detections)
        return (frame_bgr.copy(), detections)
    
    async def process_uploaded_video(self, file: UploadFile, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        处理上传的视频文件
        
        Args:
            file: 上传的文件
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果
        """
        # 保存临时文件
        import time
        suffix = Path(file.filename).suffix if file.filename else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 执行检测
            result = self.detect_video(tmp_path, conf_threshold)
            
            # 等待一下，确保所有文件句柄都已关闭
            time.sleep(0.5)
            
            return result
        finally:
            # 延迟删除临时文件，确保文件已完全释放
            import threading
            def delayed_delete(path):
                time.sleep(2)  # 等待2秒
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    print(f"[WARN] 删除临时文件失败: {e}")
            
            # 在后台线程中延迟删除
            thread = threading.Thread(target=delayed_delete, args=(tmp_path,))
            thread.daemon = True
            thread.start()


# 按「抻面/下面捞面」×「CPU/GPU 模型」缓存检测器（预留：后续可接入 GPU 训练权重）
_detector_cpu = None
_detector_gpu = None
_boiling_detector_cpu = None
_boiling_detector_gpu = None


def _latest_stretch_best_pt(project_root: Path) -> Optional[Path]:
    """返回抻面模型中修改时间最新的 best.pt（含 stretch_detection 与 stretch_detection2/14 等）。"""
    models_dir = project_root / "models"
    if not models_dir.exists():
        return None
    candidates = []
    for d in models_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("stretch_detection"):
            continue
        best_pt = d / "weights" / "best.pt"
        if best_pt.exists():
            candidates.append((best_pt.stat().st_mtime, best_pt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_stretch_model_path(model_type: str) -> str:
    """解析抻面模型路径：优先使用最新训练的 best.pt（按修改时间），model_type 为 'cpu' 或 'gpu'（预留）。"""
    project_root = Path(__file__).parent.parent.parent
    latest = _latest_stretch_best_pt(project_root)
    w = project_root / "models" / "stretch_detection" / "weights"
    if model_type == "gpu":
        if (w / "best_gpu.pt").exists():
            return str(w / "best_gpu.pt")
        if latest:
            print("[INFO] 未找到抻面 GPU 权重(best_gpu.pt)，使用最新 best.pt")
            return str(latest)
        if (w / "best_cpu.pt").exists():
            return str(w / "best_cpu.pt")
        if (w / "best.pt").exists():
            return str(w / "best.pt")
        return str(w / "best_gpu.pt")
    # cpu：优先最新 best.pt，再 fallback 到固定路径
    if latest:
        return str(latest)
    if (w / "best_cpu.pt").exists():
        return str(w / "best_cpu.pt")
    if (w / "best.pt").exists():
        return str(w / "best.pt")
    alt = project_root / "models" / "stretch_detection_model.pt"
    if alt.exists():
        return str(alt)
    return "yolov8n.pt"


def _latest_boiling_best_pt(project_root: Path) -> Optional[Path]:
    """返回下面及捞面模型中修改时间最新的 best.pt（含 boiling_scooping_detection、boiling_scooping_detection2 等）。"""
    models_dir = project_root / "models"
    if not models_dir.exists():
        return None
    candidates = []
    for d in models_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("boiling_scooping_detection"):
            continue
        best_pt = d / "weights" / "best.pt"
        if best_pt.exists():
            candidates.append((best_pt.stat().st_mtime, best_pt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_boiling_model_path(model_type: str) -> str:
    """解析下面及捞面模型路径：优先使用最新训练的 best.pt（按修改时间），model_type 为 'cpu' 或 'gpu'（预留）。"""
    project_root = Path(__file__).parent.parent.parent
    latest = _latest_boiling_best_pt(project_root)
    w = project_root / "models" / "boiling_scooping_detection" / "weights"
    if model_type == "gpu":
        if (w / "best_gpu.pt").exists():
            return str(w / "best_gpu.pt")
        if latest:
            print("[INFO] 未找到下面及捞面 GPU 权重(best_gpu.pt)，使用最新 best.pt")
            return str(latest)
        if (w / "best_cpu.pt").exists():
            return str(w / "best_cpu.pt")
        if (w / "best.pt").exists():
            return str(w / "best.pt")
        return str(w / "best_gpu.pt")
    if latest:
        return str(latest)
    if (w / "best_cpu.pt").exists():
        return str(w / "best_cpu.pt")
    if (w / "best.pt").exists():
        return str(w / "best.pt")
    alt = project_root / "models" / "boiling_scooping_detection_model.pt"
    if alt.exists():
        return str(alt)
    return "yolov8n.pt"


def get_detector(model_path: str = None, model_type: str = "cpu") -> VideoDetectionAPI:
    """获取抻面检测器。model_type: 'cpu' 时强制 CPU 加载与推理；路径变化时自动换用最新 best.pt。"""
    global _detector_cpu, _detector_gpu
    path = model_path
    if path is None:
        path = _resolve_stretch_model_path(model_type if model_type in ("cpu", "gpu") else "cpu")
    mt = model_type if model_type in ("cpu", "gpu") else "cpu"
    if mt == "gpu":
        if _detector_gpu is None:
            _detector_gpu = VideoDetectionAPI(path, model_type="gpu")
        elif getattr(_detector_gpu, "model_path", None) != path:
            _detector_gpu = VideoDetectionAPI(path, model_type="gpu")
        return _detector_gpu
    if _detector_cpu is None:
        _detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    elif getattr(_detector_cpu, "device", None) != "cpu":
        _detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    elif getattr(_detector_cpu, "model_path", None) != path:
        _detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    return _detector_cpu


def get_boiling_scooping_detector(model_path: str = None, model_type: str = "cpu") -> VideoDetectionAPI:
    """获取下面及捞面检测器。model_type: 'cpu' 时强制 CPU；优先最新 best.pt，路径变化时自动换用。"""
    global _boiling_detector_cpu, _boiling_detector_gpu
    path = model_path
    if path is None:
        path = _resolve_boiling_model_path(model_type if model_type in ("cpu", "gpu") else "cpu")
    mt = model_type if model_type in ("cpu", "gpu") else "cpu"
    if mt == "gpu":
        if _boiling_detector_gpu is None:
            _boiling_detector_gpu = VideoDetectionAPI(path, model_type="gpu")
        elif getattr(_boiling_detector_gpu, "model_path", None) != path:
            _boiling_detector_gpu = VideoDetectionAPI(path, model_type="gpu")
        return _boiling_detector_gpu
    if _boiling_detector_cpu is None:
        _boiling_detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    elif getattr(_boiling_detector_cpu, "device", None) != "cpu":
        _boiling_detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    elif getattr(_boiling_detector_cpu, "model_path", None) != path:
        _boiling_detector_cpu = VideoDetectionAPI(path, model_type="cpu")
    return _boiling_detector_cpu

