"""
拉面成品质感预测：加载训练好的融合模型，从图像预测 noodle_quality，再交由 ProductScorer 算分。
"""
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
MODEL_DIR = _project_root / "data" / "scores" / "拉面成品" / "product_model"
FEATURE_DIM = 512
IMAGE_SIZE = 224


class ProductPredictor:
    """
    使用训练好的 ResNet+手工特征 + RF/LR 融合模型预测面条质感等级。
    不依赖汤型/辣椒参与预测，仅从图像学习质感。
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self._scaler = None
        self._rf = None
        self._lr = None
        self._config = None
        self._resnet = None
        self._transform = None
        self._load_model()

    def _load_model(self) -> None:
        import json
        try:
            import joblib
        except ImportError:
            return
        joblib_path = self.model_dir / "product_model.joblib"
        if not joblib_path.exists():
            return
        data = joblib.load(joblib_path)
        self._scaler = data.get("scaler")
        self._rf = data.get("rf")
        self._lr = data.get("lr")
        self._config = data.get("config", {})
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self._scaler is not None and self._rf is not None and self._lr is not None

    def _extract_handcrafted(self, image_path: Path) -> np.ndarray:
        """27 维：24 色直方图 + 3 纹理/亮度，与训练脚本一致。"""
        try:
            import cv2
        except ImportError:
            return np.zeros(27, dtype=np.float32)
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(27, dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_b = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()
        hist_bgr = np.concatenate([hist_b, hist_g, hist_r])
        hist_bgr = hist_bgr / (hist_bgr.sum() + 1e-8)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (edges.size + 1e-8)
        mean_bright = float(np.mean(gray))
        std_bright = float(np.std(gray))
        hand = np.concatenate([
            hist_bgr.astype(np.float32),
            np.array([edge_ratio, mean_bright / 255.0, std_bright / 255.0], dtype=np.float32),
        ])
        return hand.astype(np.float32)

    def _extract_resnet(self, image_path: Path, device: str = "cpu") -> np.ndarray:
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as T
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        if self._transform is None:
            self._transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if self._resnet is None:
            self._resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self._resnet.fc = torch.nn.Identity()
            self._resnet.eval()
            self._resnet.to(device)
        try:
            img = Image.open(str(image_path)).convert("RGB")
        except Exception:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        x = self._transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = self._resnet(x)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def _extract_features(self, image_path: Path) -> Optional[np.ndarray]:
        if not self._config:
            return None
        use_resnet = self._config.get("use_resnet", True)
        use_hand = self._config.get("use_hand", True)
        feats = []
        if use_resnet:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            feats.append(self._extract_resnet(image_path, device))
        if use_hand:
            hand = self._extract_handcrafted(image_path)
            expected_dim = self._config.get("feature_dim")
            if expected_dim == 532 and len(hand) == 27:
                hand = hand[:20].astype(np.float32)
            feats.append(hand)
        if not feats:
            feats.append(self._extract_handcrafted(image_path))
        return np.concatenate(feats)

    def predict_quality(self, image_path: Path) -> Dict[str, Any]:
        """
        对单张图片预测 noodle_quality。
        Returns:
            noodle_quality: "excellent" | "good" | "fair"
            confidence: 0~1
            total_score: 若已加载 ProductScorer 规则则一并计算 0~100 分
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return {"noodle_quality": "fair", "confidence": 0.0, "total_score": None, "error": "file not found"}
        if not self.is_loaded:
            return {"noodle_quality": "fair", "confidence": 0.0, "total_score": None, "error": "model not loaded"}

        X = self._extract_features(image_path)
        if X is None:
            return {"noodle_quality": "fair", "confidence": 0.0, "total_score": None, "error": "feature extract failed"}
        X = X.reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        idx2label = self._config.get("idx2label", {0: "fair", 1: "good", 2: "excellent"})
        # JSON 保存后键可能为字符串
        idx2label = {int(k): v for k, v in idx2label.items()}
        p_rf = self._rf.predict_proba(X_scaled)[0]
        p_lr = self._lr.predict_proba(X_scaled)[0]
        p_avg = (p_rf + p_lr) / 2
        pred_idx = int(np.argmax(p_avg))
        confidence = float(p_avg[pred_idx])
        noodle_quality = idx2label.get(pred_idx, "fair")

        # 置信度过低时视为非拉面成品图（如人脸、无关场景），不给出正常评分
        CONFIDENCE_THRESHOLD = 0.65
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "error": "not_ramen",
                "message": "无法识别为拉面成品，请上传成品图片（当前图片与训练数据差异较大，可能为人脸、无关场景等）",
                "confidence": round(confidence, 4),
                "noodle_quality": noodle_quality,
            }

        total_score = None
        try:
            from .product_scorer import ProductScorer
            scorer = ProductScorer()
            out = scorer.score_from_prediction(noodle_quality, presentation_bonus=0.0)
            total_score = out["total_score"]
        except Exception:
            pass

        probs = {idx2label.get(i, "?"): round(float(p_avg[i]), 4) for i in range(len(p_avg))}
        return {
            "noodle_quality": noodle_quality,
            "confidence": round(confidence, 4),
            "total_score": total_score,
            "probabilities": probs,
        }

    def predict_and_score(self, image_path: Path, presentation_bonus: float = 0.0) -> Dict[str, Any]:
        """预测质感并返回完整评分（含 total_score, s_texture 等）。"""
        pred = self.predict_quality(image_path)
        if pred.get("error"):
            return pred
        try:
            from .product_scorer import ProductScorer
            scorer = ProductScorer()
            score_out = scorer.score_from_prediction(pred["noodle_quality"], presentation_bonus=presentation_bonus)
            pred.update(score_out)
        except Exception as e:
            pred["score_error"] = str(e)
        return pred


def score_annotations_batch(use_presentation: bool = True) -> List[Dict[str, Any]]:
    """直接对已有标注批量计算成品得分（不跑图像模型）。"""
    from .product_scorer import ProductScorer, load_annotations
    items = load_annotations()
    scorer = ProductScorer()
    return scorer.batch_score_from_annotations(items, use_presentation=use_presentation)
