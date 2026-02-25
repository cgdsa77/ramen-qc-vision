"""
拉面成品质感预测模型训练
基于标注数据，融合 ResNet 深度特征 + 手工纹理/颜色特征，训练成品质感（noodle_quality）分类器。
以面条质感为主体，汤型/辣椒不参与标签，仅用图像与标注中的 noodle_quality 进行监督学习。
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np

# 图像与标注路径
RAW_DIR = project_root / "data" / "raw" / "拉面成品"
SCORES_DIR = project_root / "data" / "scores" / "拉面成品"
ANNOTATIONS_FILE = SCORES_DIR / "annotations.json"
MODEL_DIR = SCORES_DIR / "product_model"
FEATURE_DIM = 512
IMAGE_SIZE = 224
QUALITY_ORDER = ["fair", "good", "excellent"]  # 仅数据集中出现的等级，用于分类


def load_annotation_items():
    """加载标注列表，返回 [(image_name, noodle_quality), ...]"""
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"标注文件不存在: {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    out = []
    for it in items:
        img = (it.get("image") or "").strip()
        nq = (it.get("noodle_quality") or "").strip().lower()
        if not img or not nq:
            continue
        if nq not in QUALITY_ORDER:
            continue  # 仅训练 excellent/good/fair
        out.append((img, nq))
    return out


HAND_FEATURE_DIM = 27  # 24 色直方图 + 3 纹理/亮度，与 predict 一致


def extract_features_cv(image_path: Path) -> np.ndarray:
    """手工特征：颜色直方图(24) + 边缘密度/亮度/对比度(3)，共 27 维。读图失败时用临时英文路径重试。"""
    try:
        import cv2
        import shutil
        import tempfile
    except ImportError:
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    img = cv2.imread(str(image_path))
    if img is None and image_path.exists():
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            shutil.copy2(image_path, tmp_path)
            img = cv2.imread(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    if img is None:
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
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
    return hand


def extract_features_resnet(image_path: Path, device: str = "cpu") -> np.ndarray:
    """ResNet18 全局池化特征，512 维。"""
    try:
        import torch
    except ImportError:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    img_tensor = _load_image_tensor(image_path)
    if img_tensor is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    img_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = _get_resnet_feature(img_batch, device)
    if feat is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    return feat.cpu().numpy().flatten().astype(np.float32)


# 全局 ResNet 与 transform，避免重复加载
_resnet = None
_transform = None


def _ensure_resnet(device: str):
    global _resnet
    if _resnet is not None:
        return
    try:
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        _resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # noqa: PLW0603
        _resnet.fc = torch.nn.Identity()
        _resnet.eval()
        _resnet.to(device)
    except Exception:
        _resnet = None  # noqa: PLW0603


def _ensure_transform():
    global _transform
    if _transform is not None:
        return
    try:
        import torchvision.transforms as T
        _transform = T.Compose([  # noqa: PLW0603
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except Exception:
        _transform = None


def _load_image_tensor(image_path: Path):
    try:
        from PIL import Image
    except ImportError:
        return None
    _ensure_transform()
    if _transform is None:
        return None
    try:
        img = Image.open(str(image_path)).convert("RGB")
    except Exception:
        return None
    return _transform(img)


def _get_resnet_feature(x, device: str):
    _ensure_resnet(device)
    if _resnet is None:
        try:
            import torch
            return torch.zeros(x.size(0), FEATURE_DIM, device=x.device)
        except Exception:
            return None
    return _resnet(x)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="训练拉面成品质感预测模型（融合 ResNet + 手工特征）")
    ap.add_argument("--no-resnet", action="store_true", help="仅用手工特征训练")
    ap.add_argument("--no-hand", action="store_true", help="仅用 ResNet 特征训练")
    ap.add_argument("--out-dir", type=str, default=None, help="模型输出目录，默认 data/scores/拉面成品/product_model")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_annotation_items()
    if not pairs:
        print("未找到有效标注（需 noodle_quality 为 excellent/good/fair）")
        return 1

    if not RAW_DIR.exists():
        print(f"图片目录不存在: {RAW_DIR}")
        return 1

    use_resnet = not args.no_resnet
    use_hand = not args.no_hand
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
        use_resnet = False

    X_list, y_list = [], []
    for img_name, nq in pairs:
        path = RAW_DIR / img_name
        if not path.exists():
            continue
        feats = []
        if use_resnet:
            feats.append(extract_features_resnet(path, device))
        if use_hand:
            feats.append(extract_features_cv(path))
        if not feats:
            feats.append(extract_features_cv(path))
        X_list.append(np.concatenate(feats))
        y_list.append(nq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    n_feat = X.shape[1]
    print(f"样本数: {len(y)}, 特征维度: {n_feat}, 类别: {QUALITY_ORDER}")

    # 标签转整数
    label2idx = {c: i for i, c in enumerate(QUALITY_ORDER)}
    y_idx = np.array([label2idx[yi] for yi in y])

    # 融合多方法：RandomForest + LogisticRegression 软投票
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError:
        print("请安装 scikit-learn: pip install scikit-learn")
        return 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    lr = LogisticRegression(max_iter=500, random_state=42)
    rf.fit(X_scaled, y_idx)
    lr.fit(X_scaled, y_idx)

    # 交叉验证
    for name, clf in [("RF", rf), ("LR", lr)]:
        scores = cross_val_score(clf, X_scaled, y_idx, cv=5, scoring="accuracy")
        print(f"  {name} 5-fold 准确率: {scores.mean():.3f} ± {scores.std():.3f}")

    # 软投票：平均概率后取 argmax
    def predict_ensemble(X_s):
        p_rf = rf.predict_proba(X_s)
        p_lr = lr.predict_proba(X_s)
        p_avg = (p_rf + p_lr) / 2
        return np.argmax(p_avg, axis=1), p_avg

    pred_idx, _ = predict_ensemble(X_scaled)
    acc = (pred_idx == y_idx).mean()
    print(f"融合模型训练集准确率: {acc:.3f}")

    idx2label = {i: c for i, c in enumerate(QUALITY_ORDER)}
    config = {
        "quality_order": QUALITY_ORDER,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "feature_dim": n_feat,
        "use_resnet": use_resnet,
        "use_hand": use_hand,
        "image_size": IMAGE_SIZE,
    }
    joblib.dump({
        "scaler": scaler,
        "rf": rf,
        "lr": lr,
        "config": config,
    }, out_dir / "product_model.joblib")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"模型已保存: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
