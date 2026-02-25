"""
训练自定义检测模型
基于标注数据训练YOLO检测模型
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# PyTorch 2.6+ 默认 weights_only=True，加载 last.pt/best.pt 等含 ultralytics 类的 checkpoint 会报错。
# 训练脚本内先 patch，再导入 ultralytics，以便恢复训练时能正常加载。
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

# 部分环境（如 Anaconda 新版本）ultralytics.utils.loss 无 DFLoss，恢复 last.pt 时会报错，此处注入兼容类
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


def verify_classes_consistency(labels_base: Path) -> bool:
    """
    验证所有视频目录下的classes.txt是否与主classes.txt一致
    
    Returns:
        bool: 如果一致返回True，否则返回False
    """
    main_classes_file = labels_base / "classes.txt"
    if not main_classes_file.exists():
        print(f"错误：主classes.txt文件不存在: {main_classes_file}")
        return False
    
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        main_classes = [line.strip() for line in f if line.strip()]
    
    print(f"\n验证classes.txt一致性...")
    print(f"主classes.txt顺序: {main_classes}")
    
    video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith('cm')])
    all_consistent = True
    
    for video_dir in video_dirs:
        video_classes_file = video_dir / "classes.txt"
        if not video_classes_file.exists():
            print(f"  [警告] {video_dir.name}: classes.txt 不存在")
            all_consistent = False
            continue
        
        with open(video_classes_file, 'r', encoding='utf-8') as f:
            video_classes = [line.strip() for line in f if line.strip()]
        
        if video_classes != main_classes:
            print(f"  [错误] {video_dir.name}: classes.txt顺序不一致!")
            print(f"    主文件: {main_classes}")
            print(f"    视频文件: {video_classes}")
            all_consistent = False
    
    if all_consistent:
        print(f"  [OK] 所有视频目录的classes.txt都与主文件一致")
    else:
        print(f"  [ERROR] 发现不一致的classes.txt文件，请先修复！")
    
    return all_consistent


def prepare_yolo_dataset(standard_videos: list = None, val_split: float = 0.2):
    """
    准备YOLO格式的数据集
    
    Args:
        standard_videos: 标准视频列表，默认使用cm1, cm2, cm3
        val_split: 验证集比例（0.0-1.0），如果为0则不划分验证集
    """
    if standard_videos is None:
        standard_videos = ['cm1', 'cm2', 'cm3']
    
    print(f"\n使用 {len(standard_videos)} 个视频进行训练: {standard_videos}")
    
    base_dir = project_root / "data"
    images_base = base_dir / "processed" / "抻面"
    labels_base = base_dir / "labels" / "抻面"
    
    # 验证classes.txt一致性（关键步骤）
    if not verify_classes_consistency(labels_base):
        raise ValueError("classes.txt文件不一致，请先修复后再训练！")
    
    # 创建数据集目录
    dataset_dir = project_root / "datasets" / "stretch_detection"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建YOLO格式的目录结构
    train_images_dir = dataset_dir / "images" / "train"
    train_labels_dir = dataset_dir / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果划分验证集，创建验证集目录
    val_images_dir = None
    val_labels_dir = None
    if val_split > 0:
        val_images_dir = dataset_dir / "images" / "val"
        val_labels_dir = dataset_dir / "labels" / "val"
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取类别名称
    classes_file = labels_base / "classes.txt"
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"类别: {class_names}")
    print(f"准备数据集: {standard_videos}")
    
    # 复制图片和标注文件
    train_count = 0
    val_count = 0
    
    # 计算验证集视频数量
    if val_split > 0 and len(standard_videos) > 1:
        val_video_count = max(1, int(len(standard_videos) * val_split))
        val_videos = standard_videos[-val_video_count:]  # 使用最后几个视频作为验证集
        train_videos = standard_videos[:-val_video_count]
    else:
        val_videos = []
        train_videos = standard_videos
    
    print(f"  训练集视频: {train_videos}")
    if val_videos:
        print(f"  验证集视频: {val_videos}")
    
    # 处理训练集
    for video_name in train_videos:
        images_dir = images_base / video_name
        labels_dir = labels_base / video_name
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"[WARN] 跳过 {video_name}（目录不存在）")
            continue
        
        video_img_count = 0
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        shutil.copy2(img_file, train_images_dir / img_file.name)
                        shutil.copy2(label_file, train_labels_dir / f"{img_file.stem}.txt")
                        train_count += 1
                        video_img_count += 1
        print(f"  [OK] {video_name} (训练集): {video_img_count} 张图片")
    
    # 处理验证集
    for video_name in val_videos:
        images_dir = images_base / video_name
        labels_dir = labels_base / video_name
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"[WARN] 跳过 {video_name}（目录不存在）")
            continue
        
        video_img_count = 0
        for img_file in images_dir.glob("*.jpg"):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        shutil.copy2(img_file, val_images_dir / img_file.name)
                        shutil.copy2(label_file, val_labels_dir / f"{img_file.stem}.txt")
                        val_count += 1
                        video_img_count += 1
        print(f"  [OK] {video_name} (验证集): {video_img_count} 张图片")
    
    # 创建data.yaml配置文件
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val' if val_split > 0 and val_count > 0 else 'images/train',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n数据集准备完成！")
    print(f"  - 训练集图片: {train_count} 张")
    if val_count > 0:
        print(f"  - 验证集图片: {val_count} 张")
    print(f"  - 总图片数: {train_count + val_count} 张")
    print(f"  - 类别数: {len(class_names)}")
    print(f"  - 数据集路径: {dataset_dir}")
    print(f"  - 配置文件: {yaml_path}")
    
    return dataset_dir, yaml_path, train_count + val_count


def train_yolo_model(dataset_yaml_path: Path, epochs: int = 300, imgsz: int = 640,
                    batch: int = 8, lr0: float = 0.01, patience: int = 50, resume: bool = True,
                    model_size: str = 'm', cos_lr: bool = True, label_smoothing: float = 0.1,
                    close_mosaic: int = 10, device: str = 'cpu'):
    """
    训练YOLO模型（采用更先进配置以提升精度）
    
    Args:
        dataset_yaml_path: 数据集配置文件路径
        epochs: 训练轮数（默认300）
        imgsz: 图片尺寸
        batch: 批次大小
        lr0: 初始学习率
        patience: 早停耐心值
        resume: 是否从检查点恢复
        model_size: 模型规模 n/s/m/l/x，越大精度越高、越慢（默认 m）
        cos_lr: 是否使用余弦学习率衰减
        label_smoothing: 标签平滑系数
        close_mosaic: 最后 N 轮关闭 Mosaic 做精细调优
        device: 训练设备 'cpu' 或 'cuda'（预留，后续可接 GPU）
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误：未安装 ultralytics。请运行: pip install ultralytics")
        return None

    weights_dir = project_root / "models" / "stretch_detection" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    base_weights = f'yolov8{model_size}.pt'
    
    print(f"\n开始训练YOLO模型（进阶配置）...")
    print(f"  - 基础模型: {base_weights}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 图片尺寸: {imgsz}")
    print(f"  - 批次大小: {batch}")
    print(f"  - 初始学习率: {lr0}")
    print(f"  - 早停耐心值: {patience}")
    print(f"  - 余弦学习率: {cos_lr}")
    print(f"  - 标签平滑: {label_smoothing}")
    print(f"  - 最后 {close_mosaic} 轮关闭 Mosaic")
    print(f"  - 设备: {device}")
    
    last_ckpt = weights_dir / "last.pt"
    # 若固定路径没有 last.pt，尝试从最近一次运行目录恢复（如 stretch_detection14）
    if resume and not last_ckpt.exists():
        models_dir = project_root / "models"
        candidates = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("stretch_detection") and (d / "weights" / "last.pt").exists()],
            key=lambda d: (d / "weights" / "last.pt").stat().st_mtime,
            reverse=True,
        )
        if candidates:
            last_ckpt = candidates[0] / "weights" / "last.pt"
            print(f"  - 在 {candidates[0].name} 下找到检查点: {last_ckpt}")
    def _checkpoint_valid(path):
        try:
            import torch
            try:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(str(path), map_location="cpu")
            return ckpt.get("model") is not None
        except Exception:
            return False

    do_resume = False
    if resume and last_ckpt.exists():
        if _checkpoint_valid(last_ckpt):
            do_resume = True
        else:
            print(f"  - [警告] 检查点无效或缺少 model: {last_ckpt}")
            # 尝试从最近一次运行目录用有效检查点恢复（如 stretch_detection14）
            models_dir = project_root / "models"
            candidates = sorted(
                [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("stretch_detection") and (d / "weights" / "last.pt").exists()],
                key=lambda d: (d / "weights" / "last.pt").stat().st_mtime,
                reverse=True,
            )
            for d in candidates:
                cand = d / "weights" / "last.pt"
                if cand != last_ckpt and _checkpoint_valid(cand):
                    last_ckpt = cand
                    do_resume = True
                    print(f"  - 改用有效检查点: {last_ckpt}")
                    break
            if not do_resume:
                print(f"  - 将从头训练")
    if do_resume:
        print(f"  - 从检查点恢复: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"  - 从头开始训练")
        model = YOLO(base_weights)
        print(f"  - 使用预训练: {base_weights}")
    
    # exist_ok=True：始终写入同一目录；resume=路径 让 Ultralytics 从该检查点续训（传 True 可能被 checkpoint 覆盖）
    train_kw = dict(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,
        name='stretch_detection',
        project=str(project_root / "models"),
        save=True,
        plots=True,
        device=device,
        cos_lr=cos_lr,
        label_smoothing=label_smoothing,
        close_mosaic=close_mosaic,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.2,
        scale=0.6,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
    )
    train_kw["exist_ok"] = True
    if do_resume:
        train_kw["resume"] = str(last_ckpt)
    results = model.train(**train_kw)
    
    best_model_path = weights_dir / "best.pt"
    best_cpu_path = weights_dir / "best_cpu.pt"
    final_model_path = project_root / "models" / "stretch_detection_model.pt"
    
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        shutil.copy2(best_model_path, best_cpu_path)
        print(f"\n训练完成！")
        print(f"  - 最佳模型: {best_model_path}")
        print(f"  - CPU 检测用权重（预留）: {best_cpu_path}")
        print(f"  - 兼容路径: {final_model_path}")
        return final_model_path
    else:
        print("警告：未找到训练好的模型")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练抻面检测模型（cm1~cm7 + cm10~cm12）')
    parser.add_argument('--videos', nargs='+',
                       default=['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6', 'cm7', 'cm10', 'cm11', 'cm12'],
                       help='视频列表（默认：cm1-cm7 与 cm10-cm12）')
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数（默认：300）')
    parser.add_argument('--batch', type=int, default=8,
                       help='批次大小（默认：8）')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图片尺寸（默认：640）')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='初始学习率（默认：0.01）')
    parser.add_argument('--patience', type=int, default=50,
                       help='早停耐心值（默认：50）')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='验证集比例（默认：0.2）')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                       help='模型规模 n/s/m/l/x，越大精度越高（默认：m）')
    parser.add_argument('--no-cos-lr', dest='cos_lr', action='store_false', default=True,
                       help='禁用余弦学习率（默认启用 cos_lr）')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='标签平滑（默认：0.1）')
    parser.add_argument('--close-mosaic', type=int, default=10,
                       help='最后 N 轮关闭 Mosaic（默认：10）')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='训练设备（默认：cpu，后续可改为 cuda）')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='从检查点恢复（默认：True）')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='从头训练')
    
    args = parser.parse_args()
    
    print("="*60)
    print("训练抻面检测模型（进阶配置）")
    print("="*60)
    print(f"\n训练参数：")
    print(f"  - 视频列表: {args.videos}")
    print(f"  - 模型规模: {args.model_size}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 批次大小: {args.batch}")
    print(f"  - 余弦学习率: {args.cos_lr}")
    print(f"  - 标签平滑: {args.label_smoothing}")
    print(f"  - 关闭 Mosaic 轮数: {args.close_mosaic}")
    print(f"  - 设备: {args.device}")
    print("="*60)
    
    dataset_dir, yaml_path, image_count = prepare_yolo_dataset(
        standard_videos=args.videos,
        val_split=args.val_split
    )
    
    if image_count == 0:
        print("错误：没有找到有效的训练数据")
        return
    
    model_path = train_yolo_model(
        yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        resume=args.resume,
        model_size=args.model_size,
        cos_lr=args.cos_lr,
        label_smoothing=args.label_smoothing,
        close_mosaic=args.close_mosaic,
        device=args.device,
    )
    
    if model_path:
        print("\n" + "="*60)
        print("训练成功完成！")
        print("="*60)
        print(f"\n模型已保存到: {model_path}")
        print("\n下一步：")
        print("  1. 重启Web服务器以加载新模型")
        print("  2. 在浏览器中打开: http://localhost:8000")
        print("  3. 上传未标注的视频进行检测测试")
    else:
        print("\n训练失败，请检查错误信息")


if __name__ == "__main__":
    main()

