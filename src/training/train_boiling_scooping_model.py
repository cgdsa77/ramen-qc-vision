"""
训练下面及捞面检测模型
基于标注数据训练YOLO检测模型
"""
import os
import sys

# 须在 import numpy/torch 之前：减轻 OpenBLAS “Memory allocation still failed”（Windows 常见）
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import shutil
import argparse
from pathlib import Path
import yaml

# PyTorch 2.6+ 默认 weights_only=True，旧版 ultralytics 加载 yolov8n.pt / 检查点会失败
try:
    import torch as _torch_for_load

    _torch_load_orig = _torch_for_load.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load_orig(*args, **kwargs)

    _torch_for_load.load = _torch_load_compat
except ImportError:
    pass

# NumPy 2.0+ 移除 np.trapz，旧版 ultralytics 在验证 mAP 时仍调用 np.trapz，会 AttributeError
try:
    import numpy as _np
    if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
        _np.trapz = _np.trapezoid
except Exception:
    pass

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


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
    
    video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith('xl')])
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


def prepare_yolo_dataset(standard_videos: list = None, val_split: float = 0.0):
    """
    准备YOLO格式的数据集
    
    Args:
        standard_videos: 标准视频列表，默认使用cm1, cm2, cm3
        val_split: 验证集比例（0.0-1.0），如果为0则不划分验证集
    """
    base_dir = project_root / "data"
    images_base = base_dir / "processed" / "下面及捞面"
    labels_base = base_dir / "labels" / "下面及捞面"
    
    if standard_videos is None:
        # 默认使用所有xl视频
        standard_videos = sorted([d.name for d in labels_base.iterdir() 
                                 if d.is_dir() and d.name.startswith('xl')])
    
    print(f"\n使用 {len(standard_videos)} 个视频进行训练: {standard_videos}")
    
    # 验证classes.txt一致性（关键步骤）
    if not verify_classes_consistency(labels_base):
        raise ValueError("classes.txt文件不一致，请先修复后再训练！")
    
    # 创建数据集目录
    dataset_dir = project_root / "datasets" / "boiling_scooping_detection"
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
                    batch: int = 8, lr0: float = 0.005, patience: int = 100, resume: bool = True,
                    device=None, workers: int = 2):
    """
    训练YOLO模型
    
    Args:
        dataset_yaml_path: 数据集配置文件路径
        epochs: 训练轮数（默认300，增加训练轮数以提升mAP50）
        imgsz: 图片尺寸
        batch: 批次大小（默认8，如果内存不够可以减小）
        lr0: 初始学习率（默认0.005，降低以适应更大的数据集）
        patience: 早停耐心值（默认100轮，如果100轮mAP50没有提升就停止）
        resume: 是否从检查点恢复训练（默认True）
    """
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("错误：未安装 ultralytics。请运行: pip install ultralytics")
        return None

    if device is None:
        train_device = 0 if torch.cuda.is_available() else "cpu"
    elif isinstance(device, str) and device.lower() in ("cuda", "gpu"):
        train_device = 0 if torch.cuda.is_available() else "cpu"
    else:
        train_device = device
    use_amp = train_device != "cpu" and torch.cuda.is_available()

    print(f"\n开始训练YOLO模型...")
    print(f"  - 设备: {train_device} (AMP: {use_amp})")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 图片尺寸: {imgsz}")
    print(f"  - 批次大小: {batch}")
    print(f"  - 初始学习率: {lr0}")
    print(f"  - 早停耐心值: {patience} (如果{patience}轮mAP50没有提升就提前停止)")
    print(f"  - DataLoader workers: {workers} (过大易占满内存并触发 OpenBLAS 报错，可改为 0～2)")

    # 检查是否有检查点可以恢复（优先使用最新的）
    # 检查 boiling_scooping_detection2（最新训练，Epoch 34）
    last_ckpt2 = project_root / "models" / "boiling_scooping_detection2" / "weights" / "last.pt"
    # 检查 boiling_scooping_detection（之前的训练，Epoch 30）
    last_ckpt1 = project_root / "models" / "boiling_scooping_detection" / "weights" / "last.pt"
    
    last_ckpt = None
    if resume:
        # 优先使用最新的检查点（boiling_scooping_detection2）
        if last_ckpt2.exists():
            last_ckpt = last_ckpt2
        elif last_ckpt1.exists():
            last_ckpt = last_ckpt1
    
    if last_ckpt and resume:
        print(f"  - 从检查点恢复: {last_ckpt}")
        try:
            model = YOLO(str(last_ckpt))
        except Exception as e:
            print(f"  [警告] 检查点无法加载（PyTorch/ultralytics 版本问题常见）: {e}")
            print(f"  - 改为从头训练: yolov8n.pt")
            model = YOLO("yolov8n.pt")
    else:
        print(f"  - 从头开始训练")
        model = YOLO('yolov8n.pt')
        print(f"  - 使用预训练模型: yolov8n.pt")
    
    # 训练模型
    # 优化数据增强参数（适合标准数据集训练，避免过度增强）
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,  # 早停：如果100轮mAP50没有提升就停止
        name='boiling_scooping_detection',
        project=str(project_root / "models"),
        save=True,
        plots=True,
        device=train_device,
        # 优化数据增强参数（适合标准数据集训练）
        hsv_h=0.01,    # 色调增强（降低以避免过度增强）
        hsv_s=0.5,     # 饱和度增强（降低）
        hsv_v=0.3,     # 明度增强（降低）
        degrees=10,    # 旋转角度（降低旋转范围，避免破坏面条形态）
        translate=0.1, # 平移（降低平移范围）
        scale=0.5,     # 缩放（降低缩放范围，保持物体大小一致性）
        flipud=0.0,    # 上下翻转（对于面条检测不适合）
        fliplr=0.5,    # 左右翻转（保持，有助于泛化）
        mosaic=0.5,    # Mosaic增强（降低，避免过度混合）
        mixup=0.1,     # MixUp增强（降低，保持数据真实性）
        copy_paste=0.0, # Copy-Paste增强（面条检测不适合）
        # 优化学习率策略（适合大数据集）
        lrf=0.1,       # 最终学习率因子（增加衰减，更稳定的收敛）
        momentum=0.937, # 动量
        weight_decay=0.0005, # 权重衰减
        warmup_epochs=5.0, # 预热轮数（增加预热轮数以稳定训练）
        amp=use_amp,   # GPU 时开启混合精度
        workers=workers,  # Windows 建议 0～2，默认 8 易与 OpenBLAS 争抢内存
    )
    
    # 保存最佳模型
    best_model_path = project_root / "models" / "boiling_scooping_detection" / "weights" / "best.pt"
    final_model_path = project_root / "models" / "boiling_scooping_detection_model.pt"
    
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        print(f"\n训练完成！")
        print(f"  - 最佳模型: {best_model_path}")
        print(f"  - 保存到: {final_model_path}")
        return final_model_path
    else:
        print("警告：未找到训练好的模型")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练下面及捞面检测模型')
    parser.add_argument('--videos', nargs='+', 
                       default=None,  # 默认使用所有xl视频
                       help='要使用的视频列表（默认：所有xl视频）')
    parser.add_argument('--epochs', type=int, default=300,
                       help='训练轮数（默认：300）')
    parser.add_argument('--batch', type=int, default=8,
                       help='批次大小（默认：8）')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图片尺寸（默认：640）')
    parser.add_argument('--lr0', type=float, default=0.005,
                       help='初始学习率（默认：0.005，降低以适应更大的数据集）')
    parser.add_argument('--patience', type=int, default=100,
                       help='早停耐心值（默认：100，如果100轮mAP50没有提升就提前停止）')
    parser.add_argument('--val-split', type=float, default=0.0,
                       help='验证集比例（默认：0.0，即所有数据作为训练集）')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='从检查点恢复训练（默认：True）')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='不从检查点恢复，从头开始训练')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备: cuda / cpu；默认自动（有 CUDA 则用 GPU）')
    parser.add_argument('--workers', type=int, default=2,
                       help='DataLoader 进程数（默认 2；内存不足或 OpenBLAS 报错时改为 0）')

    args = parser.parse_args()
    
    print("="*60)
    print("训练下面及捞面检测模型")
    print("="*60)
    
    # 如果没有指定视频列表，使用所有xl视频
    if args.videos is None:
        labels_base = project_root / "data" / "labels" / "下面及捞面"
        args.videos = sorted([d.name for d in labels_base.iterdir() 
                             if d.is_dir() and d.name.startswith('xl')])
    print(f"\n训练参数：")
    print(f"  - 视频列表: {args.videos}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 批次大小: {args.batch}")
    print(f"  - 图片尺寸: {args.imgsz}")
    print(f"  - 初始学习率: {args.lr0}")
    print(f"  - 早停耐心值: {args.patience}")
    print(f"  - 验证集比例: {args.val_split}")
    print(f"  - DataLoader workers: {args.workers}")
    print("="*60)
    
    # 1. 准备数据集
    dataset_dir, yaml_path, image_count = prepare_yolo_dataset(
        standard_videos=args.videos,
        val_split=args.val_split
    )
    
    if image_count == 0:
        print("错误：没有找到有效的训练数据")
        return
    
    # 2. 训练模型
    dev = args.device
    if dev is not None and dev.lower() == "cpu":
        dev = "cpu"
    model_path = train_yolo_model(
        yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        resume=args.resume,
        device=dev,
        workers=args.workers,
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

