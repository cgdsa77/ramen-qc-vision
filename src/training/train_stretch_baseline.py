"""
训练抻面标准动作基准模型
从标注数据中学习标准动作模式
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.stretch_baseline import StretchBaselineModel


def load_yolo_labels(label_path: str, class_names: List[str]) -> List[Dict]:
    """加载YOLO格式的标注文件"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                if cls_id < len(class_names):
                    annotations.append({
                        'class': class_names[cls_id],
                        'class_id': cls_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
            except (ValueError, IndexError):
                continue
    return annotations


def load_video_annotations(video_name: str, images_dir: str, labels_dir: str, 
                          class_names: List[str]) -> Dict[str, Any]:
    """加载单个视频的所有标注数据"""
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 获取所有图片文件（按文件名排序）
    image_files = sorted([f for f in images_path.glob("*.jpg")], 
                        key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    
    frame_features = []
    
    for img_file in image_files:
        # 读取对应的标注文件
        label_file = labels_path / f"{img_file.stem}.txt"
        annotations = load_yolo_labels(str(label_file), class_names)
        
        # 提取特征
        model = StretchBaselineModel(class_names)
        features = model.extract_frame_features(annotations)
        frame_features.append(features)
    
    return {
        'video_name': video_name,
        'frame_features': frame_features,
        'total_frames': len(frame_features)
    }


def train_baseline_model(standard_videos: List[str] = None):
    """
    训练基准模型
    
    Args:
        standard_videos: 标准视频列表，如果为None则使用cm1, cm2, cm3
    """
    base_dir = project_root / "data"
    images_base = base_dir / "processed" / "抻面"
    labels_base = base_dir / "labels" / "抻面"
    
    # 读取类别名称
    classes_file = labels_base / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"找不到类别文件: {classes_file}")
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"类别: {class_names}")
    
    # 确定要训练的视频
    if standard_videos is None:
        standard_videos = ['cm1', 'cm2', 'cm3']
    
    print(f"\n使用标准视频进行训练: {standard_videos}")
    print("="*60)
    
    # 加载所有标准视频的标注数据
    video_data_list = []
    model = StretchBaselineModel(class_names)
    
    for video_name in standard_videos:
        images_dir = images_base / video_name
        labels_dir = labels_base / video_name
        
        if not images_dir.exists():
            print(f"[WARN] 跳过 {video_name}（图片目录不存在）")
            continue
        if not labels_dir.exists():
            print(f"[WARN] 跳过 {video_name}（标注目录不存在）")
            continue
        
        print(f"\n正在加载 {video_name} 的标注数据...")
        video_data = load_video_annotations(
            video_name, str(images_dir), str(labels_dir), class_names
        )
        video_data_list.append(video_data)
        print(f"  [OK] 加载了 {video_data['total_frames']} 帧")
    
    if not video_data_list:
        raise ValueError("没有找到有效的训练数据")
    
    # 训练模型
    print(f"\n开始训练基准模型（基于 {len(video_data_list)} 个标准视频）...")
    training_result = model.train_from_annotations(video_data_list)
    
    print(f"\n训练完成！")
    print(f"  - 训练视频数: {training_result['num_videos']}")
    print(f"  - 特征维度: {len(training_result['feature_keys'])}")
    
    # 保存模型
    model_dir = project_root / "models" / "stretch_baseline"
    model_path = model_dir / "baseline_model.json"
    model.save_model(str(model_path))
    print(f"\n模型已保存到: {model_path}")
    
    # 打印基准统计信息
    print("\n基准统计信息（部分关键特征）:")
    key_features = ['hand_presence_rate', 'noodle_rope_presence_rate', 
                    'hand_x_mean', 'hand_y_mean', 'hand_to_rope_distance_mean']
    for key in key_features:
        if key in training_result['baseline_stats']:
            stats = training_result['baseline_stats'][key]
            print(f"  - {key}:")
            print(f"    均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
    
    return model, training_result


if __name__ == "__main__":
    try:
        model, result = train_baseline_model()
        print("\n" + "="*60)
        print("训练成功完成！")
        print("="*60)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

