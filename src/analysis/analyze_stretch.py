"""
分析新视频的抻面动作
计算与标准动作的相似度得分
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.stretch_baseline import StretchBaselineModel
from src.training.train_stretch_baseline import load_yolo_labels, load_video_annotations


def analyze_video(video_name: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    分析单个视频的抻面动作
    
    Args:
        video_name: 视频名称（如 'cm4', 'cm5'）
        model_path: 模型路径，如果为None则使用默认路径
    
    Returns:
        分析结果字典
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
    
    # 加载模型
    if model_path is None:
        model_path = project_root / "models" / "stretch_baseline" / "baseline_model.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}。请先运行训练脚本。")
    
    model = StretchBaselineModel(class_names)
    model.load_model(str(model_path))
    
    # 加载测试视频的标注数据
    images_dir = images_base / video_name
    labels_dir = labels_base / video_name
    
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到标注目录: {labels_dir}。请先标注该视频。")
    
    print(f"正在分析视频: {video_name}")
    video_data = load_video_annotations(
        video_name, str(images_dir), str(labels_dir), class_names
    )
    
    # 提取序列特征
    sequence_features = model.extract_sequence_features(video_data['frame_features'])
    
    # 计算相似度得分
    similarity_result = model.compute_similarity_score(sequence_features)
    
    # 组合结果
    result = {
        'video_name': video_name,
        'total_frames': video_data['total_frames'],
        'sequence_features': {k: float(v) for k, v in sequence_features.items()},
        'similarity_score': similarity_result['final_score'],
        'component_scores': {
            k: {
                'value': float(v['value']),
                'baseline_mean': float(v['baseline_mean']),
                'score': float(v['score'])
            }
            for k, v in similarity_result['component_scores'].items()
        },
        'baseline_info': {
            'num_baseline_videos': similarity_result['baseline_comparison']['num_baseline_videos']
        }
    }
    
    return result


def analyze_video_from_path(video_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    从视频路径分析（如果视频已标注）
    
    Args:
        video_path: 视频路径或视频名称
        model_path: 模型路径
    """
    # 如果输入是路径，提取视频名称
    if os.path.isabs(video_path) or '/' in video_path or '\\' in video_path:
        video_name = Path(video_path).stem
    else:
        video_name = video_path
    
    return analyze_video(video_name, model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析抻面视频")
    parser.add_argument("--video", type=str, required=True, help="视频名称（如 cm4）")
    parser.add_argument("--model", type=str, default=None, help="模型路径（可选）")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    try:
        result = analyze_video(args.video, args.model)
        
        print("\n" + "="*60)
        print(f"分析结果: {result['video_name']}")
        print("="*60)
        print(f"\n总帧数: {result['total_frames']}")
        print(f"相似度得分: {result['similarity_score']:.3f}")
        print(f"\n基于 {result['baseline_info']['num_baseline_videos']} 个标准视频的基准模型")
        
        print("\n各项得分详情:")
        for feature_name, score_info in result['component_scores'].items():
            print(f"  • {feature_name}:")
            print(f"    当前值: {score_info['value']:.4f}")
            print(f"    基准值: {score_info['baseline_mean']:.4f}")
            print(f"    得分: {score_info['score']:.3f}")
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
        else:
            # 默认保存到reports目录
            output_dir = project_root / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"stretch_analysis_{result['video_name']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()




