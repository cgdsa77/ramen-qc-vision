#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试抻面分析功能
基于已标注的数据（cm1, cm2, cm3, cm4）进行测试
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scoring.stretch_eval import compute_motion_scores, score_stretch


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


def analyze_image_sequence(images_dir: str, labels_dir: str, class_names: List[str]) -> Dict[str, Any]:
    """分析图片序列"""
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 获取所有图片文件
    image_files = sorted([f for f in images_path.glob("*.jpg")])
    
    frames = []
    all_annotations = []
    frame_info = []
    
    print(f"正在加载 {len(image_files)} 张图片...")
    
    for img_file in image_files:
        # 读取图片
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue
        
        frames.append(frame)
        
        # 读取对应的标注文件
        label_file = labels_path / f"{img_file.stem}.txt"
        annotations = load_yolo_labels(str(label_file), class_names)
        all_annotations.append(annotations)
        
        frame_info.append({
            'filename': img_file.name,
            'annotations_count': len(annotations),
            'has_hand': any(ann['class'] == 'hand' for ann in annotations),
            'has_noodle_rope': any(ann['class'] == 'noodle_rope' for ann in annotations),
            'has_noodle_bundle': any(ann['class'] == 'noodle_bundle' for ann in annotations),
            'has_pot_or_table': any(ann['class'] == 'pot_or_table' for ann in annotations),
        })
    
    if not frames:
        return {"error": "没有找到有效图片"}
    
    # 计算运动得分
    motion_stats = compute_motion_scores(frames, sample_stride=2)
    
    # 计算检测存在率（检测到关键物体的帧数比例）
    frames_with_hand = sum(1 for info in frame_info if info['has_hand'])
    frames_with_noodle = sum(1 for info in frame_info if info['has_noodle_rope'] or info['has_noodle_bundle'])
    hand_presence = frames_with_hand / len(frame_info) if frame_info else 0
    noodle_presence = frames_with_noodle / len(frame_info) if frame_info else 0
    detection_presence = (hand_presence + noodle_presence) / 2
    
    # 计算抻面得分
    stretch_result = score_stretch(
        det_presence=detection_presence,
        motion_mean=motion_stats['motion_mean'],
        motion_std=motion_stats['motion_std']
    )
    
    # 统计标注信息
    total_annotations = sum(len(anns) for anns in all_annotations)
    class_counts = {}
    for annotations in all_annotations:
        for ann in annotations:
            cls = ann['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    return {
        'video_name': Path(images_dir).name,
        'total_frames': len(frames),
        'total_annotations': total_annotations,
        'class_distribution': class_counts,
        'hand_presence_rate': round(hand_presence, 3),
        'noodle_presence_rate': round(noodle_presence, 3),
        'motion_stats': motion_stats,
        'stretch_score': stretch_result,
        'frame_info': frame_info[:10],  # 只保留前10帧的详细信息
    }


def main():
    """主函数"""
    base_dir = project_root / "data"
    images_base = base_dir / "processed" / "抻面"
    labels_base = base_dir / "labels" / "抻面"
    
    # 读取类别名称
    classes_file = labels_base / "classes.txt"
    if not classes_file.exists():
        print(f"错误：找不到类别文件 {classes_file}")
        return
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"类别: {class_names}")
    print("\n" + "="*60)
    print("开始分析抻面数据（cm1, cm2, cm3, cm4）")
    print("="*60 + "\n")
    
    results = {}
    videos = ['cm1', 'cm2', 'cm3', 'cm4']
    
    for video_name in videos:
        images_dir = images_base / video_name
        labels_dir = labels_base / video_name
        
        if not images_dir.exists():
            print(f"警告：跳过 {video_name}（图片目录不存在）")
            continue
        if not labels_dir.exists():
            print(f"警告：跳过 {video_name}（标注目录不存在）")
            continue
        
        print(f"\n正在分析 {video_name}...")
        result = analyze_image_sequence(str(images_dir), str(labels_dir), class_names)
        
        if 'error' not in result:
            results[video_name] = result
            print(f"  ✓ 总帧数: {result['total_frames']}")
            print(f"  ✓ 总标注数: {result['total_annotations']}")
            print(f"  ✓ 类别分布: {result['class_distribution']}")
            print(f"  ✓ 手部检测率: {result['hand_presence_rate']:.2%}")
            print(f"  ✓ 面条检测率: {result['noodle_presence_rate']:.2%}")
            print(f"  ✓ 运动均值: {result['motion_stats']['motion_mean']:.2f}")
            print(f"  ✓ 运动标准差: {result['motion_stats']['motion_std']:.2f}")
            print(f"  ✓ 抻面得分: {result['stretch_score']['stretch_score']:.3f}")
        else:
            print(f"  ✗ 错误: {result['error']}")
    
    # 汇总结果
    print("\n" + "="*60)
    print("汇总结果")
    print("="*60)
    
    if results:
        avg_score = np.mean([r['stretch_score']['stretch_score'] for r in results.values()])
        total_frames = sum(r['total_frames'] for r in results.values())
        total_annotations = sum(r['total_annotations'] for r in results.values())
        
        print(f"\n总共分析了 {len(results)} 个视频:")
        print(f"  • 总帧数: {total_frames}")
        print(f"  • 总标注数: {total_annotations}")
        print(f"  • 平均抻面得分: {avg_score:.3f}")
        
        print("\n各视频得分详情:")
        for video_name, result in results.items():
            score = result['stretch_score']['stretch_score']
            frames = result['total_frames']
            annotations = result['total_annotations']
            print(f"  • {video_name}: {score:.3f} (帧数: {frames}, 标注: {annotations})")
        
        # 保存结果到JSON
        output_file = project_root / "reports" / "stretch_analysis_test.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_videos': len(results),
                    'total_frames': total_frames,
                    'total_annotations': total_annotations,
                    'average_score': round(avg_score, 3)
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    else:
        print("没有成功分析任何视频")


if __name__ == "__main__":
    main()




