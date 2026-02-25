#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评分辅助工具
提供抽样评分、模板评分等功能，大幅减少评分工作量
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent

def select_key_frames(video_name, sample_interval=10, min_detections=1):
    """
    选择关键帧进行评分
    
    Args:
        video_name: 视频名称
        sample_interval: 采样间隔（每N帧选1帧）
        min_detections: 最少检测数量（只选择有检测结果的帧）
    
    Returns:
        关键帧列表
    """
    images_dir = project_root / "data" / "processed" / "抻面" / video_name
    labels_dir = project_root / "data" / "labels" / "抻面" / video_name
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[错误] 视频目录不存在: {video_name}")
        return []
    
    # 获取所有帧
    frames = sorted([f.name for f in images_dir.glob("*.jpg")])
    
    key_frames = []
    for i, frame_name in enumerate(frames):
        # 采样：每sample_interval帧选1帧
        if i % sample_interval != 0:
            continue
        
        # 检查是否有检测结果
        label_file = labels_dir / frame_name.replace('.jpg', '.txt')
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
                if len(lines) >= min_detections:
                    key_frames.append({
                        'frame': frame_name,
                        'index': i,
                        'detections': len(lines)
                    })
    
    return key_frames

def create_scoring_template(video_name, frame_name, scores_data):
    """
    创建评分模板
    
    Args:
        video_name: 视频名称
        frame_name: 帧名称
        scores_data: 评分数据
    
    Returns:
        模板数据
    """
    images_dir = project_root / "data" / "processed" / "抻面" / video_name
    labels_dir = project_root / "data" / "labels" / "抻面" / video_name
    
    # 读取检测数据
    image_path = images_dir / frame_name
    label_path = labels_dir / frame_name.replace('.jpg', '.txt')
    
    template = {
        'video': video_name,
        'frame': frame_name,
        'scores': scores_data,
        'features': {}
    }
    
    # 提取特征（检测框数量、位置等）
    if image_path.exists() and label_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            
            detections = []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        detections.append({
                            'class_id': cls_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
            
            template['features'] = {
                'num_detections': len(detections),
                'detection_classes': [d['class_id'] for d in detections],
                'avg_width': np.mean([d['width'] for d in detections]) if detections else 0,
                'avg_height': np.mean([d['height'] for d in detections]) if detections else 0,
            }
    
    return template

def find_similar_frames(video_name, template, threshold=0.8):
    """
    找到与模板相似的帧
    
    Args:
        video_name: 视频名称
        template: 模板数据
        threshold: 相似度阈值
    
    Returns:
        相似帧列表
    """
    images_dir = project_root / "data" / "processed" / "抻面" / video_name
    labels_dir = project_root / "data" / "labels" / "抻面" / video_name
    
    if not images_dir.exists() or not labels_dir.exists():
        return []
    
    frames = sorted([f.name for f in images_dir.glob("*.jpg")])
    similar_frames = []
    
    template_features = template['features']
    
    for frame_name in frames:
        if frame_name == template['frame']:
            continue
        
        label_path = labels_dir / frame_name.replace('.jpg', '.txt')
        if not label_path.exists():
            continue
        
        # 提取当前帧特征
        detections = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    width = float(parts[3])
                    height = float(parts[4])
                    detections.append({
                        'class_id': cls_id,
                        'width': width,
                        'height': height
                    })
        
        if not detections:
            continue
        
        # 计算相似度
        current_features = {
            'num_detections': len(detections),
            'detection_classes': [d['class_id'] for d in detections],
            'avg_width': np.mean([d['width'] for d in detections]),
            'avg_height': np.mean([d['height'] for d in detections]),
        }
        
        # 简单相似度计算
        similarity = 0.0
        
        # 检测数量相似度
        if current_features['num_detections'] == template_features['num_detections']:
            similarity += 0.3
        elif abs(current_features['num_detections'] - template_features['num_detections']) <= 1:
            similarity += 0.2
        
        # 类别相似度
        if set(current_features['detection_classes']) == set(template_features['detection_classes']):
            similarity += 0.4
        
        # 尺寸相似度
        width_diff = abs(current_features['avg_width'] - template_features['avg_width'])
        height_diff = abs(current_features['avg_height'] - template_features['avg_height'])
        if width_diff < 0.1 and height_diff < 0.1:
            similarity += 0.3
        
        if similarity >= threshold:
            similar_frames.append({
                'frame': frame_name,
                'similarity': similarity
            })
    
    return sorted(similar_frames, key=lambda x: x['similarity'], reverse=True)

def apply_template_scores(video_name, template_frame, target_frames):
    """
    将模板评分应用到目标帧
    
    Args:
        video_name: 视频名称
        template_frame: 模板帧名称
        target_frames: 目标帧列表
    
    Returns:
        应用的评分数据
    """
    scores_dir = project_root / "data" / "scores" / "抻面" / video_name
    template_file = scores_dir / f"{template_frame.replace('.jpg', '')}_scores.json"
    
    if not template_file.exists():
        print(f"[错误] 模板评分文件不存在: {template_file}")
        return {}
    
    # 读取模板评分
    with open(template_file, 'r', encoding='utf-8') as f:
        template_data = json.load(f)
    
    applied_scores = {}
    for target_frame in target_frames:
        frame_name = target_frame['frame'] if isinstance(target_frame, dict) else target_frame
        applied_scores[frame_name] = {
            'video': video_name,
            'frame': frame_name,
            'scores': template_data.get('scores', {}),
            'source': 'template',
            'template_frame': template_frame,
            'similarity': target_frame.get('similarity', 0) if isinstance(target_frame, dict) else 0
        }
    
    return applied_scores

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="批量评分辅助工具")
    parser.add_argument("--action", choices=["select", "template", "apply"], 
                       default="select", help="操作类型")
    parser.add_argument("--video", required=True, help="视频名称")
    parser.add_argument("--interval", type=int, default=10, help="采样间隔（默认10）")
    parser.add_argument("--template-frame", help="模板帧名称")
    parser.add_argument("--threshold", type=float, default=0.8, help="相似度阈值")
    
    args = parser.parse_args()
    
    if args.action == "select":
        print("=" * 60)
        print(f"选择关键帧 - {args.video}")
        print("=" * 60)
        
        key_frames = select_key_frames(args.video, args.interval)
        print(f"\n找到 {len(key_frames)} 个关键帧（每{args.interval}帧采样1帧）")
        print("\n关键帧列表:")
        for i, frame in enumerate(key_frames[:20]):  # 只显示前20个
            print(f"  {i+1}. {frame['frame']} (检测数: {frame['detections']})")
        if len(key_frames) > 20:
            print(f"  ... 还有 {len(key_frames) - 20} 个帧")
        
        # 保存到文件
        output_file = project_root / "data" / "scores" / "抻面" / args.video / "key_frames.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(key_frames, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 关键帧列表已保存到: {output_file}")
        
    elif args.action == "template":
        if not args.template_frame:
            print("[错误] 需要指定模板帧 --template-frame")
            return
        
        print("=" * 60)
        print(f"创建评分模板 - {args.video}")
        print("=" * 60)
        
        # 读取模板评分
        scores_dir = project_root / "data" / "scores" / "抻面" / args.video
        template_file = scores_dir / f"{args.template_frame.replace('.jpg', '')}_scores.json"
        
        if not template_file.exists():
            print(f"[错误] 模板评分文件不存在: {template_file}")
            print("请先对该帧进行评分")
            return
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        template = create_scoring_template(args.video, args.template_frame, template_data.get('scores', {}))
        
        # 查找相似帧
        similar_frames = find_similar_frames(args.video, template, args.threshold)
        
        print(f"\n找到 {len(similar_frames)} 个相似帧（相似度 >= {args.threshold}）")
        print("\n相似帧列表:")
        for i, frame in enumerate(similar_frames[:20]):
            print(f"  {i+1}. {frame['frame']} (相似度: {frame['similarity']:.2f})")
        if len(similar_frames) > 20:
            print(f"  ... 还有 {len(similar_frames) - 20} 个帧")
        
        # 保存模板
        template_file = scores_dir / f"{args.template_frame.replace('.jpg', '')}_template.json"
        template['similar_frames'] = similar_frames
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 模板已保存到: {template_file}")
        
    elif args.action == "apply":
        if not args.template_frame:
            print("[错误] 需要指定模板帧 --template-frame")
            return
        
        print("=" * 60)
        print(f"应用模板评分 - {args.video}")
        print("=" * 60)
        
        # 读取模板
        scores_dir = project_root / "data" / "scores" / "抻面" / args.video
        template_file = scores_dir / f"{args.template_frame.replace('.jpg', '')}_template.json"
        
        if not template_file.exists():
            print(f"[错误] 模板文件不存在: {template_file}")
            print("请先运行 --action template 创建模板")
            return
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template = json.load(f)
        
        similar_frames = template.get('similar_frames', [])
        if not similar_frames:
            print("[错误] 模板中没有相似帧")
            return
        
        # 应用评分
        applied_scores = apply_template_scores(args.video, args.template_frame, similar_frames)
        
        # 保存应用的评分
        applied_count = 0
        for frame_name, score_data in applied_scores.items():
            score_file = scores_dir / f"{frame_name.replace('.jpg', '')}_scores.json"
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump(score_data, f, ensure_ascii=False, indent=2)
            applied_count += 1
        
        print(f"\n✅ 已将模板评分应用到 {applied_count} 个相似帧")

if __name__ == "__main__":
    main()

