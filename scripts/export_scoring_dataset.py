#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出评分数据集（修复编码问题）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json

project_root = Path(__file__).parent.parent

def export_dataset():
    """导出标准数据集，排除cm4的1号关键帧"""
    scores_dir = project_root / "data" / "scores" / "抻面"
    output_file = scores_dir / "standard_dataset.json"
    
    # 排除的帧（标注有问题，未评分）
    excluded_frames = {
        'cm4_00008',      # cm4的原始1号关键帧（标注有问题）
        'cm1_00011',      # cm1的额外关键帧3号（标注有问题）
        'cm1_00012',      # cm1的额外关键帧4号（标注有问题）
        'cm4_00011'       # cm4的额外关键帧1号（标注有问题）
    }
    
    all_scores = []
    
    if not scores_dir.exists():
        print(f"[错误] 评分目录不存在: {scores_dir}")
        return
    
    print("=" * 60)
    print("导出标准数据集")
    print("=" * 60)
    
    # 遍历所有视频目录
    for video_dir in sorted(scores_dir.iterdir()):
        if not video_dir.is_dir() or not video_dir.name.startswith('cm'):
            continue
        
        video_name = video_dir.name
        video_scores = []
        
        # 读取该视频的所有评分文件
        for score_file in sorted(video_dir.glob("*_scores.json")):
            try:
                frame_name = score_file.stem.replace('_scores', '')
                frame_key = frame_name.replace('.jpg', '')
                
                # 排除指定帧
                if frame_key in excluded_frames:
                    print(f"[跳过] {video_name}/{frame_name} (标注有问题)")
                    continue
                
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
                
                video_scores.append({
                    "frame": frame_name + ".jpg" if not frame_name.endswith('.jpg') else frame_name,
                    "scores": score_data.get('scores', {})
                })
            except Exception as e:
                print(f"[警告] 读取评分文件失败 {score_file}: {e}")
        
        if video_scores:
            all_scores.append((video_name, video_scores))
            print(f"  {video_name}: {len(video_scores)} 帧")
    
    # 构建数据集
    dataset = {
        "stage": "抻面",
        "total_frames": sum(len(scores) for _, scores in all_scores),
        "videos": {video: scores for video, scores in all_scores},
        "excluded_frames": list(excluded_frames)
    }
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 标准数据集已导出到: {output_file}")
    print(f"   总帧数: {dataset['total_frames']}")
    print(f"   视频数: {len(dataset['videos'])}")
    print(f"   排除帧: {excluded_frames}")

if __name__ == "__main__":
    export_dataset()
