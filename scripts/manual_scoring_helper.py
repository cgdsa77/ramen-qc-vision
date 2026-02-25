#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动评分辅助工具
用于批量查看评分数据、统计分析、导出标准数据集
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).parent.parent

def load_all_scores(stage="抻面"):
    """加载所有评分数据"""
    scores_dir = project_root / "data" / "scores" / stage
    all_scores = []
    
    if not scores_dir.exists():
        print(f"[警告] 评分目录不存在: {scores_dir}")
        return []
    
    for video_dir in scores_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        video_name = video_dir.name
        for score_file in video_dir.glob("*_scores.json"):
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
                    score_data['video'] = video_name
                    all_scores.append(score_data)
            except Exception as e:
                print(f"[警告] 读取评分文件失败 {score_file}: {e}")
    
    return all_scores

def analyze_scores(stage="抻面"):
    """分析评分数据"""
    scores = load_all_scores(stage)
    
    if not scores:
        print("没有找到评分数据")
        return
    
    print("=" * 60)
    print(f"评分数据分析 - {stage}")
    print("=" * 60)
    print(f"\n总评分帧数: {len(scores)}")
    
    # 统计各视频的评分数量
    video_counts = defaultdict(int)
    for score in scores:
        video_counts[score['video']] += 1
    
    print("\n各视频评分数量:")
    for video, count in sorted(video_counts.items()):
        print(f"  {video}: {count} 帧")
    
    # 统计各类别的评分分布
    class_scores = defaultdict(lambda: defaultdict(list))
    
    for score_data in scores:
        for det_key, det_scores in score_data.get('scores', {}).items():
            # 从检测项中推断类别（需要结合标注数据）
            # 这里简化处理，实际应该从标注文件中获取类别信息
            for attr, value in det_scores.items():
                if attr != 'notes' and isinstance(value, (int, float)):
                    class_scores[attr]['scores'].append(value)
    
    print("\n各属性评分统计:")
    for attr, data in class_scores.items():
        scores_list = data['scores']
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            print(f"  {attr}:")
            print(f"    平均分: {avg:.2f}")
            print(f"    评分数量: {len(scores_list)}")
            print(f"    分数分布: {dict(pd.Series(scores_list).value_counts().sort_index())}")

def export_standard_dataset(stage="抻面", output_file=None):
    """导出标准数据集"""
    scores = load_all_scores(stage)
    
    if not scores:
        print("没有找到评分数据，无法导出")
        return
    
    if output_file is None:
        output_file = project_root / "data" / "scores" / stage / "standard_dataset.json"
    
    # 整理为标准数据集格式
    dataset = {
        "stage": stage,
        "total_frames": len(scores),
        "videos": {},
        "statistics": {}
    }
    
    # 按视频分组，排除cm4的1号关键帧（cm4_00008）
    excluded_frames = {'cm4_00008'}  # cm4的1号关键帧，标注有问题，排除
    
    for score_data in scores:
        video = score_data['video']
        frame_name = score_data.get('frame', score_data.get('frame_file', ''))
        
        # 排除指定帧
        frame_key = frame_name.replace('.jpg', '').replace('_scores', '')
        if frame_key in excluded_frames:
            print(f"[跳过] 排除帧: {video}/{frame_name} (标注有问题)")
            continue
        
        if video not in dataset['videos']:
            dataset['videos'][video] = []
        
        dataset['videos'][video].append({
            "frame": frame_name,
            "scores": score_data['scores']
        })
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 标准数据集已导出到: {output_file}")
    print(f"   包含 {len(scores)} 帧评分数据")

def show_scoring_progress(stage="抻面"):
    """显示评分进度"""
    scores_dir = project_root / "data" / "scores" / stage
    labels_dir = project_root / "data" / "labels" / stage
    
    if not labels_dir.exists():
        print(f"[错误] 标注目录不存在: {labels_dir}")
        return
    
    print("=" * 60)
    print(f"评分进度统计 - {stage}")
    print("=" * 60)
    
    # 统计每个视频的总帧数和已评分帧数
    videos = []
    for video_dir in labels_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name.startswith('.'):
            continue
        
        video_name = video_dir.name
        total_frames = len(list(video_dir.glob("*.txt"))) - 1  # 减去classes.txt
        
        scored_frames = 0
        if scores_dir.exists():
            video_scores_dir = scores_dir / video_name
            if video_scores_dir.exists():
                scored_frames = len(list(video_scores_dir.glob("*_scores.json")))
        
        videos.append({
            "video": video_name,
            "total": total_frames,
            "scored": scored_frames,
            "progress": (scored_frames / total_frames * 100) if total_frames > 0 else 0
        })
    
    print("\n视频评分进度:")
    print(f"{'视频':<10} {'总帧数':<10} {'已评分':<10} {'进度':<10}")
    print("-" * 40)
    
    total_all = 0
    scored_all = 0
    
    for v in sorted(videos, key=lambda x: x['video']):
        print(f"{v['video']:<10} {v['total']:<10} {v['scored']:<10} {v['progress']:.1f}%")
        total_all += v['total']
        scored_all += v['scored']
    
    print("-" * 40)
    overall_progress = (scored_all / total_all * 100) if total_all > 0 else 0
    print(f"{'总计':<10} {total_all:<10} {scored_all:<10} {overall_progress:.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="手动评分辅助工具")
    parser.add_argument("--stage", default="抻面", help="阶段名称（默认：抻面）")
    parser.add_argument("--action", choices=["analyze", "export", "progress"], 
                       default="progress", help="操作类型")
    parser.add_argument("--output", help="导出文件路径")
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        analyze_scores(args.stage)
    elif args.action == "export":
        export_standard_dataset(args.stage, args.output)
    elif args.action == "progress":
        show_scoring_progress(args.stage)

if __name__ == "__main__":
    main()

