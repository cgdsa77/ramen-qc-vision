#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析评分数据质量，评估是否需要更多关键帧或更多视频
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent

def load_all_scores():
    """加载所有评分数据"""
    scores_dir = project_root / "data" / "scores" / "抻面"
    all_scores = []
    
    if not scores_dir.exists():
        return []
    
    for video_dir in scores_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name.startswith('.'):
            continue
        
        video_name = video_dir.name
        for score_file in video_dir.glob("*_scores.json"):
            try:
                with open(score_file, 'r', encoding='utf-8') as f:
                    score_data = json.load(f)
                    score_data['video'] = video_name
                    score_data['frame_file'] = score_file.stem
                    all_scores.append(score_data)
            except Exception as e:
                print(f"[警告] 读取评分文件失败 {score_file}: {e}")
    
    return all_scores

def analyze_coverage():
    """分析数据覆盖度"""
    scores = load_all_scores()
    
    print("=" * 60)
    print("评分数据覆盖度分析")
    print("=" * 60)
    
    # 按视频统计
    video_stats = defaultdict(lambda: {'count': 0, 'frames': []})
    for score in scores:
        video = score['video']
        video_stats[video]['count'] += 1
        video_stats[video]['frames'].append(score.get('frame', score.get('frame_file', '')))
    
    print("\n各视频评分帧数:")
    print("-" * 60)
    total_frames = 0
    for video in sorted(video_stats.keys()):
        count = video_stats[video]['count']
        total_frames += count
        print(f"  {video}: {count} 帧")
    
    print(f"\n总计: {total_frames} 帧")
    
    # 检查标注数据情况
    labels_dir = project_root / "data" / "labels" / "抻面"
    print("\n" + "=" * 60)
    print("标注数据情况")
    print("=" * 60)
    
    all_videos = []
    annotated_videos = []
    scored_videos = set(video_stats.keys())
    
    if labels_dir.exists():
        for video_dir in labels_dir.iterdir():
            if not video_dir.is_dir() or not video_dir.name.startswith('cm'):
                continue
            
            video_name = video_dir.name
            all_videos.append(video_name)
            
            # 统计标注文件数（排除classes.txt）
            txt_files = [f for f in video_dir.glob("*.txt") if f.name != "classes.txt"]
            if len(txt_files) > 0:
                annotated_videos.append((video_name, len(txt_files)))
    
    print("\n已标注视频:")
    for video_name, frame_count in sorted(annotated_videos):
        scored = "✓ 已评分" if video_name in scored_videos else "✗ 未评分"
        print(f"  {video_name}: {frame_count} 帧标注 {scored}")
    
    print("\n未标注视频:")
    unannotated = [v for v in sorted(all_videos) if v not in [name for name, _ in annotated_videos]]
    if unannotated:
        for video_name in unannotated:
            print(f"  {video_name}: 无标注数据")
    else:
        print("  无")
    
    return {
        'total_scored_frames': total_frames,
        'scored_videos': len(scored_videos),
        'annotated_videos': len(annotated_videos),
        'unannotated_videos': unannotated,
        'video_stats': dict(video_stats)
    }

def analyze_score_distribution():
    """分析评分分布"""
    scores = load_all_scores()
    
    print("\n" + "=" * 60)
    print("评分分布分析")
    print("=" * 60)
    
    # 收集所有评分
    attribute_scores = defaultdict(list)
    video_attribute_scores = defaultdict(lambda: defaultdict(list))
    
    for score_data in scores:
        video = score_data['video']
        scores_dict = score_data.get('scores', {})
        
        for det_key, det_scores in scores_dict.items():
            for attr, value in det_scores.items():
                if attr != 'notes' and isinstance(value, (int, float)):
                    attribute_scores[attr].append(value)
                    video_attribute_scores[video][attr].append(value)
    
    print("\n各属性评分统计:")
    print("-" * 60)
    for attr in sorted(attribute_scores.keys()):
        scores_list = attribute_scores[attr]
        if not scores_list:
            continue
        
        mean = np.mean(scores_list)
        std = np.std(scores_list)
        min_score = np.min(scores_list)
        max_score = np.max(scores_list)
        
        # 计算分布
        distribution = {i: scores_list.count(i) for i in range(1, 6)}
        
        print(f"\n{attr}:")
        print(f"  平均分: {mean:.2f}")
        print(f"  标准差: {std:.2f}")
        print(f"  范围: {min_score:.1f} ~ {max_score:.1f}")
        print(f"  样本数: {len(scores_list)}")
        print(f"  分布: {distribution}")
        
        # 检查数据质量
        if std < 0.5:
            print(f"  ⚠️  标准差较小，可能缺乏多样性")
        if len(scores_list) < 20:
            print(f"  ⚠️  样本数较少，建议增加评分帧数")
    
    # 检查视频间差异
    print("\n" + "=" * 60)
    print("视频间评分差异分析")
    print("=" * 60)
    
    for attr in sorted(attribute_scores.keys()):
        video_means = {}
        for video, scores_list in video_attribute_scores.items():
            if attr in scores_list and len(scores_list[attr]) > 0:
                video_means[video] = np.mean(scores_list[attr])
        
        if len(video_means) > 1:
            means_list = list(video_means.values())
            video_std = np.std(means_list)
            print(f"\n{attr}:")
            print(f"  视频间标准差: {video_std:.2f}")
            if video_std < 0.3:
                print(f"  ⚠️  视频间差异较小，可能缺乏多样性")
            for video, mean in sorted(video_means.items()):
                print(f"    {video}: {mean:.2f}")

def analyze_keyframe_coverage():
    """分析关键帧覆盖度"""
    scores_dir = project_root / "data" / "scores" / "抻面"
    
    print("\n" + "=" * 60)
    print("关键帧覆盖度分析")
    print("=" * 60)
    
    for video_dir in sorted(scores_dir.iterdir()):
        if not video_dir.is_dir() or not video_dir.name.startswith('cm'):
            continue
        
        video_name = video_dir.name
        keyframes_file = video_dir / "key_frames.json"
        
        if not keyframes_file.exists():
            continue
        
        # 读取关键帧列表
        with open(keyframes_file, 'r', encoding='utf-8') as f:
            keyframes = json.load(f)
        
        # 统计已评分的帧
        scored_frames = set()
        for score_file in video_dir.glob("*_scores.json"):
            frame_name = score_file.stem.replace('_scores', '')
            scored_frames.add(frame_name)
        
        total_keyframes = len(keyframes)
        scored_keyframes = len(scored_frames)
        coverage = (scored_keyframes / total_keyframes * 100) if total_keyframes > 0 else 0
        
        print(f"\n{video_name}:")
        print(f"  关键帧总数: {total_keyframes}")
        print(f"  已评分: {scored_keyframes}")
        print(f"  覆盖率: {coverage:.1f}%")
        
        if coverage < 100:
            missing = [kf['frame'] for kf in keyframes if kf['frame'].replace('.jpg', '') not in scored_frames]
            print(f"  ⚠️  未评分帧: {len(missing)} 个")
            if len(missing) <= 3:
                print(f"     {', '.join(missing)}")

def generate_recommendations():
    """生成建议"""
    coverage_info = analyze_coverage()
    
    print("\n" + "=" * 60)
    print("建议与评估")
    print("=" * 60)
    
    total_frames = coverage_info['total_scored_frames']
    scored_videos = coverage_info['scored_videos']
    annotated_videos = coverage_info['annotated_videos']
    unannotated_videos = coverage_info['unannotated_videos']
    
    print("\n1. 关于cm8~cm12:")
    if unannotated_videos:
        print(f"   - 发现 {len(unannotated_videos)} 个未标注视频: {', '.join(unannotated_videos)}")
        print(f"   - 建议: 如果这些视频质量较好，建议标注并评分，可以增加数据多样性")
        print(f"   - 优先级: {'高' if scored_videos < 5 else '中'}（当前已评分{scored_videos}个视频）")
    else:
        print("   - 所有视频都已标注")
    
    print("\n2. 关于增加关键帧:")
    if total_frames < 50:
        print(f"   - 当前总评分帧数: {total_frames} 帧")
        print(f"   - 建议: 为已评分视频增加更多关键帧，目标至少50-80帧")
        print(f"   - 优先级: 高")
    elif total_frames < 100:
        print(f"   - 当前总评分帧数: {total_frames} 帧")
        print(f"   - 建议: 可以考虑为部分视频增加关键帧，提高数据覆盖度")
        print(f"   - 优先级: 中")
    else:
        print(f"   - 当前总评分帧数: {total_frames} 帧")
        print(f"   - 数据量充足，可以开始验证模型效果")
        print(f"   - 优先级: 低")
    
    print("\n3. 关于数据质量:")
    if scored_videos < 5:
        print(f"   - 当前已评分视频数: {scored_videos}")
        print(f"   - 建议: 至少评分5-7个视频，确保数据多样性")
        print(f"   - 优先级: 高")
    else:
        print(f"   - 当前已评分视频数: {scored_videos}")
        print(f"   - 视频数量充足")
    
    print("\n4. 下一步行动:")
    actions = []
    
    if unannotated_videos and scored_videos < 7:
        actions.append(f"标注并评分 {', '.join(unannotated_videos[:2])} 等视频")
    
    if total_frames < 60:
        actions.append("为cm4~cm7增加更多关键帧（每个视频再选5-10帧）")
    
    if not actions:
        actions.append("重新生成评分规则: python scripts/build_scoring_rules.py")
        actions.append("测试自动评分效果，验证模型准确性")
    
    for i, action in enumerate(actions, 1):
        print(f"   {i}. {action}")

def main():
    print("=" * 60)
    print("评分数据质量分析")
    print("=" * 60)
    
    # 分析覆盖度
    coverage_info = analyze_coverage()
    
    # 分析评分分布
    analyze_score_distribution()
    
    # 分析关键帧覆盖度
    analyze_keyframe_coverage()
    
    # 生成建议
    generate_recommendations()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
