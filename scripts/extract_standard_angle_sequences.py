"""
从标准视频提取角度序列并计算关键关节点权重
基于论文《基于深度学习的乒乓球姿态动作评分方法》
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.spatial_angle_extractor import SpatialAngleExtractor
from src.scoring.keypoint_weight_estimator import KeypointWeightEstimator


def main():
    """主函数"""
    print("=" * 60)
    print("提取标准角度序列并计算关键关节点权重")
    print("=" * 60)
    
    scores_dir = project_root / "data" / "scores" / "抻面"
    hand_keypoints_dir = scores_dir / "hand_keypoints"
    
    # 标准视频列表
    standard_videos = ['cm1', 'cm2', 'cm3']
    
    # 初始化组件
    angle_extractor = SpatialAngleExtractor()
    weight_estimator = KeypointWeightEstimator(bandwidth=0.5)
    
    # 提取所有标准视频的角度序列
    print("\n提取标准视频的角度序列...")
    angle_sequences = []
    
    for video_name in standard_videos:
        keypoints_file = hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            print(f"[警告] 未找到文件: {keypoints_file}")
            continue
        
        print(f"\n处理视频: {video_name}")
        
        with open(keypoints_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        frames_data = data.get('frames', [])
        print(f"  总帧数: {len(frames_data)}")
        
        # 提取角度序列
        angle_sequence = angle_extractor.extract_angle_sequence(frames_data)
        
        # 统计有效帧数
        valid_frames = sum(1 for frame_angles in angle_sequence 
                          if any(isinstance(v, dict) and len(v) > 0 
                                for v in frame_angles.values()))
        
        print(f"  有效角度帧数: {valid_frames}/{len(angle_sequence)}")
        
        if angle_sequence:
            angle_sequences.append(angle_sequence)
            print(f"  [OK] 成功提取角度序列")
    
    if len(angle_sequences) == 0:
        print("\n[错误] 未能提取任何角度序列")
        return
    
    print(f"\n共提取 {len(angle_sequences)} 个标准角度序列")
    
    # 保存角度序列
    output_file = scores_dir / "standard_angle_sequences.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 只保存关键信息（避免文件过大）
    sequences_summary = []
    for i, seq in enumerate(angle_sequences):
        summary = {
            'video': standard_videos[i] if i < len(standard_videos) else f'video_{i}',
            'total_frames': len(seq),
            'valid_frames': sum(1 for frame_angles in seq 
                               if any(isinstance(v, dict) and len(v) > 0 
                                     for v in frame_angles.values()))
        }
        sequences_summary.append(summary)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'standard_videos': standard_videos,
            'sequences_summary': sequences_summary,
            'note': '完整角度序列数据太大，仅保存摘要信息'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n角度序列摘要已保存: {output_file}")
    
    # 计算关键关节点权重
    print("\n" + "=" * 60)
    print("计算关键关节点权重（Mean Shift算法）")
    print("=" * 60)
    
    weights = weight_estimator.estimate_weights_from_standard_videos(angle_sequences)
    
    print(f"\n共识别 {len(weights)} 个关键关节点")
    print("\n关键关节点权重（前10个）：")
    
    # 按权重排序
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for i, (angle_name, weight) in enumerate(sorted_weights[:10], 1):
        print(f"  {i}. {angle_name}: {weight:.4f}")
    
    # 保存权重
    weights_file = scores_dir / "keypoint_weights.json"
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 关键关节点权重已保存: {weights_file}")
    
    # 计算角度变化统计信息
    print("\n" + "=" * 60)
    print("计算角度变化统计信息")
    print("=" * 60)
    
    all_angle_changes = {}
    for seq in angle_sequences:
        changes = angle_extractor.calculate_angle_changes(seq)
        for angle_name, change_values in changes.items():
            if angle_name not in all_angle_changes:
                all_angle_changes[angle_name] = []
            all_angle_changes[angle_name].extend(change_values)
    
    # 计算统计信息
    import numpy as np
    angle_statistics = {}
    for angle_name, changes in all_angle_changes.items():
        if changes:
            changes_array = np.array(changes)
            angle_statistics[angle_name] = {
                'mean': float(np.mean(changes_array)),
                'std': float(np.std(changes_array)),
                'min': float(np.min(changes_array)),
                'max': float(np.max(changes_array)),
                'count': len(changes)
            }
    
    # 保存统计信息
    stats_file = scores_dir / "angle_change_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(angle_statistics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 角度变化统计信息已保存: {stats_file}")
    print(f"共统计 {len(angle_statistics)} 个角度维度")


if __name__ == '__main__':
    main()
