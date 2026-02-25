"""
从标准视频（cm1, cm2, cm3）提取骨架线统计信息
用于建立评分基准
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import math


def calculate_hand_distance(hand0: Dict, hand1: Dict) -> float:
    """计算双手距离"""
    kps0 = hand0.get('keypoints', [])
    kps1 = hand1.get('keypoints', [])
    if len(kps0) < 21 or len(kps1) < 21:
        return 0.0
    wrist0 = kps0[0]
    wrist1 = kps1[0]
    return math.sqrt(
        (wrist0.get('x', 0) - wrist1.get('x', 0))**2 +
        (wrist0.get('y', 0) - wrist1.get('y', 0))**2
    )


def calculate_palm_angle(kps: List[Dict]) -> float:
    """计算手掌角度"""
    if len(kps) < 21:
        return 0.0
    pinky_base = kps[17]
    index_base = kps[5]
    palm_dir_x = index_base.get('x', 0) - pinky_base.get('x', 0)
    palm_dir_y = index_base.get('y', 0) - pinky_base.get('y', 0)
    return math.atan2(palm_dir_y, palm_dir_x) * 180 / math.pi


def extract_features_from_frame(frame_data: Dict, prev_frame_data: Dict = None,
                                fps: float = 30.0) -> Dict[str, Any]:
    """从单帧提取特征"""
    features = {}
    hands = frame_data.get('hands', [])
    
    if len(hands) == 0:
        return features
    
    # 单 hand 特征
    if len(hands) >= 1:
        hand0 = hands[0]
        kps0 = hand0.get('keypoints', [])
        if len(kps0) >= 21:
            wrist = kps0[0]
            features['wrist_x'] = wrist.get('x', 0)
            features['wrist_y'] = wrist.get('y', 0)
            features['palm_angle'] = calculate_palm_angle(kps0)
    
    # 双手特征
    if len(hands) >= 2:
        hand_distance = calculate_hand_distance(hands[0], hands[1])
        features['hand_distance'] = hand_distance
        
        wrist0 = hands[0]['keypoints'][0]
        wrist1 = hands[1]['keypoints'][0]
        features['symmetry_y_diff'] = abs(wrist0.get('y', 0) - wrist1.get('y', 0))
        
        palm_angle0 = calculate_palm_angle(hands[0]['keypoints'])
        palm_angle1 = calculate_palm_angle(hands[1]['keypoints'])
        angle_diff = abs(palm_angle0 - palm_angle1)
        angle_diff = min(angle_diff, 360 - angle_diff)
        features['angle_consistency'] = 180 - angle_diff
    
    # 时序特征
    if prev_frame_data:
        prev_hands = prev_frame_data.get('hands', [])
        if len(prev_hands) > 0 and len(hands) > 0:
            prev_wrist = prev_hands[0]['keypoints'][0]
            curr_wrist = hands[0]['keypoints'][0]
            
            frame_interval = 1.0 / fps
            velocity_x = (curr_wrist.get('x', 0) - prev_wrist.get('x', 0)) / frame_interval
            velocity_y = (curr_wrist.get('y', 0) - prev_wrist.get('y', 0)) / frame_interval
            features['velocity_magnitude'] = math.sqrt(velocity_x**2 + velocity_y**2)
            
            if len(hands) >= 2 and len(prev_hands) >= 2:
                prev_distance = calculate_hand_distance(prev_hands[0], prev_hands[1])
                curr_distance = features.get('hand_distance', 0)
                if prev_distance > 0:
                    features['stretch_velocity'] = (curr_distance - prev_distance) / frame_interval
    
    return features


def extract_statistics_from_video(keypoints_file: Path) -> Dict[str, List[float]]:
    """从视频的骨架线数据提取统计信息"""
    with open(keypoints_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    fps = data.get('fps', 30.0)
    
    all_features = defaultdict(list)
    prev_frame_data = None
    
    for frame_data in frames:
        features = extract_features_from_frame(frame_data, prev_frame_data, fps)
        
        for key, value in features.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                all_features[key].append(value)
        
        prev_frame_data = frame_data
    
    return dict(all_features)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """计算统计信息"""
    if not values:
        return {}
    
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75)),
        'count': len(values)
    }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    hand_keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    
    # 标准视频列表
    standard_videos = ['cm1', 'cm2', 'cm3']
    
    print("=" * 60)
    print("提取标准骨架线统计信息")
    print("=" * 60)
    
    # 收集所有标准视频的特征值
    all_features = defaultdict(list)
    
    for video_name in standard_videos:
        keypoints_file = hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
        if not keypoints_file.exists():
            print(f"[警告] 未找到文件: {keypoints_file}")
            continue
        
        print(f"\n处理视频: {video_name}")
        video_features = extract_statistics_from_video(keypoints_file)
        
        # 合并特征值
        for key, values in video_features.items():
            all_features[key].extend(values)
        
        print(f"  提取到 {len(video_features)} 个特征维度")
    
    # 计算统计信息
    print("\n" + "=" * 60)
    print("计算统计信息")
    print("=" * 60)
    
    statistics = {}
    for key, values in all_features.items():
        stats = calculate_statistics(values)
        statistics[key] = stats
        print(f"\n{key}:")
        print(f"  均值: {stats['mean']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  范围: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  样本数: {stats['count']}")
    
    # 保存统计信息
    output_file = project_root / "data" / "scores" / "抻面" / "standard_skeleton_stats.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 统计信息已保存到: {output_file}")
    print(f"共提取 {len(statistics)} 个特征维度的统计信息")


if __name__ == '__main__':
    main()
