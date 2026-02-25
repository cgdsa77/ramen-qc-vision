"""
分析低分视频的原因
检查骨架线检测质量、标注数据情况、评分逻辑
"""
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent

# 需要分析的视频
target_videos = ['cm6', 'cm9', 'cm10', 'cm11', 'cm12']

print("=" * 80)
print("低分视频原因分析")
print("=" * 80)

# 1. 检查骨架线检测质量
print("\n## 一、骨架线检测质量分析")
print("-" * 80)

hand_keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"

for video_name in target_videos:
    keypoints_file = hand_keypoints_dir / f"hand_keypoints_{video_name}.json"
    if not keypoints_file.exists():
        print(f"\n{video_name}: ❌ 骨架线数据文件不存在")
        continue
    
    with open(keypoints_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_frames = data.get('total_frames', 0)
    detected_frames = data.get('detected_frames', 0)
    missing_frames = data.get('missing_frames', 0)
    
    frames = data.get('frames', [])
    
    # 统计有效检测帧（至少一只手）
    valid_frames = 0
    both_hands_frames = 0
    single_hand_frames = 0
    
    for frame in frames:
        hands = frame.get('hands', [])
        if hands:
            valid_frames += 1
            if len(hands) >= 2:
                both_hands_frames += 1
            else:
                single_hand_frames += 1
    
    detection_rate = (valid_frames / total_frames * 100) if total_frames > 0 else 0
    both_hands_rate = (both_hands_frames / total_frames * 100) if total_frames > 0 else 0
    
    print(f"\n{video_name}:")
    print(f"  总帧数: {total_frames}")
    print(f"  检测到骨架: {valid_frames} 帧 ({detection_rate:.1f}%)")
    print(f"  缺失骨架: {missing_frames} 帧 ({100-detection_rate:.1f}%)")
    print(f"  双手检测: {both_hands_frames} 帧 ({both_hands_rate:.1f}%)")
    print(f"  单手检测: {single_hand_frames} 帧")

# 2. 检查标注数据情况
print("\n## 二、标注数据情况分析")
print("-" * 80)

scores_dir = project_root / "data" / "scores" / "抻面"

for video_name in target_videos:
    video_dir = scores_dir / video_name
    if not video_dir.exists():
        print(f"\n{video_name}: [ERROR] 标注文件夹不存在")
        continue
    
    # 统计标注文件
    score_files = list(video_dir.glob("*_scores.json"))
    
    # 统计各类别的标注数量
    hand_count = 0
    rope_count = 0
    bundle_count = 0
    
    for score_file in score_files:
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = data.get('scores', {})
            for detection_id, detection_scores in scores.items():
                if 'position' in detection_scores or 'action' in detection_scores:
                    hand_count += 1
                elif 'thickness' in detection_scores or 'elasticity' in detection_scores:
                    rope_count += 1
                elif 'tightness' in detection_scores or 'uniformity' in detection_scores:
                    bundle_count += 1
        except Exception as e:
            continue
    
    print(f"\n{video_name}:")
    print(f"  标注文件数: {len(score_files)}")
    print(f"  手部标注: {hand_count} 个检测框")
    print(f"  面条绳标注: {rope_count} 个检测框")
    print(f"  面条束标注: {bundle_count} 个检测框")

# 3. 检查评分结果中的问题
print("\n## 三、评分结果分析")
print("-" * 80)

reports_dir = project_root / "reports" / "comprehensive_scores_final"

for video_name in target_videos:
    detail_file = reports_dir / f"comprehensive_score_{video_name}.json"
    if not detail_file.exists():
        print(f"\n{video_name}: ❌ 评分结果文件不存在")
        continue
    
    with open(detail_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hand_score = data.get('hand_score', 0)
    rope_score = data.get('noodle_rope_score', 0)
    bundle_score = data.get('noodle_bundle_score', 0)
    total_score = data.get('total_score', 0)
    
    # 检查frame_scores中的问题
    frame_scores = data.get('frame_scores', [])
    
    # 统计手部属性得分
    hand_attrs = defaultdict(list)
    rope_attrs = defaultdict(list)
    
    for frame_score in frame_scores:
        hand_data = frame_score.get('hand', {})
        rope_data = frame_score.get('noodle_rope', {})
        
        if isinstance(hand_data, dict):
            for attr, value in hand_data.items():
                hand_attrs[attr].append(value)
        
        if isinstance(rope_data, dict):
            for attr, value in rope_data.items():
                rope_attrs[attr].append(value)
    
    print(f"\n{video_name}:")
    print(f"  总分: {total_score:.2f}")
    print(f"  手部得分: {hand_score:.2f}")
    print(f"  面条得分: {rope_score:.2f}")
    print(f"  面条束得分: {bundle_score:.2f}")
    
    if hand_attrs:
        print(f"\n  手部属性平均分:")
        for attr, values in hand_attrs.items():
            avg = sum(values) / len(values) if values else 0
            print(f"    {attr}: {avg:.2f} (共{len(values)}帧)")
    
    if rope_attrs:
        print(f"\n  面条属性平均分:")
        for attr, values in rope_attrs.items():
            avg = sum(values) / len(values) if values else 0
            print(f"    {attr}: {avg:.2f} (共{len(values)}帧)")
    
    # 检查是否有默认值2.4
    if abs(rope_score - 2.4) < 0.01:
        print(f"  ⚠️  面条得分是默认值2.4，可能缺少标注数据")

# 4. 检查评分逻辑中的默认值
print("\n## 四、评分逻辑检查")
print("-" * 80)

print("\n检查评分代码中是否有默认值或推断值...")
print("（需要查看enhanced_comprehensive_scorer.py中的_calculate_noodle_rope_scores方法）")
