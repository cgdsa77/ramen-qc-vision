"""
检查哪些视频有面条束（noodle_bundle）的标注评分
"""
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
scores_dir = project_root / "data" / "scores" / "抻面"

videos_with_bundle = set()
videos_without_bundle = set()
bundle_frame_counts = defaultdict(int)

# 检查每个视频
for video_dir in scores_dir.iterdir():
    if not video_dir.is_dir() or not video_dir.name.startswith('cm'):
        continue
    
    video_name = video_dir.name
    has_bundle = False
    bundle_frame_count = 0
    
    # 检查该视频的所有评分文件
    for score_file in video_dir.glob("*_scores.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores = data.get('scores', {})
            
            # 检查是否有noodle_bundle的评分（tightness或uniformity）
            for detection_id, detection_scores in scores.items():
                if 'tightness' in detection_scores or 'uniformity' in detection_scores:
                    has_bundle = True
                    bundle_frame_count += 1
                    break
        except Exception:
            continue
    
    if has_bundle:
        videos_with_bundle.add(video_name)
        bundle_frame_counts[video_name] = bundle_frame_count
    else:
        videos_without_bundle.add(video_name)

print("=" * 60)
print("面条束标注检查结果")
print("=" * 60)
print(f"\n有面条束标注的视频 ({len(videos_with_bundle)}个):")
for video in sorted(videos_with_bundle):
    print(f"  - {video}: {bundle_frame_counts[video]} 个关键帧有标注")

print(f"\n无面条束标注的视频 ({len(videos_without_bundle)}个):")
for video in sorted(videos_without_bundle):
    print(f"  - {video}")

print("\n" + "=" * 60)
print("说明")
print("=" * 60)
print("面条束（noodle_bundle）评分完全依赖标注数据。")
print("如果视频没有noodle_bundle的标注评分，则得分为0。")
print("\n建议：")
print("1. 为缺少标注的视频增加noodle_bundle关键帧标注")
print("2. 或者调整评分权重，降低noodle_bundle的权重")
