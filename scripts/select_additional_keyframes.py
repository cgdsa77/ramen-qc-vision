#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为已评分视频选择额外的关键帧，提高数据多样性
重点选择包含noodle_bundle的帧，增加tightness和uniformity的样本数
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent

def load_yolo_labels(label_path: Path, class_names: List[str]) -> List[Dict]:
    """加载YOLO格式的标注文件"""
    annotations = []
    if not label_path.exists():
        return annotations
    
    try:
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
                    if cls_id < len(class_names):
                        annotations.append({
                            'class': class_names[cls_id],
                            'class_id': cls_id
                        })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        pass
    
    return annotations

def load_existing_keyframes(video_name: str) -> set:
    """加载已有关键帧列表"""
    keyframes_file = project_root / "data" / "scores" / "抻面" / video_name / "key_frames.json"
    existing_frames = set()
    
    if keyframes_file.exists():
        try:
            with open(keyframes_file, 'r', encoding='utf-8') as f:
                keyframes = json.load(f)
                for kf in keyframes:
                    if 'frame' in kf:
                        existing_frames.add(kf['frame'].replace('.jpg', ''))
        except Exception as e:
            print(f"[警告] 读取已有关键帧失败 {video_name}: {e}")
    
    # 也检查已评分的帧
    scores_dir = project_root / "data" / "scores" / "抻面" / video_name
    if scores_dir.exists():
        for score_file in scores_dir.glob("*_scores.json"):
            frame_name = score_file.stem.replace('_scores', '').replace('.jpg', '')
            existing_frames.add(frame_name)
    
    return existing_frames

def select_additional_keyframes(video_name: str, num_additional: int = 8, 
                                prioritize_noodle_bundle: bool = True) -> List[Dict[str, Any]]:
    """
    为视频选择额外的关键帧
    
    Args:
        video_name: 视频名称
        num_additional: 要选择的额外关键帧数量
        prioritize_noodle_bundle: 是否优先选择包含noodle_bundle的帧
    
    Returns:
        额外关键帧列表
    """
    labels_dir = project_root / "data" / "labels" / "抻面" / video_name
    
    if not labels_dir.exists():
        print(f"[错误] 视频标注目录不存在: {labels_dir}")
        return []
    
    # 读取类别名称
    classes_file = labels_dir / "classes.txt"
    if not classes_file.exists():
        classes_file = project_root / "data" / "labels" / "抻面" / "classes.txt"
    
    if not classes_file.exists():
        print(f"[错误] 找不到classes.txt文件")
        return []
    
    class_names = []
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # 获取已有关键帧
    existing_frames = load_existing_keyframes(video_name)
    
    # 获取所有标注文件
    label_files = sorted([f for f in labels_dir.glob("*.txt") 
                         if f.name != "classes.txt"],
                        key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    
    if not label_files:
        print(f"[警告] 视频 {video_name} 没有找到标注文件")
        return []
    
    total_frames = len(label_files)
    print(f"\n视频 {video_name}: 共 {total_frames} 帧，已有关键帧 {len(existing_frames)} 个")
    
    # 分析每帧的检测情况
    candidate_frames = []
    
    for label_file in label_files:
        frame_name = label_file.stem + ".jpg"
        frame_key = frame_name.replace('.jpg', '')
        
        # 跳过已有关键帧
        if frame_key in existing_frames:
            continue
        
        annotations = load_yolo_labels(label_file, class_names)
        
        # 统计检测到的关键类别
        has_hand = any(ann['class'] == 'hand' for ann in annotations)
        has_noodle_rope = any(ann['class'] == 'noodle_rope' for ann in annotations)
        has_noodle_bundle = any(ann['class'] == 'noodle_bundle' for ann in annotations)
        
        # 计算检测数量
        detection_count = len(annotations)
        
        # 计算优先级分数
        priority_score = 0
        if has_hand:
            priority_score += 1
        if has_noodle_rope:
            priority_score += 1
        if has_noodle_bundle:
            priority_score += 3  # 重点加分，因为样本数不足
        
        frame_index = int(label_file.stem.split('_')[-1]) if label_file.stem.split('_')[-1].isdigit() else 0
        
        candidate_frames.append({
            'frame': frame_name,
            'index': frame_index,
            'detections': detection_count,
            'priority': priority_score,
            'has_hand': has_hand,
            'has_noodle_rope': has_noodle_rope,
            'has_noodle_bundle': has_noodle_bundle
        })
    
    if not candidate_frames:
        print(f"  没有可用的候选帧（所有帧都已被选择）")
        return []
    
    # 选择策略：
    # 1. 如果优先选择noodle_bundle，先选择包含noodle_bundle的帧
    # 2. 均匀分布在整个视频中
    # 3. 优先选择优先级高的帧
    
    selected = []
    
    if prioritize_noodle_bundle:
        # 先选择包含noodle_bundle的帧
        bundle_frames = [f for f in candidate_frames if f['has_noodle_bundle']]
        if bundle_frames:
            # 按优先级和检测数排序
            bundle_frames.sort(key=lambda x: (-x['priority'], -x['detections']))
            # 选择前几个，但不超过总数的一半
            num_bundle = min(len(bundle_frames), num_additional // 2)
            selected.extend(bundle_frames[:num_bundle])
            print(f"  选择了 {num_bundle} 个包含noodle_bundle的帧")
    
    # 从剩余帧中选择，确保均匀分布
    remaining_frames = [f for f in candidate_frames if f['frame'] not in [s['frame'] for s in selected]]
    
    if remaining_frames:
        # 按索引排序
        remaining_frames.sort(key=lambda x: x['index'])
        
        # 计算需要选择的帧数
        num_needed = num_additional - len(selected)
        
        if num_needed > 0:
            # 均匀分布选择
            if len(remaining_frames) <= num_needed:
                selected.extend(remaining_frames)
            else:
                # 将剩余帧分成num_needed段，每段选择优先级最高的
                segment_size = len(remaining_frames) / num_needed
                for i in range(num_needed):
                    start_idx = int(i * segment_size)
                    end_idx = int((i + 1) * segment_size) if i < num_needed - 1 else len(remaining_frames)
                    
                    segment = remaining_frames[start_idx:end_idx]
                    if segment:
                        # 选择优先级最高的
                        best = max(segment, key=lambda x: (x['priority'], x['detections']))
                        selected.append(best)
    
    # 按索引排序
    selected.sort(key=lambda x: x['index'])
    
    # 格式化输出
    result = []
    for frame in selected:
        result.append({
            'frame': frame['frame'],
            'index': frame['index'],
            'detections': frame['detections'],
            'has_noodle_bundle': frame['has_noodle_bundle']
        })
    
    print(f"  选择了 {len(result)} 个额外关键帧")
    for idx, frame in enumerate(result, 1):
        bundle_mark = " [含noodle_bundle]" if frame['has_noodle_bundle'] else ""
        print(f"    {idx}. {frame['frame']} (索引: {frame['index']}, 检测数: {frame['detections']}){bundle_mark}")
    
    return result

def main():
    """主函数"""
    print("="*60)
    print("为已评分视频选择额外的关键帧")
    print("="*60)
    print("\n目标:")
    print("  1. 提高数据多样性")
    print("  2. 重点增加包含noodle_bundle的帧（提高tightness和uniformity样本数）")
    print("  3. 为cm1~cm3各选择8帧，cm4~cm7各选择5帧")
    print("  4. 总目标：从76帧增加到120帧左右")
    
    # 已评分视频
    scored_videos = {
        'cm1': 8,  # 再选8帧
        'cm2': 8,  # 再选8帧
        'cm3': 8,  # 再选8帧（cm3帧数多，可以多选）
        'cm4': 5,  # 再选5帧
        'cm5': 5,  # 再选5帧
        'cm6': 5,  # 再选5帧
        'cm7': 5   # 再选5帧（cm7帧数少，少选一些）
    }
    
    all_additional_keyframes = {}
    
    for video_name, num_frames in scored_videos.items():
        print(f"\n{'='*60}")
        additional_frames = select_additional_keyframes(
            video_name, 
            num_additional=num_frames,
            prioritize_noodle_bundle=True
        )
        
        if additional_frames:
            all_additional_keyframes[video_name] = additional_frames
            
            # 保存到文件
            scores_dir = project_root / "data" / "scores" / "抻面" / video_name
            scores_dir.mkdir(parents=True, exist_ok=True)
            
            additional_keyframes_file = scores_dir / "additional_key_frames.json"
            with open(additional_keyframes_file, 'w', encoding='utf-8') as f:
                json.dump(additional_frames, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 已保存到: {additional_keyframes_file}")
    
    # 生成汇总文件
    summary_file = project_root / "data" / "scores" / "抻面" / "额外关键帧列表.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("额外关键帧列表（用于提高数据多样性）\n")
        f.write("="*60 + "\n\n")
        f.write("说明：\n")
        f.write("  1. 这些关键帧用于补充数据多样性\n")
        f.write("  2. 优先选择包含noodle_bundle的帧\n")
        f.write("  3. 请使用评分工具为这些帧进行评分\n")
        f.write("  4. 评分完成后运行: python scripts/export_scoring_dataset.py\n")
        f.write("  5. 然后运行: python scripts/build_scoring_rules.py\n\n")
        
        total_additional = 0
        for video_name, frames in sorted(all_additional_keyframes.items()):
            f.write(f"\n视频: {video_name}\n")
            f.write("-" * 40 + "\n")
            bundle_count = sum(1 for frame in frames if frame.get('has_noodle_bundle', False))
            f.write(f"总帧数: {len(frames)}, 包含noodle_bundle: {bundle_count} 帧\n\n")
            
            for idx, frame in enumerate(frames, 1):
                bundle_mark = " [含noodle_bundle]" if frame.get('has_noodle_bundle', False) else ""
                f.write(f"{idx:2d}. {frame['frame']} (索引: {frame['index']:5d}, 检测数: {frame['detections']}){bundle_mark}\n")
            
            total_additional += len(frames)
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write(f"总计: {total_additional} 个额外关键帧\n")
        f.write("="*60 + "\n")
    
    print("\n" + "="*60)
    print("✅ 额外关键帧选择完成！")
    print("="*60)
    print(f"\n汇总文件: {summary_file}")
    print(f"总计选择了 {sum(len(frames) for frames in all_additional_keyframes.values())} 个额外关键帧")
    
    print("\n下一步:")
    print("1. 访问 http://localhost:8000/scoring-tool")
    print("2. 选择视频并查看额外关键帧（在评分工具中取消'仅显示关键帧'选项，或手动输入帧号）")
    print("3. 为每个额外关键帧进行评分")
    print("4. 评分完成后运行: python scripts/export_scoring_dataset.py")
    print("5. 然后运行: python scripts/build_scoring_rules.py 重新生成评分规则")

if __name__ == "__main__":
    main()
