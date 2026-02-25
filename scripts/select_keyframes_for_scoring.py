#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为剩余视频选择关键帧用于手动评分
仿照cm1~cm3的方式，为每个视频选择10个关键帧
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
        print(f"[警告] 读取标注文件失败 {label_path}: {e}")
    
    return annotations

def select_keyframes(video_name: str, num_frames: int = 10) -> List[Dict[str, Any]]:
    """
    为视频选择关键帧
    
    Args:
        video_name: 视频名称（如 cm4）
        num_frames: 要选择的关键帧数量（默认10）
    
    Returns:
        关键帧列表，每个元素包含 frame, index, detections
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
    
    # 获取所有标注文件
    label_files = sorted([f for f in labels_dir.glob("*.txt") 
                         if f.name != "classes.txt"],
                        key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    
    if not label_files:
        print(f"[警告] 视频 {video_name} 没有找到标注文件")
        return []
    
    total_frames = len(label_files)
    print(f"\n视频 {video_name}: 共 {total_frames} 帧")
    
    # 分析每帧的检测情况
    frame_info = []
    for label_file in label_files:
        annotations = load_yolo_labels(label_file, class_names)
        
        # 统计检测到的关键类别
        has_hand = any(ann['class'] == 'hand' for ann in annotations)
        has_noodle_rope = any(ann['class'] == 'noodle_rope' for ann in annotations)
        has_noodle_bundle = any(ann['class'] == 'noodle_bundle' for ann in annotations)
        
        # 计算检测数量（优先考虑关键类别）
        detection_count = len(annotations)
        priority_score = 0
        if has_hand:
            priority_score += 2
        if has_noodle_rope:
            priority_score += 2
        if has_noodle_bundle:
            priority_score += 1
        
        frame_name = label_file.stem + ".jpg"
        frame_index = int(label_file.stem.split('_')[-1]) if label_file.stem.split('_')[-1].isdigit() else 0
        
        frame_info.append({
            'frame': frame_name,
            'index': frame_index,
            'detections': detection_count,
            'priority': priority_score,
            'has_hand': has_hand,
            'has_noodle_rope': has_noodle_rope,
            'has_noodle_bundle': has_noodle_bundle
        })
    
    # 选择关键帧策略：
    # 1. 均匀分布在整个视频中
    # 2. 优先选择有检测框的帧（特别是hand和noodle_rope）
    
    if total_frames <= num_frames:
        # 如果总帧数少于等于需要的帧数，返回所有帧
        selected = [{'frame': info['frame'], 'index': info['index'], 'detections': info['detections']} 
                    for info in frame_info]
    else:
        # 将视频分成num_frames段，每段选择一帧
        segment_size = total_frames / num_frames
        selected = []
        
        for i in range(num_frames):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size) if i < num_frames - 1 else total_frames
            
            # 在当前段中选择优先级最高的帧
            segment_frames = frame_info[start_idx:end_idx]
            if segment_frames:
                # 优先选择有hand和noodle_rope的帧
                best_frame = max(segment_frames, key=lambda x: (x['priority'], x['detections']))
                selected.append({
                    'frame': best_frame['frame'],
                    'index': best_frame['index'],
                    'detections': best_frame['detections']
                })
    
    print(f"  选择了 {len(selected)} 个关键帧")
    for idx, frame in enumerate(selected, 1):
        print(f"    {idx}. {frame['frame']} (索引: {frame['index']}, 检测数: {frame['detections']})")
    
    return selected

def main():
    """主函数"""
    print("="*60)
    print("为剩余视频选择关键帧用于手动评分")
    print("="*60)
    
    # 已评分过的视频
    scored_videos = {'cm1', 'cm2', 'cm3'}
    
    # 获取所有已标注的视频
    labels_dir = project_root / "data" / "labels" / "抻面"
    all_videos = [d.name for d in labels_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('cm')]
    
    # 找出需要选择关键帧的视频
    videos_to_process = [v for v in sorted(all_videos) if v not in scored_videos]
    
    if not videos_to_process:
        print("\n所有视频都已经评分过了！")
        return
    
    print(f"\n需要处理的视频: {', '.join(videos_to_process)}")
    print(f"每个视频将选择 10 个关键帧")
    
    # 为每个视频选择关键帧
    all_keyframes = {}
    
    for video_name in videos_to_process:
        keyframes = select_keyframes(video_name, num_frames=10)
        if keyframes:
            all_keyframes[video_name] = keyframes
            
            # 保存到文件
            scores_dir = project_root / "data" / "scores" / "抻面" / video_name
            scores_dir.mkdir(parents=True, exist_ok=True)
            
            keyframes_file = scores_dir / "key_frames.json"
            with open(keyframes_file, 'w', encoding='utf-8') as f:
                json.dump(keyframes, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 已保存到: {keyframes_file}")
    
    # 生成汇总文件
    summary_file = project_root / "data" / "scores" / "抻面" / "关键帧列表_剩余视频.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("剩余视频关键帧列表\n")
        f.write("="*60 + "\n\n")
        
        for video_name, keyframes in sorted(all_keyframes.items()):
            f.write(f"\n视频: {video_name}\n")
            f.write("-" * 40 + "\n")
            for idx, frame in enumerate(keyframes, 1):
                f.write(f"{idx:2d}. {frame['frame']} (索引: {frame['index']:5d}, 检测数: {frame['detections']})\n")
            f.write("\n")
    
    print("\n" + "="*60)
    print("✅ 关键帧选择完成！")
    print("="*60)
    print(f"\n汇总文件: {summary_file}")
    print(f"\n下一步:")
    print("1. 访问 http://localhost:8000/scoring-tool")
    print("2. 选择视频并查看关键帧")
    print("3. 为每个关键帧进行手动评分")

if __name__ == "__main__":
    main()
