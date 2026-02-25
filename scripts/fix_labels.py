#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复标注文件
将noodlerope（类别1）替换为hand（类别0）
"""
import os
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_labels_in_file(file_path: Path, old_class_id: int, new_class_id: int) -> tuple:
    """
    修复单个标注文件
    
    Returns:
        (修改的行数, 总行数)
    """
    if not file_path.exists():
        return (0, 0)
    
    lines = []
    modified_count = 0
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                total_count += 1
                current_class_id = int(parts[0])
                
                if current_class_id == old_class_id:
                    # 替换类别ID
                    parts[0] = str(new_class_id)
                    modified_count += 1
                    lines.append(' '.join(parts) + '\n')
                else:
                    lines.append(line + '\n')
            else:
                lines.append(line + '\n')
    
    # 写回文件
    if modified_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return (modified_count, total_count)


def fix_all_labels(video_dirs: list = None, old_class: str = 'noodlerope', new_class: str = 'hand'):
    """
    修复所有标注文件
    
    Args:
        video_dirs: 要修复的视频目录列表，如果为None则修复所有
        old_class: 旧类别名称（用于显示）
        new_class: 新类别名称（用于显示）
    """
    base_dir = project_root / "data" / "labels" / "抻面"
    
    # 读取类别文件
    classes_file = base_dir / "classes.txt"
    if not classes_file.exists():
        print("错误：找不到classes.txt文件")
        return
    
    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"类别列表: {class_names}")
    
    # 确定要替换的类别ID
    # 根据classes.txt: 0=hand, 1=noodle_rope
    # 用户说把hand标注成了noodlerope，所以noodlerope应该是类别1
    # 需要把类别1改为类别0（hand）
    
    old_class_id = 1  # noodle_rope的ID
    new_class_id = 0  # hand的ID
    
    print(f"\n将类别 {old_class_id} ({class_names[old_class_id] if old_class_id < len(class_names) else 'unknown'})")
    print(f"替换为类别 {new_class_id} ({class_names[new_class_id] if new_class_id < len(class_names) else 'unknown'})")
    print("="*60)
    
    # 获取要处理的视频目录
    if video_dirs is None:
        video_dirs = [d.name for d in base_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('cm')]
        video_dirs.sort()
    
    total_files = 0
    total_modified = 0
    total_annotations = 0
    total_changed = 0
    
    for video_name in video_dirs:
        video_dir = base_dir / video_name
        if not video_dir.exists():
            print(f"[跳过] {video_name}: 目录不存在")
            continue
        
        print(f"\n处理 {video_name}...")
        video_files = 0
        video_modified = 0
        video_annotations = 0
        video_changed = 0
        
        # 处理所有.txt文件（排除classes.txt）
        for label_file in video_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            
            modified, total = fix_labels_in_file(label_file, old_class_id, new_class_id)
            
            video_files += 1
            total_files += 1
            video_annotations += total
            total_annotations += total
            
            if modified > 0:
                video_modified += 1
                total_modified += 1
                video_changed += modified
                total_changed += modified
        
        if video_modified > 0:
            print(f"  [修改] {video_modified}/{video_files} 个文件，{video_changed} 个标注")
        else:
            print(f"  [无修改] {video_files} 个文件")
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"总计:")
    print(f"  - 处理文件数: {total_files}")
    print(f"  - 修改文件数: {total_modified}")
    print(f"  - 总标注数: {total_annotations}")
    print(f"  - 修改标注数: {total_changed}")
    print(f"\n已将 {old_class} (类别{old_class_id}) 替换为 {new_class} (类别{new_class_id})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量修复标注文件")
    parser.add_argument("--videos", type=str, nargs="+", default=None,
                       help="要修复的视频列表，例如: --videos cm1 cm2 cm3。如果不指定，则修复所有视频")
    parser.add_argument("--old", type=str, default="noodlerope",
                       help="旧类别名称（用于显示）")
    parser.add_argument("--new", type=str, default="hand",
                       help="新类别名称（用于显示）")
    
    args = parser.parse_args()
    
    print("="*60)
    print("批量修复标注文件")
    print("="*60)
    print(f"\n将 {args.old} 替换为 {args.new}")
    
    if args.videos:
        print(f"处理视频: {args.videos}")
    else:
        print("处理所有视频")
    
    # 确认
    response = input("\n确认执行？(y/n): ")
    if response.lower() != 'y':
        print("已取消")
        exit(0)
    
    fix_all_labels(args.videos, args.old, args.new)
    
    print("\n提示：修复完成后，建议重新训练模型")
    print("运行: python src/training/train_detection_model.py")

