#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复classes.txt文件
将noodlerope替换为noodle_rope
"""
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_classes_file(file_path: Path, old_name: str, new_name: str) -> bool:
    """
    修复单个classes.txt文件
    
    Returns:
        是否进行了修改
    """
    if not file_path.exists():
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        original_line = line
        # 替换noodlerope为noodle_rope
        if old_name in line:
            line = line.replace(old_name, new_name)
            modified = True
        new_lines.append(line)
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return modified


def fix_all_classes_files(video_dirs: list = None):
    """
    修复所有classes.txt文件
    """
    base_dir = project_root / "data" / "labels" / "抻面"
    
    # 获取要处理的视频目录
    if video_dirs is None:
        video_dirs = [d.name for d in base_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('cm')]
        video_dirs.sort()
    
    print("="*60)
    print("批量修复classes.txt文件")
    print("="*60)
    print(f"\n将 'noodlerope' 替换为 'noodle_rope'")
    print(f"\n处理视频目录: {video_dirs}")
    print("="*60)
    
    total_modified = 0
    
    # 1. 修复主classes.txt
    main_classes = base_dir / "classes.txt"
    if main_classes.exists():
        if fix_classes_file(main_classes, 'noodlerope', 'noodle_rope'):
            print(f"\n[修改] {main_classes}")
            total_modified += 1
        else:
            print(f"\n[无修改] {main_classes}")
    
    # 2. 修复各子目录的classes.txt
    for video_name in video_dirs:
        video_dir = base_dir / video_name
        if not video_dir.exists():
            print(f"\n[跳过] {video_name}: 目录不存在")
            continue
        
        classes_file = video_dir / "classes.txt"
        if classes_file.exists():
            if fix_classes_file(classes_file, 'noodlerope', 'noodle_rope'):
                print(f"[修改] {classes_file}")
                total_modified += 1
            else:
                print(f"[无修改] {classes_file}")
        else:
            print(f"[跳过] {video_name}: 没有classes.txt文件")
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"共修改了 {total_modified} 个文件")
    
    if total_modified > 0:
        print("\n提示：")
        print("  1. 请检查修复后的classes.txt文件是否正确")
        print("  2. 确保所有classes.txt文件中的类别顺序一致")
        print("  3. 建议重新训练模型")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量修复classes.txt文件")
    parser.add_argument("--videos", type=str, nargs="+", default=None,
                       help="要修复的视频列表，例如: --videos cm1 cm2 cm3。如果不指定，则修复所有视频")
    
    args = parser.parse_args()
    
    fix_all_classes_files(args.videos)

