#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
删除下面及捞面部分的旧标注文件（使用旧的4类别标注的文件）
由于类别已从4个改为3个，需要删除旧标注重新标注
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent

def clean_old_labels(stage_name: str = "下面及捞面"):
    """
    删除旧的标注文件（使用旧类别的标注）
    
    Args:
        stage_name: 阶段名称
    """
    labels_base = project_root / "data" / "labels" / stage_name
    
    if not labels_base.exists():
        print(f"错误：标注目录不存在: {labels_base}")
        return
    
    print("="*60)
    print("清理旧的标注文件")
    print("="*60)
    print("类别已从4个改为3个（移除了pot类别）")
    print("需要删除所有旧的标注文件，重新标注\n")
    
    video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir()])
    
    total_deleted = 0
    for video_dir in video_dirs:
        # 查找所有标注文件（排除classes.txt）
        label_files = [f for f in video_dir.glob("*.txt") if f.name != "classes.txt"]
        
        if label_files:
            print(f"  {video_dir.name}: 找到 {len(label_files)} 个标注文件")
            for label_file in label_files:
                label_file.unlink()
                total_deleted += 1
            print(f"    ✓ 已删除 {len(label_files)} 个文件")
    
    print("\n" + "="*60)
    print(f"完成！已删除 {total_deleted} 个旧标注文件")
    print("="*60)
    print("\n下一步：")
    print("1. 使用新的3类别定义重新标注")
    print("2. 新的类别顺序：0=noodle_rope, 1=hand, 2=tools_noodle")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='清理旧的标注文件')
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='确认删除（如果不加此参数，只显示将要删除的文件）'
    )
    
    args = parser.parse_args()
    
    if args.confirm:
        clean_old_labels()
    else:
        print("注意：此脚本将删除所有旧的标注文件")
        print("如果确定要删除，请运行: python scripts/clean_old_boiling_scooping_labels.py --confirm")
        print("\n当前状态下将要删除的文件：")
        labels_base = project_root / "data" / "labels" / "下面及捞面"
        if labels_base.exists():
            video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir()])
            for video_dir in video_dirs:
                label_files = [f for f in video_dir.glob("*.txt") if f.name != "classes.txt"]
                if label_files:
                    print(f"  {video_dir.name}: {len(label_files)} 个文件")

