#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改下面及捞面部分的类别：增加到4个类别
noodle_rope, hand, tools_noodle, soup_noodle
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent

def fix_classes_txt(stage_name: str = "下面及捞面"):
    """
    修改所有classes.txt文件，增加到4个类别（新增soup_noodle）
    
    Args:
        stage_name: 阶段名称
    """
    labels_base = project_root / "data" / "labels" / stage_name
    
    if not labels_base.exists():
        print(f"错误：标注目录不存在: {labels_base}")
        return False
    
    # 新的类别列表（4个类别）
    new_classes = [
        "noodle_rope",
        "hand",
        "tools_noodle",
        "soup_noodle"
    ]
    
    new_classes_content = "\n".join(new_classes) + "\n"
    
    print("="*60)
    print("修改下面及捞面部分的类别定义")
    print("="*60)
    print(f"新类别（4个）:")
    for i, cls in enumerate(new_classes):
        print(f"  {i}: {cls}")
    print()
    
    # 修改主classes.txt
    main_classes_file = labels_base / "classes.txt"
    if main_classes_file.exists():
        with open(main_classes_file, 'w', encoding='utf-8') as f:
            f.write(new_classes_content)
        print(f"✓ 已更新主classes.txt: {main_classes_file}")
    else:
        print(f"警告：主classes.txt不存在: {main_classes_file}")
    
    # 修改所有视频子目录的classes.txt
    video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir()])
    updated_count = 0
    
    for video_dir in video_dirs:
        video_classes_file = video_dir / "classes.txt"
        if video_classes_file.exists():
            with open(video_classes_file, 'w', encoding='utf-8') as f:
                f.write(new_classes_content)
            updated_count += 1
            print(f"✓ 已更新: {video_dir.name}/classes.txt")
        else:
            print(f"  - {video_dir.name}: classes.txt不存在，跳过")
    
    print("\n" + "="*60)
    print(f"完成！已更新 {updated_count + 1} 个classes.txt文件")
    print("="*60)
    print("\n新的类别顺序：")
    for i, cls in enumerate(new_classes):
        print(f"  {i}: {cls}")
    print("\n重要提醒：")
    print("1. 请使用新的4类别定义重新标注")
    print("2. soup_noodle用于标注汤中的面条")
    print("3. noodle_rope用于标注其他情况下的面条（如正在捞出、已捞出等）")
    print("="*60)
    
    return True


if __name__ == '__main__':
    fix_classes_txt()

