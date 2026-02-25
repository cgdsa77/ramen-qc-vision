#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一所有classes.txt文件的类别顺序
确保所有子目录的classes.txt与主classes.txt一致
"""
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_classes_order():
    """
    统一所有classes.txt文件的类别顺序
    """
    base_dir = project_root / "data" / "labels" / "抻面"
    
    # 读取主classes.txt作为标准
    main_classes_file = base_dir / "classes.txt"
    if not main_classes_file.exists():
        print("错误：找不到主classes.txt文件")
        return
    
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        standard_classes = [line.strip() for line in f if line.strip()]
    
    print("="*60)
    print("统一classes.txt文件")
    print("="*60)
    print(f"\n标准类别顺序（来自主classes.txt）:")
    for i, cls in enumerate(standard_classes):
        print(f"  {i}: {cls}")
    
    # 获取所有视频目录
    video_dirs = [d.name for d in base_dir.iterdir() 
                 if d.is_dir() and d.name.startswith('cm')]
    video_dirs.sort()
    
    print(f"\n处理视频目录: {video_dirs}")
    print("="*60)
    
    total_modified = 0
    
    for video_name in video_dirs:
        video_dir = base_dir / video_name
        if not video_dir.exists():
            continue
        
        classes_file = video_dir / "classes.txt"
        if not classes_file.exists():
            print(f"\n[跳过] {video_name}: 没有classes.txt文件")
            continue
        
        # 读取子目录的classes.txt
        with open(classes_file, 'r', encoding='utf-8') as f:
            current_classes = [line.strip() for line in f if line.strip()]
        
        # 检查是否需要修改
        if current_classes == standard_classes:
            print(f"[一致] {video_name}: 类别顺序正确")
        else:
            # 写入标准类别
            with open(classes_file, 'w', encoding='utf-8') as f:
                for cls in standard_classes:
                    f.write(cls + '\n')
            print(f"[修改] {video_name}: 已更新为标准顺序")
            print(f"  原顺序: {current_classes}")
            print(f"  新顺序: {standard_classes}")
            total_modified += 1
    
    print("\n" + "="*60)
    print("统一完成！")
    print("="*60)
    print(f"共修改了 {total_modified} 个文件")
    
    if total_modified > 0:
        print("\n重要提示：")
        print("  由于类别顺序已更改，标注文件中的类别ID可能需要调整！")
        print("  请检查标注文件是否正确对应新的类别顺序")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="统一所有classes.txt文件的类别顺序")
    args = parser.parse_args()
    
    # 确认
    response = input("\n确认执行？这将统一所有classes.txt文件的顺序 (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        exit(0)
    
    fix_classes_order()

