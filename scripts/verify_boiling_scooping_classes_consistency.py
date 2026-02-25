#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证下面及捞面所有labels里面对应的classes.txt是否与主文件一致
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding='utf-8')

def verify_classes_consistency():
    """验证所有视频目录下的classes.txt是否与主classes.txt一致"""
    labels_base = project_root / "data" / "labels" / "下面及捞面"
    
    if not labels_base.exists():
        print(f"错误：标注目录不存在: {labels_base}")
        return False
    
    main_classes_file = labels_base / "classes.txt"
    if not main_classes_file.exists():
        print(f"错误：主classes.txt文件不存在: {main_classes_file}")
        return False
    
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        main_classes = [line.strip() for line in f if line.strip()]
    
    print("=" * 60)
    print("验证下面及捞面 classes.txt 一致性")
    print("=" * 60)
    print(f"\n主classes.txt顺序: {main_classes}")
    print(f"类别数量: {len(main_classes)}")
    print("\n检查各视频目录...")
    
    # 查找所有xl开头的目录
    video_dirs = sorted([d for d in labels_base.iterdir() 
                        if d.is_dir() and d.name.startswith('xl')])
    
    if not video_dirs:
        print("  [警告] 未找到任何xl开头的视频目录")
        return False
    
    all_consistent = True
    inconsistent_dirs = []
    
    for video_dir in video_dirs:
        video_classes_file = video_dir / "classes.txt"
        if not video_classes_file.exists():
            print(f"  [错误] {video_dir.name}: classes.txt 不存在")
            all_consistent = False
            inconsistent_dirs.append(video_dir.name)
            continue
        
        with open(video_classes_file, 'r', encoding='utf-8') as f:
            video_classes = [line.strip() for line in f if line.strip()]
        
        if video_classes != main_classes:
            print(f"  [错误] {video_dir.name}: classes.txt顺序不一致!")
            print(f"    主文件: {main_classes}")
            print(f"    视频文件: {video_classes}")
            all_consistent = False
            inconsistent_dirs.append(video_dir.name)
        else:
            print(f"  [OK] {video_dir.name}: classes.txt一致")
    
    print("\n" + "=" * 60)
    if all_consistent:
        print(f"✅ 所有视频目录的classes.txt都与主文件一致（共{len(video_dirs)}个）")
        return True
    else:
        print(f"❌ 发现 {len(inconsistent_dirs)} 个不一致的classes.txt文件:")
        for dir_name in inconsistent_dirs:
            print(f"  - {dir_name}")
        print("\n请先修复不一致的classes.txt文件后再进行训练！")
        return False

if __name__ == "__main__":
    success = verify_classes_consistency()
    sys.exit(0 if success else 1)

