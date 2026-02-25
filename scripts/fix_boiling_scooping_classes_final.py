#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复下面及捞面所有视频目录的classes.txt，使其与主文件一致
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding='utf-8')

def fix_all_classes_txt():
    """修复所有视频目录的classes.txt"""
    labels_base = project_root / "data" / "labels" / "下面及捞面"
    
    if not labels_base.exists():
        print(f"错误：标注目录不存在: {labels_base}")
        return False
    
    main_classes_file = labels_base / "classes.txt"
    if not main_classes_file.exists():
        print(f"错误：主classes.txt文件不存在: {main_classes_file}")
        return False
    
    # 读取主classes.txt
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        main_classes = [line.strip() for line in f if line.strip()]
    
    print("=" * 60)
    print("修复下面及捞面 classes.txt 文件")
    print("=" * 60)
    print(f"\n主classes.txt顺序: {main_classes}")
    print(f"类别数量: {len(main_classes)}")
    
    # 查找所有xl开头的目录
    video_dirs = sorted([d for d in labels_base.iterdir() 
                        if d.is_dir() and d.name.startswith('xl')])
    
    if not video_dirs:
        print("\n[警告] 未找到任何xl开头的视频目录")
        return False
    
    print(f"\n找到 {len(video_dirs)} 个视频目录，开始修复...")
    
    fixed_count = 0
    for video_dir in video_dirs:
        video_classes_file = video_dir / "classes.txt"
        
        # 写入正确的classes.txt
        with open(video_classes_file, 'w', encoding='utf-8') as f:
            for class_name in main_classes:
                f.write(f"{class_name}\n")
        
        print(f"  [OK] 已修复: {video_dir.name}")
        fixed_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"✅ 已修复 {fixed_count} 个视频目录的classes.txt文件")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = fix_all_classes_txt()
    if success:
        print("\n现在请运行验证脚本确认修复结果：")
        print("  python scripts/verify_boiling_scooping_classes_consistency.py")
    sys.exit(0 if success else 1)

