#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整修复标注系统
1. 统一所有classes.txt文件的类别顺序
2. 修复标注文件中的类别ID映射
"""
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 标准类别顺序（与主classes.txt一致）
STANDARD_CLASSES = ['hand', 'noodle_rope', 'noodle_bundle', 'pot_or_table']


def get_class_mapping(old_classes: list, new_classes: list) -> dict:
    """
    获取类别ID映射关系
    
    Returns:
        {old_id: new_id} 的字典
    """
    mapping = {}
    for old_id, old_name in enumerate(old_classes):
        if old_name in new_classes:
            new_id = new_classes.index(old_name)
            if old_id != new_id:
                mapping[old_id] = new_id
    return mapping


def fix_label_file(file_path: Path, class_mapping: dict) -> tuple:
    """
    修复单个标注文件中的类别ID
    
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
            original_line = line
            line = line.strip()
            if not line:
                lines.append('\n')
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                total_count += 1
                old_class_id = int(parts[0])
                
                # 如果类别ID需要映射
                if old_class_id in class_mapping:
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    modified_count += 1
                    lines.append(' '.join(parts) + '\n')
                else:
                    lines.append(original_line)
            else:
                lines.append(original_line)
    
    # 写回文件
    if modified_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return (modified_count, total_count)


def fix_all():
    """
    完整修复所有标注文件
    """
    base_dir = project_root / "data" / "labels" / "抻面"
    
    # 读取主classes.txt
    main_classes_file = base_dir / "classes.txt"
    if not main_classes_file.exists():
        print("错误：找不到主classes.txt文件")
        return
    
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        main_classes = [line.strip() for line in f if line.strip()]
    
    # 确保主classes.txt是正确的
    if main_classes != STANDARD_CLASSES:
        print(f"更新主classes.txt为标准顺序...")
        with open(main_classes_file, 'w', encoding='utf-8') as f:
            for cls in STANDARD_CLASSES:
                f.write(cls + '\n')
        main_classes = STANDARD_CLASSES
    
    print("="*60)
    print("完整修复标注系统")
    print("="*60)
    print(f"\n标准类别顺序:")
    for i, cls in enumerate(STANDARD_CLASSES):
        print(f"  {i}: {cls}")
    
    # 获取所有视频目录
    video_dirs = [d.name for d in base_dir.iterdir() 
                 if d.is_dir() and d.name.startswith('cm')]
    video_dirs.sort()
    
    print(f"\n处理视频目录: {video_dirs}")
    print("="*60)
    
    total_files_modified = 0
    total_labels_modified = 0
    total_labels = 0
    
    for video_name in video_dirs:
        video_dir = base_dir / video_name
        if not video_dir.exists():
            continue
        
        classes_file = video_dir / "classes.txt"
        
        # 读取或创建子目录的classes.txt
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                old_classes = [line.strip() for line in f if line.strip()]
        else:
            old_classes = []
        
        # 统一classes.txt
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in STANDARD_CLASSES:
                f.write(cls + '\n')
        
        # 获取类别ID映射
        class_mapping = get_class_mapping(old_classes, STANDARD_CLASSES)
        
        if class_mapping:
            print(f"\n{video_name}:")
            print(f"  原类别顺序: {old_classes}")
            print(f"  新类别顺序: {STANDARD_CLASSES}")
            print(f"  需要映射的类别ID: {class_mapping}")
        else:
            print(f"\n{video_name}: 类别顺序已正确")
            continue
        
        # 修复所有标注文件
        video_files = 0
        video_labels = 0
        video_modified = 0
        
        for label_file in video_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            
            modified, total = fix_label_file(label_file, class_mapping)
            video_files += 1
            video_labels += total
            total_labels += total
            
            if modified > 0:
                video_modified += 1
                total_files_modified += 1
                video_modified += modified
                total_labels_modified += modified
        
        if video_modified > 0:
            print(f"  修复了 {video_modified}/{video_files} 个文件，{video_modified} 个标注")
        else:
            print(f"  无需修复: {video_files} 个文件")
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"总计:")
    print(f"  - 修改文件数: {total_files_modified}")
    print(f"  - 修改标注数: {total_labels_modified}/{total_labels}")
    print(f"\n所有classes.txt文件已统一为标准顺序")
    print(f"所有标注文件中的类别ID已正确映射")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="完整修复标注系统")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="自动确认，跳过交互提示")
    
    args = parser.parse_args()
    
    if not args.yes:
        print("\n此脚本将：")
        print("  1. 统一所有classes.txt文件的类别顺序")
        print("  2. 修复标注文件中的类别ID映射")
        print("  3. 将noodlerope改为noodle_rope（如果还有）")
        print("\n标准类别顺序: hand, noodle_rope, noodle_bundle, pot_or_table")
        
        try:
            response = input("\n确认执行？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                exit(0)
        except EOFError:
            print("\n非交互模式，自动执行...")
    
    fix_all()
    
    print("\n提示：修复完成后，建议重新训练模型")
    print("运行: python src/training/train_detection_model.py")

