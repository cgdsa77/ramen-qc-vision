#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全修复下面及捞面classes.txt顺序不一致问题
在统一classes.txt之前，先转换标注文件中的类别ID，确保标注正确
"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent
labels_base = project_root / "data" / "labels" / "下面及捞面"
main_classes_file = labels_base / "classes.txt"

def read_classes(classes_file: Path) -> list:
    """读取classes.txt文件"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def create_id_mapping(old_classes: list, new_classes: list) -> dict:
    """
    创建类别ID映射关系
    
    Args:
        old_classes: 旧的类别顺序列表
        new_classes: 新的类别顺序列表
    
    Returns:
        dict: {old_id: new_id} 映射字典
    """
    mapping = {}
    for old_id, old_class in enumerate(old_classes):
        if old_class in new_classes:
            new_id = new_classes.index(old_class)
            mapping[old_id] = new_id
        else:
            print(f"警告：类别 '{old_class}' 在新类别列表中不存在")
    return mapping

def convert_label_file(label_file: Path, id_mapping: dict) -> bool:
    """
    转换标注文件中的类别ID
    
    Args:
        label_file: 标注文件路径
        id_mapping: ID映射字典 {old_id: new_id}
    
    Returns:
        bool: 是否成功转换
    """
    try:
        # 读取原文件内容
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 转换每一行
        new_lines = []
        converted = False
        for line in lines:
            line = line.strip()
            if not line:
                new_lines.append('')
                continue
            
            parts = line.split()
            if len(parts) < 5:
                new_lines.append(line)
                continue
            
            try:
                old_id = int(parts[0])
                if old_id in id_mapping:
                    new_id = id_mapping[old_id]
                    if old_id != new_id:
                        converted = True
                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                else:
                    print(f"警告：标注文件 {label_file.name} 中的类别ID {old_id} 无法映射")
                    new_lines.append(line + '\n')
            except ValueError:
                new_lines.append(line + '\n')
        
        # 写回文件
        if converted or True:  # 即使没有转换也写回，确保格式正确
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        
        return converted
    except Exception as e:
        print(f"错误：转换标注文件 {label_file.name} 时出错: {e}")
        return False

def fix_video_classes(video_dir: Path, main_classes: list) -> tuple:
    """
    修复单个视频的classes.txt和标注文件
    
    Returns:
        tuple: (fixed_classes, converted_labels_count, total_labels_count)
    """
    video_classes_file = video_dir / "classes.txt"
    
    if not video_classes_file.exists():
        print(f"警告：{video_dir.name} 目录下没有 classes.txt")
        return (False, 0, 0)
    
    # 读取视频的classes.txt
    video_classes = read_classes(video_classes_file)
    
    # 检查是否需要修复
    if video_classes == main_classes:
        return (True, 0, 0)
    
    # 创建ID映射
    id_mapping = create_id_mapping(video_classes, main_classes)
    
    print(f"\n{video_dir.name}:")
    print(f"  旧顺序: {video_classes}")
    print(f"  新顺序: {main_classes}")
    print(f"  ID映射: {id_mapping}")
    
    # 转换所有标注文件
    label_files = list(video_dir.glob("*.txt"))
    label_files = [f for f in label_files if f.name != "classes.txt"]
    
    converted_count = 0
    total_count = len(label_files)
    
    for label_file in label_files:
        if convert_label_file(label_file, id_mapping):
            converted_count += 1
    
    # 更新classes.txt
    with open(video_classes_file, 'w', encoding='utf-8') as f:
        for cls in main_classes:
            f.write(f"{cls}\n")
    
    print(f"  转换标注文件: {converted_count}/{total_count} 个文件需要转换")
    
    return (True, converted_count, total_count)

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='安全修复下面及捞面classes.txt顺序不一致问题')
    parser.add_argument('--yes', '-y', action='store_true', help='自动确认，不询问')
    args = parser.parse_args()
    
    print("="*60)
    print("安全修复下面及捞面classes.txt顺序不一致问题")
    print("="*60)
    
    # 读取主classes.txt
    if not main_classes_file.exists():
        print(f"错误：主classes.txt文件不存在: {main_classes_file}")
        return
    
    main_classes = read_classes(main_classes_file)
    print(f"\n主classes.txt顺序: {main_classes}")
    print(f"类别数量: {len(main_classes)}")
    
    # 获取所有xl视频目录
    xl_dirs = sorted([d for d in labels_base.iterdir() 
                     if d.is_dir() and d.name.startswith('xl')])
    
    if not xl_dirs:
        print("错误：未找到任何xl视频目录")
        return
    
    print(f"\n找到 {len(xl_dirs)} 个视频目录")
    
    # 确认操作
    print("\n此操作将：")
    print("  1. 转换所有标注文件中的类别ID（根据映射关系）")
    print("  2. 统一所有视频的classes.txt文件与主文件一致")
    
    if not args.yes:
        print("\n请确认要继续吗？(y/n): ", end='')
        try:
            confirm = input().strip().lower()
            if confirm != 'y':
                print("操作已取消")
                return
        except EOFError:
            print("\n非交互模式，自动执行...")
    else:
        print("\n自动确认模式，开始执行...")
    
    # 修复每个视频
    total_converted = 0
    total_labels = 0
    fixed_count = 0
    
    for video_dir in xl_dirs:
        fixed, converted, total = fix_video_classes(video_dir, main_classes)
        if fixed:
            fixed_count += 1
            total_converted += converted
            total_labels += total
    
    # 汇总结果
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"修复视频数: {fixed_count}/{len(xl_dirs)}")
    print(f"转换标注文件: {total_converted}/{total_labels} 个文件需要转换")
    print(f"总标注文件数: {total_labels}")
    
    # 验证修复结果
    print("\n验证修复结果...")
    all_consistent = True
    for video_dir in xl_dirs:
        video_classes_file = video_dir / "classes.txt"
        if video_classes_file.exists():
            video_classes = read_classes(video_classes_file)
            if video_classes != main_classes:
                print(f"  ❌ {video_dir.name}: 仍然不一致")
                all_consistent = False
            else:
                print(f"  ✓ {video_dir.name}: 已修复")
    
    if all_consistent:
        print("\n✓ 所有视频的classes.txt已统一！")
        print("\n下一步：可以开始训练模型了")
    else:
        print("\n❌ 部分视频修复失败，请检查")

if __name__ == "__main__":
    main()

