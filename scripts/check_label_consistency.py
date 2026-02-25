#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查标注数据的一致性，防止被意外修改"""
from pathlib import Path
from collections import defaultdict
import hashlib

project_root = Path(__file__).parent.parent
labels_dir = project_root / "data" / "labels" / "抻面"
checksum_file = project_root / "data" / "labels" / "label_checksums.txt"

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def save_checksums():
    """保存所有标注文件的校验和"""
    checksums = {}
    
    # 读取所有标注文件
    for video_dir in labels_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name == "labels_backup":
            continue
        
        for label_file in video_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            
            rel_path = label_file.relative_to(labels_dir)
            checksums[str(rel_path)] = calculate_file_hash(label_file)
    
    # 保存校验和
    with open(checksum_file, 'w', encoding='utf-8') as f:
        for path, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {path}\n")
    
    print(f"[OK] 已保存 {len(checksums)} 个标注文件的校验和到: {checksum_file}")
    return checksums

def check_checksums():
    """检查标注文件是否被修改"""
    if not checksum_file.exists():
        print("[警告] 校验和文件不存在，首次运行将创建校验和")
        save_checksums()
        return
    
    # 读取保存的校验和
    saved_checksums = {}
    with open(checksum_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('  ', 1)
            if len(parts) == 2:
                checksum, path = parts
                saved_checksums[path] = checksum
    
    # 检查当前文件
    modified_files = []
    missing_files = []
    new_files = []
    
    current_paths = set()
    for video_dir in labels_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name == "labels_backup":
            continue
        
        for label_file in video_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            
            rel_path = str(label_file.relative_to(labels_dir))
            current_paths.add(rel_path)
            current_checksum = calculate_file_hash(label_file)
            
            if rel_path not in saved_checksums:
                new_files.append(rel_path)
            elif saved_checksums[rel_path] != current_checksum:
                modified_files.append(rel_path)
    
    # 查找缺失的文件
    for path in saved_checksums:
        if path not in current_paths:
            missing_files.append(path)
    
    # 报告结果
    print("="*60)
    print("标注文件一致性检查")
    print("="*60)
    
    if not modified_files and not missing_files and not new_files:
        print("[OK] 所有标注文件未被修改，一致性良好！")
    else:
        if modified_files:
            print(f"\n[警告] 发现 {len(modified_files)} 个文件被修改:")
            for path in modified_files[:10]:  # 只显示前10个
                print(f"  - {path}")
            if len(modified_files) > 10:
                print(f"  ... 还有 {len(modified_files) - 10} 个文件")
        
        if missing_files:
            print(f"\n[警告] 发现 {len(missing_files)} 个文件缺失:")
            for path in missing_files[:10]:
                print(f"  - {path}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件")
        
        if new_files:
            print(f"\n[信息] 发现 {len(new_files)} 个新文件:")
            for path in new_files[:10]:
                print(f"  + {path}")
            if len(new_files) > 10:
                print(f"  ... 还有 {len(new_files) - 10} 个文件")
    
    print("\n" + "="*60)
    return modified_files, missing_files, new_files

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_checksums()
    else:
        modified, missing, new = check_checksums()
        if modified or missing:
            print("\n建议: 如果发现标注被意外修改，请检查:")
            print("  1. 是否使用了标注工具（labelImg等）修改了标注")
            print("  2. 是否运行了可能修改标注的脚本")
            print("  3. 如果有备份，可以从备份恢复")
            print("\n要更新校验和，请运行: python scripts/check_label_consistency.py save")

