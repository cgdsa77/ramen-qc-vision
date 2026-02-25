#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""备份标注数据，防止意外修改"""
from pathlib import Path
import shutil
from datetime import datetime

project_root = Path(__file__).parent.parent
labels_dir = project_root / "data" / "labels" / "抻面"
backup_dir = project_root / "data" / "labels_backup"

def backup_labels():
    """备份所有标注文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}"
    
    if not labels_dir.exists():
        print(f"[错误] 标注目录不存在: {labels_dir}")
        return
    
    print(f"开始备份标注数据...")
    print(f"  源目录: {labels_dir}")
    print(f"  备份到: {backup_path}")
    
    # 复制整个目录
    shutil.copytree(labels_dir, backup_path)
    
    # 统计备份的文件
    txt_files = list(backup_path.rglob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "classes.txt"]
    
    print(f"\n[OK] 备份完成！")
    print(f"  备份了 {len(txt_files)} 个标注文件")
    print(f"  备份位置: {backup_path}")
    print(f"\n提示: 如果发现标注被意外修改，可以使用此备份恢复")

if __name__ == "__main__":
    backup_labels()

