#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用主 classes.txt 覆盖所有抻面 cm* 目录下的 classes.txt，不修改任何标注 *.txt 文件。
用于：labelImg 或误操作改动了某视频的 classes.txt 后，恢复与主文件一致，避免标签映射错乱。
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
main_classes_file = labels_base / "classes.txt"


def main():
    if not main_classes_file.exists():
        print(f"错误：主 classes.txt 不存在: {main_classes_file}")
        sys.exit(1)
    master_content = main_classes_file.read_text(encoding="utf-8")

    video_dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith("cm")])
    if not video_dirs:
        print("未找到任何 cm* 视频目录")
        sys.exit(0)

    updated = []
    for video_dir in video_dirs:
        classes_file = video_dir / "classes.txt"
        if not classes_file.exists():
            classes_file.write_text(master_content, encoding="utf-8")
            print(f"  已创建并写入: {video_dir.name}/classes.txt")
            updated.append(video_dir.name)
        else:
            current = classes_file.read_text(encoding="utf-8")
            if current.strip() != master_content.strip():
                classes_file.write_text(master_content, encoding="utf-8")
                print(f"  已覆盖: {video_dir.name}/classes.txt")
                updated.append(video_dir.name)
            else:
                print(f"  一致，未改: {video_dir.name}/classes.txt")

    print()
    print(f"主文件: {main_classes_file}")
    print(f"已同步 {len(updated)} 个视频目录的 classes.txt；未修改任何标注 *.txt 文件。")
    print("建议随后运行: python scripts/verify_classes_consistency.py")


if __name__ == "__main__":
    main()
