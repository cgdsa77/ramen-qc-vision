#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全新增抻面视频目录：只创建新目录并从主 classes.txt 复制，不修改任何已有数据。
用法: python scripts/add_stretch_video_safe.py cm13 [cm14] [cm15]
"""
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
images_base = project_root / "data" / "processed" / "抻面"
main_classes = labels_base / "classes.txt"

# 抻面固定类别顺序（与 data/labels/抻面/classes.txt 必须一致）
CANONICAL_CLASSES = ["hand", "noodle_rope", "noodle_bundle"]


def main():
    if not main_classes.exists():
        print(f"错误：主 classes.txt 不存在: {main_classes}")
        sys.exit(1)
    with open(main_classes, "r", encoding="utf-8") as f:
        content = [line.strip() for line in f if line.strip()]
    if content != CANONICAL_CLASSES:
        print(f"警告：主 classes.txt 内容与规范不一致，当前为: {content}")
        print(f"规范顺序应为: {CANONICAL_CLASSES}")

    names = [a for a in sys.argv[1:] if a.startswith("cm") and a[2:].isdigit()]
    if not names:
        existing = sorted([d.name for d in labels_base.iterdir() if d.is_dir() and d.name.startswith("cm")])
        print(f"当前已存在的抻面视频目录（共 {len(existing)} 个）: {existing}")
        print()
        print("用法: python scripts/add_stretch_video_safe.py cm13 [cm14] [cm15]")
        print("示例: python scripts/add_stretch_video_safe.py cm13 cm14 cm15")
        sys.exit(1)

    existing = sorted([d.name for d in labels_base.iterdir() if d.is_dir() and d.name.startswith("cm")])
    print(f"当前已有抻面视频目录（共 {len(existing)} 个）: {existing}")
    created = []
    skipped = []

    for name in names:
        label_dir = labels_base / name
        if label_dir.exists():
            print(f"  跳过 {name}：目录已存在，不覆盖。")
            skipped.append(name)
            continue
        label_dir.mkdir(parents=True, exist_ok=True)
        dest_classes = label_dir / "classes.txt"
        shutil.copy2(main_classes, dest_classes)
        print(f"  已创建 {label_dir} 并复制 classes.txt")
        # 可选：创建对应图片目录（空目录，供后续放抽帧图片）
        img_dir = images_base / name
        if not img_dir.exists():
            img_dir.mkdir(parents=True, exist_ok=True)
            print(f"  已创建图片目录 {img_dir}（可放入抽帧图片）")
        created.append(name)

    print()
    print(f"已安全创建: {created}")
    if skipped:
        print(f"已跳过（已存在）: {skipped}")
    print("请勿用 labelImg 修改新目录下的 classes.txt；标注时类别顺序保持: hand, noodle_rope, noodle_bundle。")
    print("标注完成后请运行: python scripts/sync_stretch_classes_from_master.py")


if __name__ == "__main__":
    main()
