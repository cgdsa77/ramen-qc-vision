#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅将主 classes.txt 同步到 cm10、cm11、cm12 三个目录，不读取、不修改 cm1～cm9 及任何其他目录。
若这三个目录的 classes 顺序与主文件不一致（如为 hand,noodle_bundle,noodle_rope），会先对标签文件
中的类别索引做 1<->2 重映射，再覆盖 classes.txt，保证与主文件 hand,noodle_rope,noodle_bundle 一致。
"""
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
labels_base = project_root / "data" / "labels" / "抻面"
main_classes_file = labels_base / "classes.txt"

ONLY_VIDEOS = ["cm10", "cm11", "cm12"]

# 主文件顺序: 0=hand, 1=noodle_rope, 2=noodle_bundle
# 若视频目录为 hand,noodle_bundle,noodle_rope 则: 旧1->新2, 旧2->新1
REMAP_OLD_BUNDLE_ROPE_TO_MAIN = {0: 0, 1: 2, 2: 1}


def remap_label_file(txt_path: Path, remap: dict) -> int:
    """重写标签文件中的类别索引，返回修改的行数。"""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                c = int(parts[0])
                parts[0] = str(remap.get(c, c))
            except ValueError:
                pass
        new_lines.append(" ".join(parts) + "\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    return len(new_lines)


def main():
    if not main_classes_file.exists():
        print(f"错误：主 classes.txt 不存在: {main_classes_file}")
        sys.exit(1)

    with open(main_classes_file, "r", encoding="utf-8") as f:
        main_classes = [line.strip() for line in f if line.strip()]

    print("仅处理以下三个目录（不触碰 cm1～cm9）：")
    print("  " + ", ".join(ONLY_VIDEOS))
    print("主 classes 顺序:", main_classes)
    print()

    for video_name in ONLY_VIDEOS:
        label_dir = labels_base / video_name
        if not label_dir.exists():
            print(f"  [跳过] {video_name}: 目录不存在")
            continue
        video_classes_file = label_dir / "classes.txt"
        if video_classes_file.exists():
            with open(video_classes_file, "r", encoding="utf-8") as f:
                video_classes = [line.strip() for line in f if line.strip()]
            if video_classes != main_classes:
                # 需要重映射标签中的类别索引（旧顺序 1=noodle_bundle, 2=noodle_rope -> 主顺序 1=noodle_rope, 2=noodle_bundle）
                if video_classes == ["hand", "noodle_bundle", "noodle_rope"]:
                    remap = REMAP_OLD_BUNDLE_ROPE_TO_MAIN
                    n_updated = 0
                    for txt_path in label_dir.glob("*.txt"):
                        if txt_path.name == "classes.txt":
                            continue
                        n_updated += remap_label_file(txt_path, remap)
                    if n_updated > 0:
                        print(f"  [重映射] {video_name}: 已更新 {n_updated} 个标签文件中的类别索引 (1<->2)")
                else:
                    print(f"  [警告] {video_name}: 类别顺序与主文件不同且非 [hand,noodle_bundle,noodle_rope]，请手动核对: {video_classes}")
        shutil.copy2(main_classes_file, video_classes_file)
        print(f"  [OK] 已同步 -> {video_name}/classes.txt")

    print()
    print("完成。未修改任何 cm1～cm9 及其他目录。")
    print("建议随后运行: python scripts/verify_classes_consistency.py")


if __name__ == "__main__":
    main()
