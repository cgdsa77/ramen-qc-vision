#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 data/labels/下面及捞面 下各 xl* 子目录的 YOLO 标签类别 ID 按「类别名称」对齐到主 classes.txt。

用法:
  python scripts/remap_boiling_labels_to_master_classes.py
  python scripts/remap_boiling_labels_to_master_classes.py --dry-run
  python scripts/remap_boiling_labels_to_master_classes.py --only xl11 xl12 xl13
"""
import argparse
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
labels_base = project_root / "data" / "labels" / "下面及捞面"
main_classes_file = labels_base / "classes.txt"


def read_classes(path: Path) -> list:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="按主 classes.txt 重映射 YOLO 标签类别 ID")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="仅处理这些子目录名")
    args = ap.parse_args()

    if not main_classes_file.exists():
        print(f"错误: 缺少 {main_classes_file}")
        sys.exit(1)

    master = read_classes(main_classes_file)
    master_set = set(master)
    print(f"主类别顺序 ({len(master)}): {master}\n")

    dirs = sorted([d for d in labels_base.iterdir() if d.is_dir() and d.name.startswith("xl")])
    if args.only:
        dirs = [labels_base / n for n in args.only]
        dirs = [d for d in dirs if d.is_dir()]

    for video_dir in dirs:
        cf = video_dir / "classes.txt"
        if not cf.exists():
            print(f"[跳过] {video_dir.name}: 无 classes.txt")
            continue
        local = read_classes(cf)
        unknown = [c for c in local if c not in master_set]
        if unknown:
            print(f"[错误] {video_dir.name}: 未知类别 {unknown}")
            sys.exit(1)
        missing_in_local = [c for c in master if c not in local]
        if missing_in_local:
            print(f"[提示] {video_dir.name}: 本地未列出的类别（可能该视频未标）: {missing_in_local}")

        id_map = {i: master.index(local[i]) for i in range(len(local))}

        n_txt = 0
        files_rewritten = 0
        lines_total = 0

        for tf in sorted(video_dir.glob("*.txt")):
            if tf.name == "classes.txt":
                continue
            n_txt += 1
            lines_out = []
            file_changed = False
            for line in tf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    old_id = int(parts[0])
                except ValueError:
                    print(f"[错误] {tf}: 无法解析 {line}")
                    sys.exit(1)
                if old_id < 0 or old_id >= len(local):
                    print(f"[错误] {tf}: 类别 ID {old_id} 超出本地类别数 {len(local)}")
                    sys.exit(1)
                new_id = id_map[old_id]
                if new_id != old_id:
                    file_changed = True
                parts[0] = str(new_id)
                lines_out.append(" ".join(parts))
                lines_total += 1

            if file_changed:
                files_rewritten += 1
                if not args.dry_run:
                    tf.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

        if not args.dry_run:
            shutil.copy2(main_classes_file, cf)

        print(
            f"  {video_dir.name}: {n_txt} 个标签文件, {lines_total} 行框; "
            f"重写过 ID 的文件: {files_rewritten}; classes.txt 已对齐主文件"
            if not args.dry_run
            else f"  {video_dir.name}: {n_txt} 标签文件, {lines_total} 行; 将需重写: {files_rewritten} 个"
        )

    if args.dry_run:
        print("\n[dry-run] 未写入。去掉 --dry-run 后执行。")
    else:
        print("\n[OK] 完成。")


if __name__ == "__main__":
    main()
