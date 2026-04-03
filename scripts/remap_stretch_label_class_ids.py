#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将抻面子目录下 YOLO 标注的 class id 按「旧 classes.txt 行序 → 类名 → 主 classes.txt 行序」重映射。

用于 LabelImg 子目录 classes.txt 与 data/labels/抻面/classes.txt 顺序不一致时，
仅改 classes.txt 会导致类别语义错误，必须同步改每行第一个数字。

用法（项目根目录）:
  python scripts/remap_stretch_label_class_ids.py --dry-run
  python scripts/remap_stretch_label_class_ids.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 标准顺序（须与 data/labels/抻面/classes.txt 一致）
CANONICAL = ["hand", "noodle_rope", "noodle_bundle"]

# 各视频在修正前 LabelImg 中 classes.txt 的行顺序（从上到下为 id 0,1,2）
# cm14 已与主文件一致，无需重映射
OLD_ORDER: dict[str, list[str]] = {
    "cm13": ["hand", "noodle_bundle", "noodle_rope"],
    "cm15": ["noodle_bundle", "hand", "noodle_rope"],
    "cm16": ["hand", "noodle_bundle", "noodle_rope"],
    "cm17": ["hand", "noodle_bundle", "noodle_rope"],
}


def build_map(old_names: list[str]) -> dict[int, int]:
    name_to_canon_id = {n: i for i, n in enumerate(CANONICAL)}
    out = {}
    for old_id, name in enumerate(old_names):
        if name not in name_to_canon_id:
            raise ValueError(f"未知类名: {name}")
        out[old_id] = name_to_canon_id[name]
    return out


def remap_file(path: Path, id_map: dict[int, int], dry: bool) -> bool:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return False
    new_lines = []
    changed = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        old = int(float(parts[0]))
        new = id_map.get(old, old)
        if new != old:
            changed = True
        parts[0] = str(new)
        new_lines.append(" ".join(parts))
    text = "\n".join(new_lines) + ("\n" if new_lines else "")
    if changed and not dry:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="重映射抻面 YOLO 标注 class id")
    parser.add_argument("--dry-run", action="store_true", help="只报告会修改的文件，不写盘")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "data" / "labels" / "抻面"
    main_classes = root / "classes.txt"
    if not main_classes.exists():
        raise SystemExit(f"缺少主 classes: {main_classes}")

    for sub, old_names in OLD_ORDER.items():
        id_map = build_map(old_names)
        label_dir = root / sub
        if not label_dir.is_dir():
            print(f"[跳过] 无目录: {label_dir}")
            continue
        print(f"\n{sub} 映射 {id_map} (旧顺序 {old_names})")
        n_changed = 0
        for txt in sorted(label_dir.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            if remap_file(txt, id_map, args.dry_run):
                n_changed += 1
                print(f"  {'[dry] ' if args.dry_run else ''}更新: {txt.name}")
        print(f"  共修改 {n_changed} 个文件" if not args.dry_run else f"  将修改 {n_changed} 个文件（dry-run）")

    # 写回标准 classes.txt
    canon_text = "\n".join(CANONICAL) + "\n"
    for sub in OLD_ORDER:
        p = root / sub / "classes.txt"
        if p.exists() and p.read_text(encoding="utf-8").strip().split() != CANONICAL:
            if not args.dry_run:
                p.write_text(canon_text, encoding="utf-8")
            print(f"{'[dry] ' if args.dry_run else ''}classes.txt -> {p.relative_to(root)}")


if __name__ == "__main__":
    main()
