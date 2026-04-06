#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 data/labels 下各子目录中「至少含一个框」的 YOLO 标注文件数（有效标注帧）。"""
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def count_labeled_frames(labels_root: Path):
    total_frames = 0
    total_boxes = 0
    by_sub = {}
    if not labels_root.exists():
        return by_sub, 0, 0
    for sub in sorted(labels_root.iterdir()):
        if not sub.is_dir():
            continue
        n_frames = 0
        n_boxes = 0
        for f in sub.glob("*.txt"):
            if f.name.lower() == "classes.txt":
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                continue
            n_frames += 1
            n_boxes += len(lines)
        if n_frames:
            by_sub[sub.name] = {"frames": n_frames, "boxes": n_boxes}
            total_frames += n_frames
            total_boxes += n_boxes
    return by_sub, total_frames, total_boxes


def main():
    root = project_root / "data" / "labels"
    grand_frames = 0
    grand_boxes = 0
    print("=== 有效标注帧数（至少含 1 个框的标注文件数）===\n")
    for stage in sorted(root.iterdir()):
        if not stage.is_dir():
            continue
        by_sub, tf, tb = count_labeled_frames(stage)
        grand_frames += tf
        grand_boxes += tb
        print(f"【{stage.name}】 小计: {tf} 帧, {tb} 个框")
        for k in sorted(by_sub.keys()):
            d = by_sub[k]
            print(f"  - {k}: {d['frames']} 帧, {d['boxes']} 框")
        print()
    print(f"--- data/labels 合计: {grand_frames} 有效标注帧, {grand_boxes} 个标注框 ---")


if __name__ == "__main__":
    main()
