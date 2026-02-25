#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证抻面模型权重：检查 models/stretch_detection* 下 best.pt 的类别数，
确认是否为 3 类 (hand / noodle_rope / noodle_bundle)，并可选复制到标准位置。
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 请先安装 ultralytics: pip install ultralytics")
        return 1

    models_dir = project_root / "models"
    if not models_dir.exists():
        print("[ERROR] 未找到 models 目录")
        return 1

    # 收集所有 stretch_detection* / weights / best.pt
    candidates = []
    for d in sorted(models_dir.iterdir()):
        if d.is_dir() and d.name.startswith("stretch_detection"):
            best_pt = d / "weights" / "best.pt"
            if best_pt.exists():
                num_str = d.name.replace("stretch_detection", "") or "1"
                num = int(num_str) if num_str.isdigit() else 1
                candidates.append((num, d.name, best_pt))

    if not candidates:
        print("未找到任何 models/stretch_detection*/weights/best.pt")
        print("请先训练抻面模型，或运行 scripts/copy_latest_model.py 复制权重。")
        return 1

    print("=" * 60)
    print("抻面模型权重验证（3 类: hand / noodle_rope / noodle_bundle）")
    print("=" * 60)

    results = []  # (num, dir_name, best_pt, n_cls)
    for num, dir_name, best_pt in sorted(candidates, key=lambda x: x[0]):
        try:
            model = YOLO(str(best_pt))
            names = getattr(model, "names", None)
            if names is None:
                n_cls = 0
                names_str = "?"
            else:
                n_cls = len(names) if hasattr(names, "__len__") else len(list(names.values())) if isinstance(names, dict) else 0
                if isinstance(names, dict):
                    names_str = ", ".join(f"{k}={v}" for k, v in sorted(names.items()))
                else:
                    names_str = str(names)
            ok = "✓ 3类" if n_cls == 3 else f"✗ {n_cls}类（应为3类）"
            print(f"\n  {dir_name}/weights/best.pt")
            print(f"    类别数: {n_cls}  {ok}")
            print(f"    类别名: {names_str}")
            results.append((num, dir_name, best_pt, n_cls))
        except Exception as e:
            print(f"\n  {dir_name}/weights/best.pt  加载失败: {e}")

    standard_dir = models_dir / "stretch_detection" / "weights"
    standard_best = standard_dir / "best.pt"

    three_class = [(n, pt) for n, _, pt, nc in results if nc == 3]
    if not three_class:
        print("\n[WARN] 未发现 3 类模型，检测页 hand/面条束 可能仍为 0。请使用 datasets/stretch_detection 训练的权重。")
        return 0

    three_class.sort(key=lambda x: x[0])
    best_num, best_pt = three_class[-1]
    print(f"\n[OK] 推荐使用（编号最大的 3 类）: {best_pt}")

    standard_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    if best_pt.resolve() != standard_best.resolve():
        shutil.copy2(best_pt, standard_best)
        print(f"[OK] 已复制到标准位置: {standard_best}")
    else:
        print("标准位置已是该 3 类模型，无需复制。")
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
