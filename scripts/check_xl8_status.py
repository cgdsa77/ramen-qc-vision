#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查xl8的状态，确认是否需要清理
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding='utf-8')

def check_xl8_status():
    """检查xl8相关的文件和目录"""
    base_dir = project_root / "data"
    
    # 检查raw视频
    raw_dir = base_dir / "raw" / "下面及捞面"
    raw_xl8 = list(raw_dir.glob("xl8.*")) if raw_dir.exists() else []
    
    # 检查processed图片
    processed_dir = base_dir / "processed" / "下面及捞面"
    processed_xl8 = processed_dir / "xl8" if processed_dir.exists() else None
    
    # 检查labels标注
    labels_dir = base_dir / "labels" / "下面及捞面"
    labels_xl8 = labels_dir / "xl8" if labels_dir.exists() else None
    
    print("=" * 60)
    print("xl8 状态检查")
    print("=" * 60)
    
    print("\n1. Raw视频文件:")
    if raw_xl8:
        for f in raw_xl8:
            print(f"   ❌ 存在: {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print("   ✅ 已删除（不存在）")
    
    print("\n2. Processed图片目录:")
    if processed_xl8 and processed_xl8.exists():
        img_count = len(list(processed_xl8.glob("*.jpg")))
        print(f"   ❌ 存在: {processed_xl8} ({img_count} 张图片)")
    else:
        print("   ✅ 已删除（不存在）")
    
    print("\n3. Labels标注目录:")
    if labels_xl8 and labels_xl8.exists():
        txt_count = len([f for f in labels_xl8.glob("*.txt") if f.name != "classes.txt"])
        classes_exists = (labels_xl8 / "classes.txt").exists()
        print(f"   ❌ 存在: {labels_xl8}")
        print(f"      - 标注文件: {txt_count} 个")
        print(f"      - classes.txt: {'存在' if classes_exists else '不存在'}")
    else:
        print("   ✅ 已删除（不存在）")
    
    print("\n" + "=" * 60)
    print("建议操作:")
    print("=" * 60)
    
    needs_cleanup = False
    if raw_xl8 or (processed_xl8 and processed_xl8.exists()) or (labels_xl8 and labels_xl8.exists()):
        needs_cleanup = True
        print("\n⚠️  发现xl8相关文件/目录仍存在，建议清理：")
        if raw_xl8:
            print(f"   - 删除raw视频: {[f.name for f in raw_xl8]}")
        if processed_xl8 and processed_xl8.exists():
            print(f"   - 删除processed目录: {processed_xl8}")
        if labels_xl8 and labels_xl8.exists():
            print(f"   - 删除labels目录: {labels_xl8}")
    else:
        print("\n✅ 所有xl8相关文件已删除，无需清理")
    
    # 检查文档中的引用
    print("\n" + "=" * 60)
    print("文档检查:")
    print("=" * 60)
    
    docs_to_update = []
    quick_start = project_root / "docs" / "下面及捞面标注快速开始.md"
    if quick_start.exists():
        content = quick_start.read_text(encoding='utf-8')
        if 'xl8' in content:
            docs_to_update.append(quick_start)
            print(f"   ⚠️  {quick_start.name} 中提到了xl8")
    
    class_update = project_root / "docs" / "类别更新说明_下面及捞面_v2.md"
    if class_update.exists():
        content = class_update.read_text(encoding='utf-8')
        if 'xl8' in content:
            docs_to_update.append(class_update)
            print(f"   ⚠️  {class_update.name} 中提到了xl8")
    
    if docs_to_update:
        print(f"\n建议更新 {len(docs_to_update)} 个文档，移除xl8的引用")
    else:
        print("\n✅ 文档中未发现xl8引用")
    
    return needs_cleanup, docs_to_update

if __name__ == "__main__":
    check_xl8_status()

