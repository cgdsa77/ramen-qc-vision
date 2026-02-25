#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理xl8相关的所有文件和目录
xl8与xl3重复，已删除
"""
import sys
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding='utf-8')

def cleanup_xl8():
    """清理xl8相关的所有文件和目录"""
    base_dir = project_root / "data"
    
    print("=" * 60)
    print("清理xl8相关文件")
    print("=" * 60)
    print("\n说明：xl8与xl3重复，已删除")
    
    deleted_items = []
    
    # 1. 删除raw视频
    raw_dir = base_dir / "raw" / "下面及捞面"
    if raw_dir.exists():
        raw_xl8_files = list(raw_dir.glob("xl8.*"))
        for f in raw_xl8_files:
            try:
                f.unlink()
                deleted_items.append(f"Raw视频: {f.name}")
                print(f"✅ 已删除: {f}")
            except Exception as e:
                print(f"❌ 删除失败: {f} - {e}")
    
    # 2. 删除processed图片目录
    processed_dir = base_dir / "processed" / "下面及捞面"
    processed_xl8 = processed_dir / "xl8" if processed_dir.exists() else None
    if processed_xl8 and processed_xl8.exists():
        try:
            shutil.rmtree(processed_xl8)
            deleted_items.append(f"Processed目录: {processed_xl8}")
            print(f"✅ 已删除: {processed_xl8}")
        except Exception as e:
            print(f"❌ 删除失败: {processed_xl8} - {e}")
    
    # 3. 删除labels标注目录
    labels_dir = base_dir / "labels" / "下面及捞面"
    labels_xl8 = labels_dir / "xl8" if labels_dir.exists() else None
    if labels_xl8 and labels_xl8.exists():
        try:
            shutil.rmtree(labels_xl8)
            deleted_items.append(f"Labels目录: {labels_xl8}")
            print(f"✅ 已删除: {labels_xl8}")
        except Exception as e:
            print(f"❌ 删除失败: {labels_xl8} - {e}")
    
    print("\n" + "=" * 60)
    print("清理完成")
    print("=" * 60)
    
    if deleted_items:
        print(f"\n共删除 {len(deleted_items)} 项：")
        for item in deleted_items:
            print(f"  - {item}")
    else:
        print("\n✅ 未发现xl8相关文件，无需清理")
    
    # 验证清理结果
    print("\n" + "=" * 60)
    print("验证清理结果")
    print("=" * 60)
    
    remaining = []
    if raw_dir.exists():
        raw_xl8 = list(raw_dir.glob("xl8.*"))
        if raw_xl8:
            remaining.extend([f"Raw: {f.name}" for f in raw_xl8])
    
    if processed_dir.exists() and (processed_dir / "xl8").exists():
        remaining.append("Processed: xl8目录")
    
    if labels_dir.exists() and (labels_dir / "xl8").exists():
        remaining.append("Labels: xl8目录")
    
    if remaining:
        print("\n⚠️  仍有残留文件/目录：")
        for item in remaining:
            print(f"  - {item}")
    else:
        print("\n✅ 所有xl8相关文件已完全清理")

if __name__ == "__main__":
    cleanup_xl8()

