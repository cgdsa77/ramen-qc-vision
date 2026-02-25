"""
检查视频转换状态
"""
import sys
from pathlib import Path

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent

print("=" * 60)
print("检查视频转换状态")
print("=" * 60)

# 检查抻面视频
stretch_dir = project_root / "data" / "processed_videos" / "抻面"
if stretch_dir.exists():
    stretch_files = list(stretch_dir.glob("*_with_skeleton.mp4"))
    print(f"\n抻面视频: {len(stretch_files)} 个")
    for f in sorted(stretch_files)[:3]:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")

# 检查下面及捞面视频
boiling_dir = project_root / "data" / "processed_videos" / "下面及捞面"
if boiling_dir.exists():
    boiling_files = list(boiling_dir.glob("*_with_skeleton.mp4"))
    print(f"\n下面及捞面视频: {len(boiling_files)} 个")
    for f in sorted(boiling_files)[:3]:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("提示: 视频应该已经转换为H.264编码")
print("可以在浏览器中打开: http://127.0.0.1:8000/video-skeleton-local")
print("=" * 60)
