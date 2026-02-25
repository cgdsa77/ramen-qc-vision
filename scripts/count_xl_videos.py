"""统计xl视频数量"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

labels_dir = Path('data/labels/下面及捞面')
xl_dirs = sorted([d.name for d in labels_dir.iterdir() if d.is_dir() and d.name.startswith('xl')])

print(f"找到 {len(xl_dirs)} 个xl视频目录:")
for d in xl_dirs:
    print(f"  - {d}")

