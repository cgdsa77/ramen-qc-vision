"""复制下面及捞面最佳模型到部署位置"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent

# 自动找到最新的训练目录
models_dir = project_root / "models"
training_dirs = sorted([d for d in models_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('boiling_scooping_detection') and d.name != 'boiling_scooping_detection'],
                      key=lambda x: int(x.name.replace('boiling_scooping_detection', '') or '0'),
                      reverse=True)

if not training_dirs:
    print("[错误] 未找到训练目录")
    sys.exit(1)

latest_training_dir = training_dirs[0]
source = latest_training_dir / "weights" / "best.pt"

# 目标1：标准部署目录
target_dir1 = project_root / "models" / "boiling_scooping_detection" / "weights"
target_dir1.mkdir(parents=True, exist_ok=True)
target1 = target_dir1 / "best.pt"

# 目标2：最终部署模型文件
target2 = project_root / "models" / "boiling_scooping_detection_model.pt"

print("=" * 60)
print("复制下面及捞面最佳模型")
print("=" * 60)
print()

if not source.exists():
    print(f"[错误] 源文件不存在: {source}")
    sys.exit(1)

print(f"训练目录: {latest_training_dir.name}")
print(f"源文件: {source}")
print(f"  大小: {source.stat().st_size / 1024 / 1024:.2f} MB")
print()

# 复制到标准部署目录
print(f"复制到: {target1}")
shutil.copy2(source, target1)
print(f"  [OK] 已复制到标准部署目录")
print()

# 复制到最终部署文件
print(f"复制到: {target2}")
shutil.copy2(source, target2)
print(f"  [OK] 已复制到最终部署文件")
print()

print("=" * 60)
print("模型复制完成！")
print("=" * 60)
print()
print("模型位置:")
print(f"  1. 标准部署目录: {target1}")
print(f"  2. 最终部署文件: {target2}")
print()
print("现在可以启动Web服务使用新模型了！")

