"""检查训练检查点"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

ckpt1 = Path('models/boiling_scooping_detection/weights/last.pt')
ckpt2 = Path('models/boiling_scooping_detection2/weights/last.pt')

print("检查点状态:")
print(f"  boiling_scooping_detection: {ckpt1.exists()}")
print(f"  boiling_scooping_detection2: {ckpt2.exists()}")

if ckpt2.exists():
    print("\n使用 boiling_scooping_detection2 的检查点继续训练（最新，Epoch 34）")
elif ckpt1.exists():
    print("\n使用 boiling_scooping_detection 的检查点继续训练（Epoch 30）")

