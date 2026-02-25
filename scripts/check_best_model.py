"""检查最佳模型"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

best_model = Path('models/boiling_scooping_detection4/weights/best.pt')
if best_model.exists():
    size_mb = best_model.stat().st_size / 1024 / 1024
    print(f'最佳模型已保存: {best_model}')
    print(f'模型大小: {size_mb:.2f} MB')
else:
    print('最佳模型未找到')

