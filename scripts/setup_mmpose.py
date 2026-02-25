"""
安装和配置MMPose（OpenPose的替代方案，推荐）
MMPose是更专业的姿态估计框架，安装简单，精度高
"""
import subprocess
import sys

print("="*60)
print("MMPose 安装指南（推荐替代OpenPose）")
print("="*60)

print("\nMMPose的优势：")
print("  ✅ 安装简单：pip install mmpose")
print("  ✅ 精度高：专业姿态估计框架")
print("  ✅ 速度快：支持GPU加速")
print("  ✅ 文档完善：中文文档齐全")
print("  ✅ 模型丰富：RTMPose、HRNet等")

choice = input("\n是否安装MMPose？(y/n): ").strip().lower()

if choice != 'y':
    print("\n取消安装")
    sys.exit(0)

print("\n开始安装MMPose...")

try:
    # 安装MMPose
    print("1. 安装MMPose...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mmpose", "mmcv", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    print("   ✅ MMPose安装成功")
    
    # 安装MMEngine
    print("2. 安装MMEngine...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mmengine", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    print("   ✅ MMEngine安装成功")
    
    print("\n" + "="*60)
    print("安装完成！")
    print("="*60)
    print("\n下一步：")
    print("  1. 运行测试: python scripts/test_mmpose.py")
    print("  2. 提取关键点: python scripts/extract_hand_keypoints_mmpose.py")
    print("  3. 在Web界面中使用MMPose")
    
except subprocess.CalledProcessError as e:
    print(f"\n[错误] 安装失败: {e}")
    print("\n请手动安装：")
    print("  pip install mmpose mmcv mmengine")
    sys.exit(1)
except Exception as e:
    print(f"\n[错误] 发生错误: {e}")
    sys.exit(1)
