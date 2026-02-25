"""
安装OpenPose的辅助脚本
提供多种安装方式
"""
import subprocess
import sys
from pathlib import Path

print("="*60)
print("OpenPose 安装指南")
print("="*60)
print("\nOpenPose有多种安装方式，请选择：\n")

print("方案1：使用pyopenpose（推荐，最简单）")
print("  - 需要先下载OpenPose的Python包")
print("  - 适合Windows系统")
print("\n方案2：使用OpenPose Python API（需要编译）")
print("  - 需要从源码编译OpenPose")
print("  - 较复杂但功能完整")
print("\n方案3：使用OpenPose Docker（推荐Linux/Mac）")
print("  - 最简单，但Windows需要Docker Desktop")

print("\n" + "="*60)
print("推荐：使用轻量级替代方案")
print("="*60)
print("\n由于OpenPose安装复杂，建议使用以下替代方案：")
print("1. MediaPipe（已安装，速度快，精度可接受）")
print("2. MMPose（专业框架，精度高，安装简单）")
print("3. OpenPose Python包（如果必须使用OpenPose）")

choice = input("\n是否继续安装OpenPose？(y/n): ").strip().lower()

if choice != 'y':
    print("\n建议使用MMPose替代OpenPose：")
    print("  pip install mmpose mmcv")
    print("\n或者使用MediaPipe（已安装）：")
    print("  当前代码已支持MediaPipe手部检测")
    sys.exit(0)

print("\n开始安装OpenPose Python包...")
print("注意：这需要先下载OpenPose的预编译包")

# 检查是否有CUDA
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    has_cuda = result.returncode == 0
    if has_cuda:
        print("检测到NVIDIA GPU，可以使用GPU加速版本")
    else:
        print("未检测到NVIDIA GPU，将使用CPU版本")
except:
    print("无法检测GPU，将使用CPU版本")

print("\n请访问以下链接下载OpenPose：")
print("  https://github.com/CMU-Perceptual-Computing-Lab/openpose")
print("\n或者使用预编译版本：")
print("  Windows: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases")

print("\n安装完成后，需要设置环境变量：")
print("  OPENPOSE_DIR = <OpenPose安装路径>")
