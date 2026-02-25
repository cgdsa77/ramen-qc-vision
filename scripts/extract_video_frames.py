#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取视频帧为图片，用于标注
"""
import sys
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent


def extract_frames(video_dir: Path, output_dir: Path, fps: float = 2.0):
    """
    提取视频帧
    
    Args:
        video_dir: 视频文件目录
        output_dir: 输出图片目录
        fps: 提取帧率（每秒提取的帧数）
    """
    if not video_dir.exists():
        print(f"错误：视频目录不存在: {video_dir}")
        return
    
    # 获取所有视频文件（支持.mp4和.MP4）
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))
    
    if not video_files:
        print(f"警告：在 {video_dir} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"提取帧率: {fps} fps\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_file in sorted(video_files):
        video_name = video_file.stem  # 获取不带扩展名的文件名
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出文件名模式：视频名_%05d.jpg
        output_pattern = str(video_output_dir / f"{video_name}_%05d.jpg")
        
        print(f"正在提取: {video_file.name} -> {video_output_dir.name}")
        
        # 使用ffmpeg提取帧
        cmd = [
            'ffmpeg',
            '-i', str(video_file),
            '-vf', f'fps={fps}',
            '-q:v', '2',  # 高质量JPEG
            '-y',  # 覆盖已存在的文件
            output_pattern
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # 统计提取的图片数量
            extracted_count = len(list(video_output_dir.glob("*.jpg")))
            print(f"  ✓ 完成，提取了 {extracted_count} 张图片\n")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 错误: {e}")
            if e.stderr:
                print(f"  错误信息: {e.stderr}\n")
        except FileNotFoundError:
            print(f"  ✗ 错误: 未找到ffmpeg，请先安装ffmpeg")
            print(f"  下载地址: https://ffmpeg.org/download.html\n")
            break
    
    print("=" * 60)
    print(f"完成！所有图片已提取到: {output_dir}")
    print("=" * 60)


def main():
    """主函数"""
    # 默认提取下面及捞面的视频
    stage_name = "下面及捞面"
    fps = 2.0
    
    video_dir = project_root / "data" / "raw" / stage_name
    output_dir = project_root / "data" / "processed" / stage_name
    
    extract_frames(video_dir, output_dir, fps)


if __name__ == '__main__':
    main()

