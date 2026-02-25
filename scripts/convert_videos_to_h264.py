"""
将现有的视频文件转换为H.264编码（浏览器兼容）
使用moviepy转换，确保视频能在浏览器播放
"""
import sys
from pathlib import Path

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试多种导入方式（moviepy 2.1.2可以直接从moviepy导入）
MOVIEPY_AVAILABLE = False
VideoFileClip = None

try:
    # 优先尝试从moviepy直接导入（2.1.2版本）
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        # 备选：从moviepy.editor导入（旧版本）
        from moviepy.editor import VideoFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError as e:
        MOVIEPY_AVAILABLE = False
        print("[错误] 无法导入moviepy")
        print(f"错误详情: {e}")
        print("请安装或更新moviepy: pip install --upgrade moviepy")
        sys.exit(1)


def convert_video(input_path: Path, output_path: Path):
    """转换单个视频为H.264编码"""
    if not input_path.exists():
        print(f"[错误] 文件不存在: {input_path}")
        return False
    
    try:
        print(f"  转换: {input_path.name}")
        print(f"  文件大小: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # 读取视频
        clip = VideoFileClip(str(input_path))
        print(f"  视频信息: {clip.w}x{clip.h}, {clip.fps} fps, 时长: {clip.duration:.1f}秒")
        
        # 写入H.264编码的MP4
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            fps=clip.fps,
            preset='medium',
            bitrate='5000k',
            audio=False,
            logger=None,
            threads=4
        )
        
        clip.close()
        
        new_size = output_path.stat().st_size / 1024 / 1024
        print(f"  ✅ 转换成功 (新文件大小: {new_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [错误] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：转换所有视频"""
    print("=" * 60)
    print("转换视频为H.264编码（浏览器兼容）")
    print("=" * 60)
    
    # 处理抻面和下面及捞面两个目录
    directories = [
        project_root / "data" / "processed_videos" / "抻面",
        project_root / "data" / "processed_videos" / "下面及捞面"
    ]
    
    all_video_files = []
    for processed_dir in directories:
        if processed_dir.exists():
            video_files = list(processed_dir.glob("*_with_skeleton.mp4"))
            all_video_files.extend(video_files)
    
    if not all_video_files:
        print("[警告] 未找到视频文件")
        return
    
    print(f"\n找到 {len(all_video_files)} 个视频文件")
    print("=" * 60)
    
    # 转换每个视频
    success_count = 0
    for i, video_file in enumerate(all_video_files, 1):
        print(f"\n[{i}/{len(all_video_files)}] 处理: {video_file.name}")
        
        # 创建临时输出文件
        temp_output = video_file.with_suffix('.h264.mp4')
        
        if convert_video(video_file, temp_output):
            # 替换原文件
            video_file.unlink()
            temp_output.rename(video_file)
            success_count += 1
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"成功转换 {success_count}/{len(all_video_files)} 个视频")
    print("\n所有视频现在使用H.264编码，浏览器兼容！")


if __name__ == "__main__":
    main()
