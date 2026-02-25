"""
转换单个视频为H.264编码（测试用）
"""
import sys
from pathlib import Path

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent

try:
    from moviepy import VideoFileClip
except ImportError:
    print("[错误] 无法导入moviepy")
    sys.exit(1)

# 测试转换cm3视频
video_file = project_root / "data" / "processed_videos" / "抻面" / "cm3_with_skeleton.mp4"

if not video_file.exists():
    print(f"[错误] 文件不存在: {video_file}")
    sys.exit(1)

print("=" * 60)
print("转换视频为H.264编码（测试）")
print("=" * 60)
print(f"输入文件: {video_file.name}")
print(f"文件大小: {video_file.stat().st_size / 1024 / 1024:.1f} MB")

temp_output = video_file.with_suffix('.h264.mp4')
print(f"\n开始转换...")

try:
    clip = VideoFileClip(str(video_file))
    print(f"视频信息: {clip.w}x{clip.h}, {clip.fps} fps, 时长: {clip.duration:.1f}秒")
    
    clip.write_videofile(
        str(temp_output),
        codec='libx264',
        fps=clip.fps,
        preset='medium',
        bitrate='5000k',
        audio=False,
        logger=None,
        threads=4
    )
    
    clip.close()
    
    # 替换原文件
    video_file.unlink()
    temp_output.rename(video_file)
    
    print(f"\n✅ 转换成功！")
    print(f"新文件大小: {video_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n现在可以在浏览器中播放了！")
    
except Exception as e:
    print(f"\n[错误] 转换失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
