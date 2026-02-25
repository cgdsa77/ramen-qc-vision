"""
使用ffmpeg将视频转换为H.264编码（浏览器兼容）
如果moviepy不可用，使用ffmpeg作为替代方案
"""
import sys
import subprocess
from pathlib import Path

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_video_ffmpeg(input_path: Path, output_path: Path):
    """使用ffmpeg转换视频为H.264编码"""
    if not input_path.exists():
        print(f"[错误] 文件不存在: {input_path}")
        return False
    
    try:
        print(f"  转换: {input_path.name} -> {output_path.name}")
        
        # 使用ffmpeg转换为H.264编码
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',  # H.264编码
            '-preset', 'medium',
            '-crf', '23',  # 质量参数（18-28，23是默认值）
            '-c:a', 'copy',  # 如果有音频，复制音频流
            '-movflags', '+faststart',  # 优化网络播放
            '-y',  # 覆盖输出文件
            str(output_path)
        ]
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            print(f"  ✅ 转换成功")
            return True
        else:
            print(f"  [错误] 转换失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  [错误] 转换超时")
        return False
    except Exception as e:
        print(f"  [错误] 转换失败: {e}")
        return False


def main():
    """主函数：转换所有视频"""
    print("=" * 60)
    print("使用ffmpeg转换视频为H.264编码（浏览器兼容）")
    print("=" * 60)
    
    # 检查ffmpeg
    if not check_ffmpeg():
        print("\n[错误] 未找到ffmpeg")
        print("请安装ffmpeg:")
        print("  Windows: 下载 https://ffmpeg.org/download.html")
        print("  或使用: choco install ffmpeg")
        sys.exit(1)
    
    print("\n✅ ffmpeg 可用")
    
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
        print("\n[警告] 未找到视频文件")
        return
    
    print(f"\n找到 {len(all_video_files)} 个视频文件")
    print("=" * 60)
    
    # 转换每个视频
    success_count = 0
    for i, video_file in enumerate(all_video_files, 1):
        print(f"\n[{i}/{len(all_video_files)}] 处理: {video_file.name}")
        
        # 创建临时输出文件
        temp_output = video_file.with_suffix('.h264.mp4')
        
        if convert_video_ffmpeg(video_file, temp_output):
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
