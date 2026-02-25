"""检查moviepy是否可用"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip
    print("moviepy 已安装且可用")
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    print(f"moviepy 未安装或导入失败: {e}")
    print("请安装: pip install moviepy")
    MOVIEPY_AVAILABLE = False
    sys.exit(1)
