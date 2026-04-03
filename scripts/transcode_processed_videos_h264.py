"""
将 processed_videos 下指定 mp4 转为浏览器可播的 H.264（yuv420p + faststart）。
使用 imageio_ffmpeg 自带的 ffmpeg，无需系统安装 ffmpeg。

用法（项目根目录）:
  python scripts/transcode_processed_videos_h264.py --stretch cm13 cm14 cm15 cm16 cm17
  python scripts/transcode_processed_videos_h264.py --stretch-all   # 抻面目录下全部 *_with_skeleton.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def _stretch_dirs() -> list[Path]:
    base = project_root / "data" / "processed_videos"
    if not base.is_dir():
        return []
    return [d for d in base.iterdir() if d.is_dir()]


def _find_stretch_videos(names: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for d in _stretch_dirs():
        if names:
            for n in names:
                p = d / f"{n}_with_skeleton.mp4"
                if p.is_file():
                    out.append(p)
        else:
            for p in d.glob("*_with_skeleton.mp4"):
                out.append(p)
    return sorted(set(out), key=lambda p: (p.parent.name, p.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stretch", nargs="*", help="仅处理这些视频名，如 cm13 cm14")
    parser.add_argument("--stretch-all", action="store_true", help="处理抻面目录下全部 *_with_skeleton.mp4")
    args = parser.parse_args()

    try:
        import imageio_ffmpeg
    except ImportError:
        print("[错误] 需要 imageio_ffmpeg（通常随 moviepy 安装）: pip install imageio-ffmpeg")
        return 1

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    names = None if args.stretch_all else (args.stretch or [])
    if not args.stretch_all and not names:
        print("请指定 --stretch cm13 ... 或 --stretch-all")
        return 1

    files = _find_stretch_videos(None if args.stretch_all else names)
    if not files:
        print("[错误] 未找到匹配的 mp4 文件")
        return 1

    for src in files:
        tmp = src.with_suffix(".h264_reencode.mp4")
        print(f"转码: {src.name} -> H.264 ...")
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
        r = subprocess.run(cmd, cwd=str(project_root))
        if r.returncode != 0:
            print(f"[失败] {src}")
            return r.returncode
        bak = src.with_suffix(".bak_before_h264.mp4")
        try:
            if bak.exists():
                bak.unlink()
            src.rename(bak)
        except OSError as e:
            print(f"[错误] 备份失败: {e}")
            return 1
        tmp.rename(src)
        try:
            bak.unlink()
        except OSError:
            pass
        print(f"  [OK] {src}")

    print("全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
