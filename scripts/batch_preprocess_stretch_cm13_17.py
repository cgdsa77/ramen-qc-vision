"""
一键将 cm13～cm17 处理为与 cm1～cm12 相同的预渲染带骨架视频：
1) extract_hand_keypoints_from_video.py --video cmXX
2) generate_video_with_skeleton.py --video cmXX

依赖：data/raw/抻面/cmXX.mp4（或 processed_videos 中同名原片）已存在。
用法：在项目根目录执行  python scripts/batch_preprocess_stretch_cm13_17.py
"""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
TARGETS = [f"cm{i}" for i in range(13, 18)]


def main() -> int:
    py = sys.executable
    extract = project_root / "scripts" / "extract_hand_keypoints_from_video.py"
    gen = project_root / "scripts" / "generate_video_with_skeleton.py"
    for script in (extract, gen):
        if not script.is_file():
            print(f"[错误] 未找到 {script}")
            return 1

    print("=" * 60)
    print("抻面 cm13～cm17：抽关键点 + 生成带骨架 mp4（与 cm1～cm12 一致）")
    print("=" * 60)

    for name in TARGETS:
        print(f"\n>>> [{name}] 抽手部关键点 …")
        r = subprocess.run([py, str(extract), "--video", name], cwd=str(project_root))
        if r.returncode != 0:
            print(f"[失败] {name} 抽关键点退出码 {r.returncode}")
            return r.returncode

    for name in TARGETS:
        print(f"\n>>> [{name}] 生成 processed_videos/抻面/{name}_with_skeleton.mp4 …")
        r = subprocess.run([py, str(gen), "--video", name], cwd=str(project_root))
        if r.returncode != 0:
            print(f"[失败] {name} 生成预渲染退出码 {r.returncode}")
            return r.returncode

    tc = project_root / "scripts" / "transcode_processed_videos_h264.py"
    if tc.is_file():
        print("\n>>> 浏览器可播 H.264：转码 cm13～cm17 …")
        r = subprocess.run([py, str(tc), "--stretch"] + TARGETS, cwd=str(project_root))
        if r.returncode != 0:
            print(f"[警告] 转码退出码 {r.returncode}，若网页仍报 MEDIA_ERR，请手动: python scripts/transcode_processed_videos_h264.py --stretch cm13 ...")

    out = project_root / "data" / "processed_videos" / "抻面"
    print("\n" + "=" * 60)
    print("全部完成。请在本页刷新后选择 cm13～cm17，应显示「已预处理」。")
    print(f"输出目录: {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
