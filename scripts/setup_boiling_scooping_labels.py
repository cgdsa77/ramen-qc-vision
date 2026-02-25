#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为下面及捞面部分创建标注目录结构和classes.txt文件
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent.parent

def setup_label_directories(stage_name: str = "下面及捞面"):
    """
    为每个视频创建标注目录，并复制classes.txt
    
    Args:
        stage_name: 阶段名称
    """
    labels_base = project_root / "data" / "labels" / stage_name
    processed_base = project_root / "data" / "processed" / stage_name
    
    # 确保主classes.txt存在
    main_classes_file = labels_base / "classes.txt"
    if not main_classes_file.exists():
        print(f"错误：主classes.txt文件不存在: {main_classes_file}")
        return
    
    # 读取主classes.txt内容
    with open(main_classes_file, 'r', encoding='utf-8') as f:
        classes_content = f.read()
    
    print(f"主classes.txt内容:\n{classes_content}")
    print("\n" + "="*60)
    print("为每个视频创建标注目录...")
    print("="*60)
    
    # 获取所有已提取的视频目录
    if not processed_base.exists():
        print(f"警告：processed目录不存在: {processed_base}")
        print("请先运行 extract_video_frames.py 提取视频帧")
        return
    
    video_dirs = sorted([d for d in processed_base.iterdir() if d.is_dir()])
    
    if not video_dirs:
        print(f"警告：在 {processed_base} 中未找到视频目录")
        return
    
    created_count = 0
    for video_dir in video_dirs:
        video_name = video_dir.name
        label_dir = labels_base / video_name
        
        # 创建标注目录
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建该视频的classes.txt（必须与主文件一致）
        video_classes_file = label_dir / "classes.txt"
        with open(video_classes_file, 'w', encoding='utf-8') as f:
            f.write(classes_content)
        
        # 统计该视频的图片数量
        image_count = len(list(video_dir.glob("*.jpg")))
        print(f"  ✓ {video_name}: {image_count} 张图片")
        created_count += 1
    
    print("\n" + "="*60)
    print(f"完成！已为 {created_count} 个视频创建标注目录")
    print(f"标注目录: {labels_base}")
    print("\n下一步:")
    print("1. 使用标注工具（如X-AnyLabeling或labelImg）打开图片目录进行标注")
    print("2. 确保标注文件保存在对应的标注目录中")
    print("3. 标注完成后，运行训练脚本训练模型")
    print("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='设置标注目录结构')
    parser.add_argument(
        '--stage',
        type=str,
        default='下面及捞面',
        help='阶段名称（默认：下面及捞面）'
    )
    
    args = parser.parse_args()
    setup_label_directories(args.stage)


if __name__ == '__main__':
    main()

