#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的抻面分析测试脚本
用于快速测试新视频的分析效果
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.analyze_stretch import analyze_video


def main():
    """主函数"""
    print("="*60)
    print("抻面动作分析测试")
    print("="*60)
    print("\n说明：")
    print("  • 此工具基于已训练的基准模型分析新视频")
    print("  • 基准模型基于 cm1, cm2, cm3 三个标准视频训练")
    print("  • 分析结果会给出与标准动作的相似度得分（0-1，越高越相似）")
    print("\n" + "-"*60)
    
    # 获取要分析的视频名称
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        print("\n可用的视频（已标注的）:")
        labels_dir = project_root / "data" / "labels" / "抻面"
        if labels_dir.exists():
            videos = [d.name for d in labels_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('cm')]
            for v in sorted(videos):
                print(f"  • {v}")
        
        video_name = input("\n请输入要分析的视频名称（如 cm4）: ").strip()
    
    if not video_name:
        print("错误：未指定视频名称")
        return
    
    try:
        result = analyze_video(video_name)
        
        print("\n" + "="*60)
        print(f"分析结果: {result['video_name']}")
        print("="*60)
        print(f"\n📊 总帧数: {result['total_frames']}")
        print(f"🎯 相似度得分: {result['similarity_score']:.3f} / 1.000")
        
        # 得分等级
        score = result['similarity_score']
        if score >= 0.9:
            level = "优秀 ⭐⭐⭐"
        elif score >= 0.8:
            level = "良好 ⭐⭐"
        elif score >= 0.7:
            level = "一般 ⭐"
        else:
            level = "需要改进"
        
        print(f"📈 评价: {level}")
        print(f"\n基于 {result['baseline_info']['num_baseline_videos']} 个标准视频的基准模型")
        
        print("\n" + "-"*60)
        print("各项得分详情:")
        print("-"*60)
        for feature_name, score_info in result['component_scores'].items():
            score_val = score_info['score']
            bar_length = int(score_val * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"\n{feature_name}:")
            print(f"  当前值: {score_info['value']:.4f}")
            print(f"  基准值: {score_info['baseline_mean']:.4f}")
            print(f"  得分: {score_val:.3f} [{bar}]")
        
        # 保存结果
        output_dir = project_root / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"stretch_analysis_{result['video_name']}.json"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "="*60)
        print(f"✅ 分析完成！结果已保存到: {output_file}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示：")
        print("  1. 请确保视频已标注（在 data/labels/抻面/ 下有对应的标注文件）")
        print("  2. 如果模型不存在，请先运行训练脚本:")
        print("     python src/training/train_stretch_baseline.py")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




