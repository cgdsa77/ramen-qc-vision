"""
运行增强的综合评分系统
基于论文《基于深度学习的乒乓球姿态动作评分方法》
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.enhanced_comprehensive_scorer import EnhancedComprehensiveScorer


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行增强的综合评分系统')
    parser.add_argument('video', nargs='?', help='视频名称（如cm1），如果不指定则处理所有视频')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    
    args = parser.parse_args()
    
    scorer = EnhancedComprehensiveScorer(project_root)
    
    # 确定要处理的视频列表
    if args.video:
        videos = [args.video]
    else:
        # 处理所有有骨架线数据的视频
        hand_keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
        videos = []
        for keypoints_file in hand_keypoints_dir.glob("hand_keypoints_*.json"):
            video_name = keypoints_file.stem.replace("hand_keypoints_", "")
            videos.append(video_name)
    
    print("=" * 60)
    print("增强的综合评分系统（基于论文方法优化）")
    print("=" * 60)
    print(f"将处理 {len(videos)} 个视频")
    print("=" * 60)
    
    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "reports" / "enhanced_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个视频
    all_results = {}
    for i, video_name in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 处理视频: {video_name}")
        
        try:
            result = scorer.score_video(video_name)
            
            if 'error' in result:
                print(f"  [错误] {result['error']}")
                continue
            
            # 保存结果
            output_file = output_dir / f"enhanced_score_{video_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  [完成] 总帧数: {result.get('total_frames', 0)}")
            print(f"  DTW距离: {result.get('dtw_result', {}).get('distance', 0):.2f}")
            print(f"  DTW评分: {result.get('dtw_result', {}).get('score', 0):.2f}")
            print(f"  手部得分: {result.get('hand_score', 0):.2f}")
            print(f"  总分: {result.get('total_score', 0):.2f}")
            print(f"  结果已保存: {output_file}")
            
            all_results[video_name] = {
                'total_frames': result.get('total_frames', 0),
                'dtw_distance': result.get('dtw_result', {}).get('distance', 0),
                'dtw_score': result.get('dtw_result', {}).get('score', 0),
                'hand_score': result.get('hand_score', 0),
                'total_score': result.get('total_score', 0)
            }
        
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"汇总结果已保存: {summary_file}")


if __name__ == '__main__':
    main()
