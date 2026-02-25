"""
下面及捞面综合评分：批量处理所有 xl 视频
结合骨架线与标注数据，输出与抻面一致的报告结构，供 Web 可视化使用。
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.boiling_comprehensive_scorer import BoilingComprehensiveScorer


def main():
    print("=" * 60)
    print("下面及捞面综合评分 - 批量处理")
    print("=" * 60)

    scorer = BoilingComprehensiveScorer(project_root)
    scores_base = project_root / "data" / "scores" / "下面及捞面"
    videos = sorted([d.name for d in scores_base.iterdir() if d.is_dir() and d.name.startswith("xl")])
    # 仅保留有骨架数据的
    kp_dir = scores_base / "hand_keypoints"
    videos = [v for v in videos if (kp_dir / f"hand_keypoints_{v}.json").exists()]

    print(f"\n找到 {len(videos)} 个视频（有骨架线且为 xl*）")
    print("=" * 60)

    output_dir = project_root / "reports" / "comprehensive_scores_boiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    success_count = 0

    for i, video_name in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_name}")
        try:
            result = scorer.score_video(video_name)
            if result.get("error"):
                print(f"  [错误] {result['error']}")
                all_results[video_name] = {"error": result["error"]}
                continue
            out_file = output_dir / f"comprehensive_score_{video_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  总分: {result.get('total_score', 0):.2f}  手部: {result.get('hand_score', 0):.2f}  面条: {result.get('noodle_rope_score', 0):.2f}  操作: {result.get('noodle_bundle_score', 0):.2f}")
            print(f"  已保存: {out_file}")
            all_results[video_name] = {
                "total_frames": result.get("total_frames", 0),
                "scored_frames": result.get("scored_frames", 0),
                "hand_score": result.get("hand_score", 0),
                "noodle_rope_score": result.get("noodle_rope_score", 0),
                "noodle_bundle_score": result.get("noodle_bundle_score", 0),
                "total_score": result.get("total_score", 0),
            }
            success_count += 1
        except Exception as e:
            print(f"  [异常] {e}")
            import traceback
            traceback.print_exc()
            all_results[video_name] = {"error": str(e)}

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_videos": len(videos), "success_count": success_count, "results": all_results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功: {success_count}/{len(videos)}")
    print(f"汇总: {summary_file}")


if __name__ == "__main__":
    main()
