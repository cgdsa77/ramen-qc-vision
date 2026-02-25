"""
完整处理抻面部分的所有视频综合评分
流程：先用当前最佳抻面模型对 cm1~cm12 做检测得分，再与骨架线、DTW 等融合得到综合分与细分，
结果仍由抻面综合评分可视化系统展示。
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.enhanced_comprehensive_scorer import EnhancedComprehensiveScorer
from src.scoring.detection_to_annotation import resolve_stretch_video_path, build_annotation_from_detection


def _get_skeleton_frame_count(project_root: Path, video_name: str) -> int:
    """读取骨架线文件得到帧数，用于与检测帧对齐。"""
    keypoints_file = project_root / "data" / "scores" / "抻面" / "hand_keypoints" / f"hand_keypoints_{video_name}.json"
    if not keypoints_file.exists():
        return 0
    try:
        with open(keypoints_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("frames", []))
    except Exception:
        return 0


def main():
    """主函数：最佳模型检测 + 骨架线/DTW 融合。支持 --video 仅处理单个视频（不重跑其余）。"""
    import argparse
    parser = argparse.ArgumentParser(description="抻面综合评分：最佳模型+骨架线+DTW 融合")
    parser.add_argument("--video", type=str, default=None, help="仅处理指定视频（如 cm8），不重跑其余已跑过的")
    args = parser.parse_args()
    
    print("=" * 60)
    print("抻面综合评分系统 - 最佳模型检测 + 骨架线/DTW 融合")
    print("=" * 60)
    
    scorer = EnhancedComprehensiveScorer(project_root)
    hand_keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    videos = []
    for keypoints_file in hand_keypoints_dir.glob("hand_keypoints_*.json"):
        video_name = keypoints_file.stem.replace("hand_keypoints_", "")
        videos.append(video_name)
    videos = sorted(videos)
    if args.video:
        if args.video not in videos:
            print(f"[错误] 未找到 --video {args.video} 的骨架线数据，请先运行: python scripts/extract_hand_keypoints_from_video.py --video {args.video}")
            return
        videos = [args.video]
        print(f"\n仅处理: {args.video}（不重跑其余视频）")
    print(f"\n找到 {len(videos)} 个视频（有骨架线数据）")
    keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    missing = [f"cm{i}" for i in range(1, 13) if not (keypoints_dir / f"hand_keypoints_cm{i}.json").exists()]
    if missing:
        print(f"  说明：骨架线来自 data/scores/抻面/hand_keypoints/，当前缺少 {missing}，如需 12 个请先生成并放入该目录。")
    print("=" * 60)
    
    output_dir = project_root / "reports" / "comprehensive_scores_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    success_count = 0
    model_source_used = None
    
    for i, video_name in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 处理视频: {video_name}")
        
        try:
            video_path = resolve_stretch_video_path(video_name, project_root)
            if not video_path or not video_path.exists():
                result = scorer.score_video(video_name)
                if "error" in result:
                    print(f"  [错误] {result['error']}（且未找到本地视频，无法用最佳模型检测）")
                    all_results[video_name] = {"error": result["error"]}
                    continue
                print("  [说明] 未找到本地视频文件，使用原有骨架+DTW+关键帧标注融合")
            else:
                num_skeleton_frames = _get_skeleton_frame_count(project_root, video_name)
                max_frames = num_skeleton_frames if num_skeleton_frames > 0 else None
                annotation_scores, annotation_confidences, model_source_used = build_annotation_from_detection(
                    str(video_path), project_root=project_root, max_frames=max_frames
                )
                print("  正在与骨架线、DTW 融合计算综合分...")
                result = scorer.calculate_comprehensive_score(
                    video_name, annotation_scores, annotation_confidences
                )
                if "error" in result:
                    print(f"  [错误] {result['error']}")
                    all_results[video_name] = {"error": result["error"]}
                    continue
                result["model_source"] = model_source_used
                result["score_basis"] = "最佳抻面模型检测 + 骨架线 + DTW 融合"
            
            output_file = output_dir / f"comprehensive_score_{video_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  [完成] 总帧数: {result.get('total_frames', 0)}")
            print(f"  评分帧数: {result.get('scored_frames', 0)}")
            dtw_result = result.get("dtw_result", {})
            print(f"  DTW距离: {dtw_result.get('distance', 0):.2f}")
            print(f"  DTW评分: {dtw_result.get('score', 0):.2f}")
            class_scores = result.get("class_average_scores", {})
            print(f"  手部: {class_scores.get('hand', 0):.2f}  面条: {class_scores.get('noodle_rope', 0):.2f}  面条束: {class_scores.get('noodle_bundle', 0):.2f}")
            print(f"  总分: {result.get('total_score', 0):.2f}")
            print(f"  结果已保存: {output_file}")
            
            all_results[video_name] = {
                "total_frames": result.get("total_frames", 0),
                "scored_frames": result.get("scored_frames", 0),
                "dtw_distance": dtw_result.get("distance", 0),
                "dtw_score": dtw_result.get("score", 0),
                "hand_score": class_scores.get("hand", 0),
                "noodle_rope_score": class_scores.get("noodle_rope", 0),
                "noodle_bundle_score": class_scores.get("noodle_bundle", 0),
                "total_score": result.get("total_score", 0),
            }
            success_count += 1
        
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[video_name] = {"error": str(e)}
    
    # 保存汇总结果（若 --video 只处理了单个，则合并进已有 summary，不覆盖其余）
    summary_file = output_dir / "summary.json"
    if args.video and summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_results = old.get("results") or {}
            old_results.update(all_results)
            all_results = old_results
        except Exception:
            pass
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_videos": len(all_results),
            "success_count": sum(1 for r in all_results.values() if "error" not in r),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    successful_results = [r for r in all_results.values() if "error" not in r]
    if successful_results:
        avg_total_score = sum(r.get('total_score', 0) for r in successful_results) / len(successful_results)
        avg_hand_score = sum(r.get('hand_score', 0) for r in successful_results) / len(successful_results)
        avg_rope_score = sum(r.get('noodle_rope_score', 0) for r in successful_results) / len(successful_results)
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"成功处理: {len(successful_results)}/{len(all_results)} 个视频（汇总表已更新）")
        if args.video:
            print(f"  本次仅处理 {args.video}，其余结果保留不变。")
        print(f"\n平均得分（当前汇总内）:")
        print(f"  总分: {avg_total_score:.2f}")
        print(f"  手部: {avg_hand_score:.2f}")
        print(f"  面条: {avg_rope_score:.2f}")
        print(f"\n汇总结果已保存: {summary_file}")
    else:
        print("\n" + "=" * 60)
        print("汇总已更新（当前无成功结果或仅更新失败项）")
        print(f"  汇总结果已保存: {summary_file}")


if __name__ == "__main__":
    main()
