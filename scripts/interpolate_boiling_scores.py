#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下面及捞面：基于关键帧时间线性插值生成非关键帧评分
- 读取每个视频的 key_frames.json 与关键帧 *_scores.json
- 对非关键帧按帧号在前后关键帧之间做线性插值（规则与抻面一致：间隔过大×0.8，仅一侧×0.9）
- 写回 {frame}_scores.json，并标记 "interpolated": true
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
scores_base = project_root / "data" / "scores" / "下面及捞面"
processed_base = project_root / "data" / "processed" / "下面及捞面"

# 与抻面一致：关键帧间隔超过此帧数时插值分×0.8
KEYFRAME_GAP_FRAMES_SPARSE = 150


def _frame_name_to_index(frame_name: str) -> Optional[int]:
    """xl1_00025.jpg 或 xl1_00025 -> 25"""
    base = frame_name.replace(".jpg", "").strip()
    parts = base.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _aggregate_attributes_per_frame(scores_dict: Dict[str, Any]) -> Dict[str, float]:
    """从单帧 scores 中按属性聚合：每个属性取该帧内所有 detection 的平均值（忽略 notes）。"""
    by_attr: Dict[str, List[float]] = {}
    for det_key, det_val in (scores_dict or {}).items():
        if not isinstance(det_val, dict):
            continue
        for k, v in det_val.items():
            if k == "notes":
                continue
            try:
                x = float(v)
                if 1 <= x <= 5:
                    by_attr.setdefault(k, []).append(x)
            except (TypeError, ValueError):
                pass
    return {k: sum(v) / len(v) for k, v in by_attr.items() if v}


def _interpolate_value(
    frame_index: int,
    keyframe_indices: List[int],
    keyframe_attr_values: Dict[int, float],
    keyframe_gap_frames: int = KEYFRAME_GAP_FRAMES_SPARSE,
) -> Optional[float]:
    """对单个属性在 frame_index 处做线性插值。"""
    if not keyframe_indices or frame_index in keyframe_attr_values:
        return keyframe_attr_values.get(frame_index)

    sorted_kf = sorted(keyframe_attr_values.keys())
    prev_frame = next_frame = None
    for kf in sorted_kf:
        if kf <= frame_index:
            prev_frame = kf
        if kf > frame_index:
            next_frame = kf
            break

    prev_score = keyframe_attr_values.get(prev_frame) if prev_frame is not None else None
    next_score = keyframe_attr_values.get(next_frame) if next_frame is not None else None

    interpolated = None
    gap_penalty = 1.0
    if prev_score is not None and next_score is not None:
        alpha = (frame_index - prev_frame) / (next_frame - prev_frame) if next_frame != prev_frame else 0
        interpolated = prev_score * (1 - alpha) + next_score * alpha
        if (next_frame - prev_frame) > keyframe_gap_frames:
            gap_penalty = 0.8
    elif prev_score is not None:
        interpolated = prev_score * 0.9
        if prev_frame is not None and (frame_index - prev_frame) > keyframe_gap_frames:
            gap_penalty = 0.8
    elif next_score is not None:
        interpolated = next_score * 0.9
        if next_frame is not None and (next_frame - frame_index) > keyframe_gap_frames:
            gap_penalty = 0.8

    if interpolated is not None:
        return float(max(1.0, min(5.0, round(interpolated * gap_penalty, 2))))
    return None


def load_keyframes_and_scores(video_name: str) -> tuple[List[int], Dict[int, Dict[str, float]]]:
    """返回 (关键帧索引列表, 关键帧索引 -> 属性名 -> 聚合分数)。"""
    key_frames_file = scores_base / video_name / "key_frames.json"
    keyframe_indices: List[int] = []
    keyframe_scores: Dict[int, Dict[str, float]] = {}

    if key_frames_file.exists():
        with open(key_frames_file, "r", encoding="utf-8") as f:
            for item in json.load(f):
                frame_name = (item.get("frame") or "").replace(".jpg", "").strip()
                idx = item.get("index")
                if idx is None:
                    idx = _frame_name_to_index(frame_name)
                if idx is None:
                    continue
                keyframe_indices.append(idx)
                score_file = scores_base / video_name / (frame_name + "_scores.json")
                if score_file.exists():
                    try:
                        with open(score_file, "r", encoding="utf-8-sig") as sf:
                            data = json.load(sf)
                        keyframe_scores[idx] = _aggregate_attributes_per_frame(data.get("scores", {}))
                    except Exception:
                        pass

    return keyframe_indices, keyframe_scores


def get_all_attributes(keyframe_scores: Dict[int, Dict[str, float]]) -> Set[str]:
    """所有关键帧中出现过的属性名（排除 notes）。"""
    out: Set[str] = set()
    for attrs in keyframe_scores.values():
        out.update(k for k in attrs if k != "notes")
    return out


def interpolate_video(video_name: str, overwrite_inferred_only: bool = True) -> int:
    """
    对单个视频做插值：非关键帧写入插值后的 *_scores.json。
    overwrite_inferred_only: 若 True，仅覆盖此前为 inferred 的 JSON；否则覆盖所有非关键帧。
    返回本视频写入的插值帧数。
    """
    processed_dir = processed_base / video_name
    scores_dir = scores_base / video_name
    if not processed_dir.exists() or not scores_dir.exists():
        return 0

    keyframe_indices, keyframe_scores = load_keyframes_and_scores(video_name)
    keyframe_set: Set[int] = set(keyframe_scores.keys())
    if not keyframe_scores:
        return 0

    all_attrs = get_all_attributes(keyframe_scores)
    if not all_attrs:
        return 0

    # 每个属性：关键帧索引 -> 值
    attr_to_keyframe_values: Dict[str, Dict[int, float]] = {
        attr: {kf: keyframe_scores[kf][attr] for kf in keyframe_scores if attr in keyframe_scores[kf]}
        for attr in all_attrs
    }

    written = 0
    for jp in sorted(processed_dir.glob("*.jpg")):
        base = jp.stem
        frame_index = _frame_name_to_index(base)
        if frame_index is None or frame_index in keyframe_set:
            continue

        score_file = scores_dir / (base + "_scores.json")
        if score_file.exists() and overwrite_inferred_only:
            try:
                with open(score_file, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                if data.get("interpolated") is not True and data.get("inferred") is not True:
                    continue
            except Exception:
                pass

        interpolated_attrs: Dict[str, float] = {}
        for attr in all_attrs:
            val = _interpolate_value(
                frame_index,
                sorted(attr_to_keyframe_values[attr].keys()),
                attr_to_keyframe_values[attr],
            )
            if val is not None:
                interpolated_attrs[attr] = val

        if not interpolated_attrs:
            continue

        frame_name = base + ".jpg"
        out = {
            "video": video_name,
            "frame": frame_name,
            "scores": {
                "detection_0": {**interpolated_attrs, "notes": "关键帧时间线性插值"}
            },
            "stage": "boiling_scooping",
            "interpolated": True,
        }
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        written += 1

    return written


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下面及捞面：基于关键帧时间线性插值生成非关键帧评分")
    parser.add_argument("--video", type=str, default=None, help="仅处理指定视频（如 xl1）；默认处理所有 xl*")
    parser.add_argument("--overwrite-all", action="store_true", help="覆盖所有非关键帧（默认仅覆盖此前 inferred/interpolated 的）")
    args = parser.parse_args()

    if args.video:
        video_dirs = [processed_base / args.video] if (processed_base / args.video).exists() else []
    else:
        video_dirs = sorted([d for d in processed_base.iterdir() if d.is_dir() and d.name.startswith("xl")])

    total = 0
    for d in video_dirs:
        video_name = d.name
        n = interpolate_video(video_name, overwrite_inferred_only=not args.overwrite_all)
        if n > 0:
            print(f"  {video_name}: 写入 {n} 帧插值评分")
        total += n

    print(f"共写入 {total} 帧插值评分。")


if __name__ == "__main__":
    main()
