#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下面及捞面（xl 视频）骨架线数据检查与修复
- 检查：统计缺失帧、低置信度帧、异常关键点（越界等）
- 修复：对缺失帧做保守时间线性插值补全，提高召回率；插值范围与置信度折扣严格控制，不牺牲精确度
- 与抻面 generate_video_with_skeleton 的插值逻辑对齐，但参数更保守
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
keypoints_dir = project_root / "data" / "scores" / "下面及捞面" / "hand_keypoints"

# 保守插值参数（不牺牲精确度）
MAX_INTERPOLATE_RANGE_TWO_SIDED = 30   # 前后都有有效帧时，最大间隔帧数
MAX_INTERPOLATE_RANGE_ONE_SIDED = 12  # 仅前或仅后时，最大间隔帧数
MIN_KEYPOINTS_PER_HAND = 21
# 插值置信度：根据到最近有效帧的距离打折，避免远距离插值被当作高置信
CONFIDENCE_DECAY_PER_FRAME = 0.03   # 每帧距离降低的置信度
MIN_INTERPOLATED_CONFIDENCE = 0.25


def _confidence_discount(distance: int, two_sided: bool) -> float:
    """根据距离计算置信度折扣，保证不虚高。"""
    if two_sided:
        # 总距离 = prev_d + next_d，取一半作为“等效距离”
        effective = distance / 2.0
    else:
        effective = float(distance)
    decay = min(0.7, effective * CONFIDENCE_DECAY_PER_FRAME)
    return max(MIN_INTERPOLATED_CONFIDENCE, 1.0 - decay)


def _check_frame_quality(frame_data: Dict, width: int, height: int) -> Tuple[bool, List[str]]:
    """
    检查单帧质量：是否有效、是否有异常。
    返回 (is_valid, list of issue descriptions)。
    """
    issues = []
    hands = frame_data.get("hands") or []
    if not hands:
        return False, ["无手部数据"]

    for hi, hand in enumerate(hands):
        kps = hand.get("keypoints") or []
        if len(kps) < MIN_KEYPOINTS_PER_HAND:
            issues.append(f"手{hi}关键点数不足: {len(kps)}")
            continue
        confs = [kp.get("confidence", 0) for kp in kps]
        avg_conf = sum(confs) / len(confs)
        if avg_conf < 0.2:
            issues.append(f"手{hi}平均置信度过低: {avg_conf:.3f}")
        out_of_bounds = 0
        for kp in kps:
            x, y = kp.get("x", 0), kp.get("y", 0)
            if x < -0.05 * width or x > width * 1.05 or y < -0.05 * height or y > height * 1.05:
                out_of_bounds += 1
        if out_of_bounds > 2:
            issues.append(f"手{hi}有{out_of_bounds}个关键点越界")

    is_valid = len(hands) > 0 and all(
        (h.get("keypoints") or []) and len(h.get("keypoints", [])) >= MIN_KEYPOINTS_PER_HAND
        for h in hands
    )
    return is_valid, issues


def interpolate_missing_frames(
    frames_list: List[Dict],
    total_frames: int,
    width: int,
    height: int,
) -> Tuple[List[Dict], int]:
    """
    对缺失帧做保守插值补全。
    返回 (新的 frames 列表, 补全的帧数)。
    """
    frame_map: Dict[int, Dict] = {}
    for fd in frames_list:
        idx = fd.get("frame_index", len(frame_map))
        frame_map[idx] = fd.copy()

    valid_frames: Dict[int, Dict] = {}
    for idx, fd in frame_map.items():
        if fd.get("hands") and len(fd["hands"]) > 0:
            valid = all(
                (h.get("keypoints") or []) and len(h.get("keypoints", [])) >= MIN_KEYPOINTS_PER_HAND
                for h in fd["hands"]
            )
            if valid:
                valid_frames[idx] = fd

    filled_count = 0
    result: List[Dict] = []

    for frame_idx in range(total_frames):
        current = frame_map.get(frame_idx)
        has_valid = (
            current
            and current.get("hands")
            and len(current["hands"]) > 0
            and all(
                len((h.get("keypoints") or [])) >= MIN_KEYPOINTS_PER_HAND
                for h in current["hands"]
            )
        )

        if has_valid:
            result.append(current)
            continue

        # 向前找最近有效帧
        prev_frame = None
        prev_dist = None
        for i in range(frame_idx - 1, max(-1, frame_idx - MAX_INTERPOLATE_RANGE_TWO_SIDED - 1), -1):
            if i in valid_frames:
                prev_frame = valid_frames[i]
                prev_dist = frame_idx - i
                break

        # 向后找最近有效帧
        next_frame = None
        next_dist = None
        for i in range(frame_idx + 1, min(total_frames, frame_idx + MAX_INTERPOLATE_RANGE_TWO_SIDED + 1)):
            if i in valid_frames:
                next_frame = valid_frames[i]
                next_dist = i - frame_idx
                break

        interpolated_hands = None

        if prev_frame and next_frame and prev_dist is not None and next_dist is not None:
            if prev_dist <= MAX_INTERPOLATE_RANGE_TWO_SIDED and next_dist <= MAX_INTERPOLATE_RANGE_TWO_SIDED:
                total_d = prev_dist + next_dist
                alpha = prev_dist / total_d
                discount = _confidence_discount(total_d, two_sided=True)
                prev_hands = prev_frame.get("hands", [])
                next_hands = next_frame.get("hands", [])
                interpolated_hands = []
                max_hands = max(len(prev_hands), len(next_hands))
                for hand_idx in range(max_hands):
                    if hand_idx < len(prev_hands) and hand_idx < len(next_hands):
                        ph, nh = prev_hands[hand_idx], next_hands[hand_idx]
                        pkps = ph.get("keypoints", [])
                        nkps = nh.get("keypoints", [])
                        n_kp = min(len(pkps), len(nkps))
                        if n_kp < MIN_KEYPOINTS_PER_HAND:
                            continue
                        new_kps = []
                        for k in range(n_kp):
                            pk, nk = pkps[k], nkps[k]
                            c_prev = pk.get("confidence", 0.5)
                            c_next = nk.get("confidence", 0.5)
                            new_conf = (c_prev * (1 - alpha) + c_next * alpha) * discount
                            new_conf = max(MIN_INTERPOLATED_CONFIDENCE, min(1.0, new_conf))
                            new_kps.append({
                                "x": pk["x"] * (1 - alpha) + nk["x"] * alpha,
                                "y": pk["y"] * (1 - alpha) + nk["y"] * alpha,
                                "z": pk.get("z", 0) * (1 - alpha) + nk.get("z", 0) * alpha,
                                "confidence": new_conf,
                            })
                        if len(new_kps) >= MIN_KEYPOINTS_PER_HAND:
                            interpolated_hands.append({"id": ph.get("id", hand_idx), "keypoints": new_kps})

        if not interpolated_hands and prev_frame and prev_dist is not None and prev_dist <= MAX_INTERPOLATE_RANGE_ONE_SIDED:
            discount = _confidence_discount(prev_dist, two_sided=False)
            interpolated_hands = []
            for hand in prev_frame.get("hands", []):
                kps = hand.get("keypoints", [])
                if len(kps) < MIN_KEYPOINTS_PER_HAND:
                    continue
                new_kps = [
                    {**kp, "confidence": min(1.0, (kp.get("confidence", 0.5) * discount))}
                    for kp in kps
                ]
                interpolated_hands.append({"id": hand.get("id", 0), "keypoints": new_kps})

        if not interpolated_hands and next_frame and next_dist is not None and next_dist <= MAX_INTERPOLATE_RANGE_ONE_SIDED:
            discount = _confidence_discount(next_dist, two_sided=False)
            interpolated_hands = []
            for hand in next_frame.get("hands", []):
                kps = hand.get("keypoints", [])
                if len(kps) < MIN_KEYPOINTS_PER_HAND:
                    continue
                new_kps = [
                    {**kp, "confidence": min(1.0, (kp.get("confidence", 0.5) * discount))}
                    for kp in kps
                ]
                interpolated_hands.append({"id": hand.get("id", 0), "keypoints": new_kps})

        if interpolated_hands:
            result.append({
                "frame_index": frame_idx,
                "hands": interpolated_hands,
                "interpolated": True,
            })
            filled_count += 1
        else:
            # 保持原样（缺失或无效）
            if current is not None:
                result.append(current)
            else:
                result.append({"frame_index": frame_idx, "hands": []})

    return result, filled_count


def process_video(video_name: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    检查并修复单个视频的 hand_keypoints JSON。
    dry_run 为 True 时只统计不写回。
    返回统计信息。
    """
    path = keypoints_dir / f"hand_keypoints_{video_name}.json"
    if not path.exists():
        return {"error": "文件不存在", "video": video_name}

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    total_frames = data.get("total_frames", 0)
    width = data.get("width", 1280)
    height = data.get("height", 720)
    frames_list = data.get("frames", [])

    if total_frames == 0 and frames_list:
        total_frames = len(frames_list)

    # 检查
    detected_before = 0
    missing_before = 0
    low_confidence_frames = 0
    issue_frames: List[int] = []

    for fd in frames_list:
        idx = fd.get("frame_index", detected_before + missing_before)
        valid, issues = _check_frame_quality(fd, width, height)
        if valid:
            detected_before += 1
            hands = fd.get("hands", [])
            for h in hands:
                confs = [kp.get("confidence", 0) for kp in (h.get("keypoints") or [])]
                if confs and sum(confs) / len(confs) < 0.35:
                    low_confidence_frames += 1
                    break
        else:
            missing_before += 1
            if issues:
                issue_frames.append(idx)

    # 修复：插值补全缺失帧
    new_frames, filled_count = interpolate_missing_frames(frames_list, total_frames, width, height)

    detected_after = sum(1 for fd in new_frames if fd.get("hands") and len(fd["hands"]) > 0)
    missing_after = total_frames - detected_after

    stats = {
        "video": video_name,
        "total_frames": total_frames,
        "detected_before": detected_before,
        "missing_before": missing_before,
        "filled_count": filled_count,
        "detected_after": detected_after,
        "missing_after": missing_after,
        "low_confidence_frames": low_confidence_frames,
        "issue_frame_count": len(issue_frames),
    }

    if not dry_run and filled_count > 0:
        data["frames"] = new_frames
        data["detected_frames"] = detected_after
        data["missing_frames"] = missing_after
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        stats["written"] = True
    else:
        stats["written"] = False

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下面及捞面 xl 视频骨架线数据检查与修复（保守插值，不牺牲精确度）")
    parser.add_argument("--video", type=str, default=None, help="仅处理指定视频，如 xl1")
    parser.add_argument("--dry-run", action="store_true", help="仅检查与统计，不写回文件")
    args = parser.parse_args()

    if not keypoints_dir.exists():
        print(f"[错误] 目录不存在: {keypoints_dir}")
        return

    if args.video:
        files = [keypoints_dir / f"hand_keypoints_{args.video}.json"]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(keypoints_dir.glob("hand_keypoints_xl*.json"))

    if not files:
        print("未找到任何 hand_keypoints_xl*.json 文件")
        return

    print("=" * 60)
    print("下面及捞面 xl 骨架线数据检查与修复")
    print("=" * 60)
    if args.dry_run:
        print("（仅检查，不写回）")
    print()

    total_filled = 0
    for path in files:
        video_name = path.stem.replace("hand_keypoints_", "")
        stats = process_video(video_name, dry_run=args.dry_run)
        if stats.get("error"):
            print(f"  {video_name}: {stats['error']}")
            continue
        total_filled += stats.get("filled_count", 0)
        rate_before = (stats["detected_before"] / stats["total_frames"] * 100) if stats["total_frames"] else 0
        rate_after = (stats["detected_after"] / stats["total_frames"] * 100) if stats["total_frames"] else 0
        print(f"  {video_name}: 总帧 {stats['total_frames']}, "
              f"修复前检测 {stats['detected_before']} ({rate_before:.1f}%), "
              f"补全 {stats['filled_count']} 帧, "
              f"修复后 {stats['detected_after']} ({rate_after:.1f}%)"
              + (" [已写回]" if stats.get("written") else ""))

    print()
    print(f"共补全 {total_filled} 帧。")
    if args.dry_run and total_filled > 0:
        print("去掉 --dry-run 将写回修复后的 JSON。")


if __name__ == "__main__":
    main()
