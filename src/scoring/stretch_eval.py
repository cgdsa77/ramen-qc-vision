import cv2
import numpy as np
from typing import Dict, Any


def compute_motion_scores(frames, sample_stride: int = 2) -> Dict[str, float]:
    """Compute simple motion stats to gauge stretch continuity."""
    prev = None
    diffs = []
    for idx, frame in enumerate(frames):
        if idx % sample_stride != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            diffs.append(float(diff.mean()))
        prev = gray
    if not diffs:
        return {"motion_mean": 0.0, "motion_std": 0.0}
    return {
        "motion_mean": float(np.mean(diffs)),
        "motion_std": float(np.std(diffs)),
    }


def score_stretch(det_presence: float, motion_mean: float, motion_std: float) -> Dict[str, Any]:
    """Heuristic stretch score combining detection presence and motion stability."""
    presence_score = max(0.0, min(1.0, det_presence))
    # Lower motion_std implies steadier motion; soft normalization
    motion_stability = max(0.0, 1.0 - (motion_std / 25.0))
    final = 0.6 * presence_score + 0.4 * motion_stability
    return {
        "presence_score": round(presence_score, 3),
        "motion_stability": round(motion_stability, 3),
        "stretch_score": round(final, 3),
        "motion_mean": round(motion_mean, 3),
        "motion_std": round(motion_std, 3),
    }
