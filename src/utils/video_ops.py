from typing import Any, Dict
import cv2


class Segmenter:
    def __init__(self, cfg: Dict[str, Any]):
        self.segment_seconds = cfg["input"].get("segment_seconds", 5)
        self.fps = cfg["input"].get("frame_rate", 15)

    def split(self, video) -> list:
        frames_per_seg = int(self.segment_seconds * self.fps)
        segments = []
        buffer = []
        for idx, frame in enumerate(video):
            buffer.append(frame)
            if len(buffer) >= frames_per_seg:
                segments.append(FrameSegment(buffer, idx // frames_per_seg))
                buffer = []
        if buffer:
            segments.append(FrameSegment(buffer, len(segments)))
        return segments


class FrameSegment:
    def __init__(self, frames, seg_id):
        self.frames = frames
        self.id = seg_id

    def __iter__(self):
        for f in self.frames:
            yield f
