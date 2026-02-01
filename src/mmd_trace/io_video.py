"""Video I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: int
    frame_count: int
    width: int
    height: int


def iter_video_frames(
    path: str,
    fps_override: Optional[int] = None,
) -> Tuple[VideoInfo, Iterator[Tuple[int, float, np.ndarray]]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or unreadable: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps_override:
        fps = fps_override

    info = VideoInfo(fps=int(round(fps)), frame_count=frame_count, width=width, height=height)

    def _iter() -> Iterator[Tuple[int, float, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / info.fps if info.fps > 0 else 0.0
            yield idx, t, frame
            idx += 1
        cap.release()

    return info, _iter()
