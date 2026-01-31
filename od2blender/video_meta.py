from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoMetaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }


def read_video_meta(video_path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoMetaError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise VideoMetaError(
            "Invalid video metadata: "
            f"fps={fps}, width={width}, height={height}, frame_count={frame_count}"
        )

    return VideoMeta(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )
