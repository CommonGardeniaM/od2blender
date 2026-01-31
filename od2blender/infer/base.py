from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path

from od2blender.video_meta import VideoMeta


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceResult:
    frames: list[dict]
    detector_meta: dict
    mode: str
    skeleton: dict | None = None
    units: dict | None = None
    axis: str | None = None
    extra: dict | None = None


class InferenceBackend:
    id: str = ""
    mode: str = ""
    default_processors: list[str] = []

    def infer(
        self,
        video_path: Path,
        video_meta: VideoMeta,
        config: dict,
        log_warn: Callable[[str], None] | None = None,
    ) -> InferenceResult:
        raise NotImplementedError
