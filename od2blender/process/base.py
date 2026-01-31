from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from od2blender.video_meta import VideoMeta


@dataclass(frozen=True)
class ProcessorContext:
    video_meta: VideoMeta
    settings: dict
    log_warn: Callable[[str], None] | None = None


class FrameProcessor:
    id: str = ""

    def process(self, frames: list[dict], context: ProcessorContext) -> list[dict]:
        raise NotImplementedError

    def describe(self, context: ProcessorContext) -> dict | None:
        return {"id": self.id}
