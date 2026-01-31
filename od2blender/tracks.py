from __future__ import annotations

from od2blender.ir.frames import build_empty_frames
from od2blender.ir.schema import IR_VERSION, build_ir_payload
from od2blender.json_utils import save_json
from od2blender.paths import make_blender_relpath
from od2blender.process.dedupe import dedupe_detections

TRACKS_VERSION = IR_VERSION


def build_tracks_payload(
    video_meta: dict,
    detector_meta: dict,
    frames: list[dict],
    mode: str | None = None,
    skeleton: dict | None = None,
    units: dict | None = None,
    axis: str | None = None,
    processing: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    return build_ir_payload(
        video_meta=video_meta,
        detector_meta=detector_meta,
        frames=frames,
        mode=mode or "unknown",
        skeleton=skeleton,
        units=units,
        axis=axis,
        processing=processing,
        extra=extra,
    )


__all__ = [
    "TRACKS_VERSION",
    "build_empty_frames",
    "build_tracks_payload",
    "dedupe_detections",
    "make_blender_relpath",
    "save_json",
]
