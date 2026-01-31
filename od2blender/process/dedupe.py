from __future__ import annotations

from typing import Callable, Iterable

from od2blender.process.base import FrameProcessor, ProcessorContext


def dedupe_detections(
    detections: Iterable[dict],
    frame_index: int,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    best_by_id: dict[int, dict] = {}
    for det in detections:
        det_id = int(det["id"])
        if det_id in best_by_id:
            if det.get("conf", 0.0) > best_by_id[det_id].get("conf", 0.0):
                best_by_id[det_id] = det
            if log_fn:
                log_fn(
                    f"duplicate track id {det_id} at frame {frame_index}; keeping highest conf"
                )
        else:
            best_by_id[det_id] = det
    return list(best_by_id.values())


class DedupeDetectionsProcessor(FrameProcessor):
    id = "dedupe_detections"

    def process(self, frames: list[dict], context: ProcessorContext) -> list[dict]:
        for frame in frames:
            detections = frame.get("detections") or []
            frame_index = int(frame.get("i", 0))
            frame["detections"] = dedupe_detections(
                detections,
                frame_index=frame_index,
                log_fn=context.log_warn,
            )
        return frames
