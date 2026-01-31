from __future__ import annotations

from od2blender.process.base import FrameProcessor
from od2blender.process.dedupe import DedupeDetectionsProcessor
from od2blender.process.pose3d_projector import Pose3DProjectorProcessor


_PROCESSORS: dict[str, FrameProcessor] = {
    DedupeDetectionsProcessor.id: DedupeDetectionsProcessor(),
    Pose3DProjectorProcessor.id: Pose3DProjectorProcessor(),
}


def get_processor(processor_id: str) -> FrameProcessor | None:
    return _PROCESSORS.get(processor_id)


def list_processors() -> list[str]:
    return sorted(_PROCESSORS.keys())
