from __future__ import annotations

from od2blender.process.base import FrameProcessor, ProcessorContext


def run_processors(
    frames: list[dict],
    processors: list[FrameProcessor],
    context: ProcessorContext,
) -> tuple[list[dict], list[dict]]:
    processing_meta: list[dict] = []
    current_frames = frames
    for processor in processors:
        current_frames = processor.process(current_frames, context)
        description = processor.describe(context)
        if description:
            processing_meta.append(description)
    return current_frames, processing_meta
