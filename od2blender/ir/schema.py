from __future__ import annotations

IR_VERSION = "od2blender.ir.v1"
SCHEMA_NAME = "od2blender.tracks"
SCHEMA_VERSION = 1


def build_ir_payload(
    video_meta: dict,
    detector_meta: dict,
    frames: list[dict],
    *,
    mode: str,
    skeleton: dict | None = None,
    units: dict | None = None,
    axis: str | None = None,
    processing: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "version": IR_VERSION,
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "mode": mode,
        "video": video_meta,
        "detector": detector_meta,
        "frames": frames,
    }
    if skeleton is not None:
        payload["skeleton"] = skeleton
    if units is not None:
        payload["units"] = units
    if axis is not None:
        payload["axis"] = axis
    if processing:
        payload["processing"] = processing
    if extra:
        payload["extra"] = extra
    return payload
