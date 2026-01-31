from __future__ import annotations


def build_empty_frames(frame_count: int, fps: float) -> list[dict]:
    frames: list[dict] = []
    for i in range(frame_count):
        t = i / fps if fps else 0.0
        frames.append({"i": i, "t": t, "detections": [], "poses3d": []})
    return frames


def normalize_frames(frames: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        detections = frame.get("detections")
        if not isinstance(detections, list):
            detections = []
        poses3d = frame.get("poses3d")
        if not isinstance(poses3d, list):
            poses3d = []
        payload = {
            "i": int(frame.get("i", 0)),
            "t": float(frame.get("t", 0.0)),
            "detections": detections,
            "poses3d": poses3d,
        }
        poses3d_raw = frame.get("poses3d_raw")
        if isinstance(poses3d_raw, list):
            payload["poses3d_raw"] = poses3d_raw
        extra = frame.get("extra")
        if isinstance(extra, dict) and extra:
            payload["extra"] = dict(extra)
        normalized.append(payload)
    return normalized
