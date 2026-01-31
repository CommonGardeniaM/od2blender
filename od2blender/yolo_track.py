from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ultralytics import YOLO
from loguru import logger

from od2blender.tracks import build_empty_frames, dedupe_detections
from od2blender.video_meta import VideoMeta


class TrackingError(RuntimeError):
    pass


DEFAULT_MODEL_CANDIDATES = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]


def select_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


def load_model(
    candidates: Iterable[str],
    log_warn: Callable[[str], None] | None = None,
) -> tuple[YOLO, str]:
    last_error: Exception | None = None
    for name in candidates:
        try:
            model = YOLO(name)
            logger.info("Using model: {}", name)
            return model, name
        except Exception as exc:
            last_error = exc
            if log_warn:
                log_warn(f"Failed to load model {name}: {exc}")
            continue
    raise TrackingError("No usable YOLO model found") from last_error


def extract_detections(
    result,
    width: int,
    height: int,
    log_warn: Callable[[str], None] | None = None,
) -> list[dict]:
    detections: list[dict] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    ids = boxes.id
    xywh = boxes.xywh
    clss = boxes.cls
    confs = boxes.conf

    for idx in range(len(boxes)):
        track_id = None
        if ids is not None:
            track_id = int(ids[idx].item())
        if track_id is None:
            if log_warn:
                log_warn("Skipping detection without track id")
            continue
        cx, cy, w, h = [float(v) for v in xywh[idx].tolist()]
        u = cx / width if width else 0.0
        v = cy / height if height else 0.0
        cls_id = int(clss[idx].item()) if clss is not None else 0
        conf = float(confs[idx].item()) if confs is not None else 0.0
        detections.append(
            {
                "id": track_id,
                "cls": cls_id,
                "conf": conf,
                "xywh": [cx, cy, w, h],
                "center_norm": [u, v],
            }
        )
    return detections


def run_tracking(
    video_path: Path,
    video_meta: VideoMeta,
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    model_candidates: Sequence[str] | None = None,
    log_warn: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    device = select_device(device)
    candidates = model_candidates or DEFAULT_MODEL_CANDIDATES
    model, model_name = load_model(candidates, log_warn=log_warn)

    frames = build_empty_frames(video_meta.frame_count, video_meta.fps)

    try:
        results = model.track(
            source=str(video_path),
            tracker=tracker,
            persist=True,
            stream=True,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for frame_index, result in enumerate(results):
            if frame_index >= len(frames):
                if log_warn:
                    log_warn(
                        f"Received extra frame {frame_index}; ignoring beyond frame_count"
                    )
                break
            detections = extract_detections(
                result,
                width=video_meta.width,
                height=video_meta.height,
                log_warn=log_warn,
            )
            frames[frame_index]["detections"] = dedupe_detections(
                detections,
                frame_index=frame_index,
                log_fn=log_warn,
            )
    except Exception as exc:
        raise TrackingError("Ultralytics tracking failed") from exc

    detector_meta = {
        "backend": "ultralytics",
        "model": model_name,
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "device": device,
    }
    return frames, detector_meta
