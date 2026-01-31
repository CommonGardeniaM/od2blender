from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from od2blender.infer.base import InferenceError
from od2blender.ir.frames import build_empty_frames
from od2blender.video_meta import VideoMeta


class Pose3DError(InferenceError):
    pass


LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

POSE3D_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE3D_MODEL_NAME = "pose_landmarker_lite.task"


MEDIAPIPE_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

MEDIAPIPE_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


@dataclass(frozen=True)
class Pose3DConfig:
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Path | None = None


def get_pose3d_skeleton() -> dict:
    return {
        "name": "mediapipe_pose_v1",
        "landmarks": list(MEDIAPIPE_LANDMARKS),
        "bones": [[int(a), int(b)] for a, b in MEDIAPIPE_BONES],
        "root": {"kind": "mid_hips", "indices": [LEFT_HIP_INDEX, RIGHT_HIP_INDEX]},
        "unit": "meter",
        "axis": "blender_xy_z",
    }


def _value(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _coords(landmarks) -> list[list[float]]:
    coords: list[list[float]] = []
    for landmark in landmarks:
        visibility = getattr(landmark, "visibility", None)
        visibility_value = float(visibility) if visibility is not None else 1.0
        coords.append(
            [
                _value(landmark.x),
                _value(landmark.y),
                _value(landmark.z),
                visibility_value,
            ]
        )
    return coords


def _timestamp_ms(frame_index: int, fps: float) -> int:
    if fps <= 0:
        return frame_index * 33
    return int(round(frame_index / fps * 1000.0))


def resolve_pose3d_model(
    model_path: Path | None,
    log_warn: Callable[[str], None] | None = None,
) -> Path:
    if model_path:
        candidate = Path(model_path).expanduser()
        if candidate.is_file():
            return candidate
        raise Pose3DError(f"Pose3D model not found: {candidate}")

    cache_dir = Path.home() / ".cache" / "od2blender"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Pose3DError("Failed to create cache directory for pose model") from exc

    target = cache_dir / POSE3D_MODEL_NAME
    if target.is_file():
        return target

    if log_warn:
        log_warn(f"Downloading pose model to {target}")

    tmp_path = target.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(POSE3D_MODEL_URL) as response:
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(target)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise Pose3DError("Failed to download pose model") from exc
    return target


def run_pose3d(
    video_path: Path,
    video_meta: VideoMeta,
    config: Pose3DConfig | None = None,
    log_warn: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict, dict]:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except Exception as exc:
        raise Pose3DError("mediapipe tasks API is required for pose3d") from exc

    config = config or Pose3DConfig()
    model_path = resolve_pose3d_model(config.model_path, log_warn=log_warn)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_segmentation_masks=False,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        landmarker.close()
        raise Pose3DError(f"Unable to open video: {video_path}")

    frames = build_empty_frames(video_meta.frame_count, video_meta.fps)

    try:
        for frame_index in range(video_meta.frame_count):
            ok, frame = capture.read()
            if not ok:
                if log_warn:
                    log_warn(f"Pose3D: video ended at frame {frame_index}")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = _timestamp_ms(frame_index, video_meta.fps)
            result = landmarker.detect_for_video(image, timestamp_ms)

            frame_entry = frames[frame_index]
            frame_entry["poses3d_raw"] = []

            if not result.pose_world_landmarks or not result.pose_landmarks:
                continue

            world = result.pose_world_landmarks[0]
            image_landmarks = result.pose_landmarks[0]

            frame_entry["poses3d_raw"].append(
                {
                    "id": 0,
                    "world": _coords(world),
                    "image": _coords(image_landmarks),
                }
            )
    finally:
        capture.release()
        landmarker.close()

    detector_meta = {
        "backend": "mediapipe",
        "task": "pose3d",
        "version": getattr(mp, "__version__", "unknown"),
        "model": model_path.name,
        "model_path": str(model_path),
        "min_detection_confidence": config.min_detection_confidence,
        "min_presence_confidence": config.min_presence_confidence,
        "min_tracking_confidence": config.min_tracking_confidence,
    }
    return frames, detector_meta, get_pose3d_skeleton()
