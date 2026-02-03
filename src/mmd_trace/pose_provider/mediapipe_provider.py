"""MediaPipe pose provider."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerResult,
)

from .base import PoseBundleNp

LOG = logging.getLogger(__name__)


def _resolve_model_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    model_path = repo_root / "models" / "pose_landmarker_heavy.task"
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
    return model_path


class MediaPipePoseProvider:
    """MediaPipe Tasks pose provider."""

    def __init__(self, det_conf: float = 0.5) -> None:
        model_path = _resolve_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=det_conf,
            output_segmentation_masks=False,
        )
        self.detector = PoseLandmarker.create_from_options(options)
        LOG.info("MediaPipe PoseLandmarker initialized")

    def _detect(self, image: np.ndarray) -> PoseLandmarkerResult:
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect(mp_image)

    def detect(self, image: np.ndarray) -> PoseBundleNp:
        result = self._detect(image)

        if not result.pose_world_landmarks or not result.pose_landmarks:
            raise RuntimeError("PoseLandmarker detected no poses.")

        world_landmarks = result.pose_world_landmarks[0]
        image_landmarks = result.pose_landmarks[0]

        world_points = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float64
        )
        world_vis = np.array(
            [
                lm.visibility if lm.visibility is not None else 1.0
                for lm in world_landmarks
            ],
            dtype=np.float64,
        )

        image_points = np.array(
            [[lm.x, lm.y, lm.z] for lm in image_landmarks], dtype=np.float64
        )
        image_vis = np.array(
            [
                lm.visibility if lm.visibility is not None else 1.0
                for lm in image_landmarks
            ],
            dtype=np.float64,
        )

        return PoseBundleNp(
            world_points=world_points,
            world_vis=world_vis,
            image_points=image_points,
            image_vis=image_vis,
        )
