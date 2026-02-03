"""Pose provider base types and factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PoseBundleNp:
    """Pose landmarks bundle in NumPy arrays."""

    world_points: np.ndarray
    world_vis: np.ndarray
    image_points: np.ndarray
    image_vis: np.ndarray


class PoseProvider(Protocol):
    """Pose provider interface."""

    def detect(self, image: np.ndarray) -> PoseBundleNp:  # pragma: no cover - protocol
        ...


def create_pose_provider(
    name: str = "mediapipe",
    det_conf: float = 0.5,
) -> PoseProvider:
    if name != "mediapipe":
        raise ValueError(f"Unsupported pose provider: {name}")
    from .mediapipe_provider import MediaPipePoseProvider

    return MediaPipePoseProvider(det_conf=det_conf)
