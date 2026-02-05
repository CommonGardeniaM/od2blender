"""MiKaPo-style 3D skeleton image renderer."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from mmd_trace.retarget.indices import POSE_CONNECTIONS

LOG = logging.getLogger(__name__)

EPS = 1e-6


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    cos_pitch = float(np.cos(pitch))
    sin_pitch = float(np.sin(pitch))
    cos_roll = float(np.cos(roll))
    sin_roll = float(np.sin(roll))

    rot_yaw = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    rot_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float64,
    )
    rot_roll = np.array(
        [
            [cos_roll, -sin_roll, 0.0],
            [sin_roll, cos_roll, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rot_roll @ rot_pitch @ rot_yaw


def _project_points(
    points: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    matrix = _rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    rotated = points @ matrix.T
    return rotated[:, :2]


def render_pose_skeleton_image(
    points: np.ndarray,
    vis: np.ndarray | None,
    width: int = 768,
    height: int = 768,
    margin: float = 0.1,
    thickness: int = 2,
    yaw_deg: float = 45.0,
    pitch_deg: float = 20.0,
    roll_deg: float = 0.0,
    vis_th: float = 0.2,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if thickness <= 0:
        raise ValueError("thickness must be positive")

    margin = float(np.clip(margin, 0.0, 0.45))

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a Nx3 array")

    if vis is None:
        mask = np.ones(points.shape[0], dtype=bool)
    else:
        vis = np.asarray(vis, dtype=np.float64)
        if vis.shape[0] != points.shape[0]:
            raise ValueError("vis length must match points length")
        mask = vis >= vis_th

    if not np.any(mask):
        return np.zeros((height, width, 3), dtype=np.uint8)

    points_2d = _project_points(points, yaw_deg, pitch_deg, roll_deg)
    points_use = points_2d[mask]

    min_xy = points_use.min(axis=0)
    max_xy = points_use.max(axis=0)
    span = np.maximum(max_xy - min_xy, EPS)

    avail_w = float(width) * (1.0 - 2.0 * margin)
    avail_h = float(height) * (1.0 - 2.0 * margin)
    scale = min(avail_w / span[0], avail_h / span[1])

    center = (min_xy + max_xy) * 0.5
    points_scaled = (points_2d - center) * scale
    pixels_x = points_scaled[:, 0] + float(width) * 0.5
    pixels_y = float(height) * 0.5 - points_scaled[:, 1]
    pixels = np.stack([pixels_x, pixels_y], axis=1).round().astype(int)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for start, end in POSE_CONNECTIONS:
        if start >= len(pixels) or end >= len(pixels):
            continue
        if not mask[start] or not mask[end]:
            continue
        pt_start = tuple(pixels[start])
        pt_end = tuple(pixels[end])
        cv2.line(
            image,
            pt_start,
            pt_end,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def write_pose_skeleton_png(out_path: str, image: np.ndarray) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")
    LOG.info("Saved skeleton image: %s", path)
