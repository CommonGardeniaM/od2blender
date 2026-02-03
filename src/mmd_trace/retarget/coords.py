"""Coordinate transforms and centering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indices import POSE_IDX


@dataclass(frozen=True)
class AxisTransform:
    """Axis transform settings."""

    axis_x: float = 1.0
    axis_y: float = 1.0
    axis_z: float = -1.0
    axis_matrix: np.ndarray | None = None
    flip_y: bool = True

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if self.axis_matrix is not None:
            matrix = np.asarray(self.axis_matrix, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("axis_matrix must be 3x3")
        else:
            matrix = np.diag([self.axis_x, self.axis_y, self.axis_z]).astype(np.float64)

        mapped = points @ matrix.T
        if self.flip_y:
            mapped[:, 1] *= -1.0
        return mapped


def drop_z(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 2] = 0.0
    return points


def _pose_point(
    points: np.ndarray,
    vis: np.ndarray | None,
    name: str,
    vis_th: float,
) -> np.ndarray | None:
    idx = POSE_IDX[name]
    if vis is not None and float(vis[idx]) < vis_th:
        return None
    return points[idx]


def center_points(
    points: np.ndarray,
    vis: np.ndarray | None,
    vis_th: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    lhip = _pose_point(points, vis, "left_hip", vis_th)
    rhip = _pose_point(points, vis, "right_hip", vis_th)
    lsho = _pose_point(points, vis, "left_shoulder", vis_th)
    rsho = _pose_point(points, vis, "right_shoulder", vis_th)

    center: np.ndarray | None = None
    if lhip is not None and rhip is not None:
        center = 0.5 * (lhip + rhip)
    elif lsho is not None and rsho is not None:
        center = 0.5 * (lsho + rsho)
    else:
        if vis is not None:
            mask = np.asarray(vis, dtype=np.float64) >= vis_th
            if np.any(mask):
                center = points[mask].mean(axis=0)
        if center is None:
            center = points.mean(axis=0)

    return points - center, center
