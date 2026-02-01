"""Quaternion utilities for retargeting."""

from __future__ import annotations

from typing import Tuple
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion from v0 to v1."""

    v0_norm = _safe_normalize(v0)
    v1_norm = _safe_normalize(v1)
    if v0_norm is None or v1_norm is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -0.999999:
        axis = _orthogonal(v0_norm)
        return quat_normalize(np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64))

    axis = np.cross(v0_norm, v1_norm)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float64)
    return quat_normalize(q)


def _safe_normalize(v: np.ndarray) -> Tuple[float, float, float] | None:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return v / norm


def _orthogonal(v: np.ndarray) -> np.ndarray:
    if abs(v[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(v, other)
    return axis / (np.linalg.norm(axis) + 1e-8)
