"""Quaternion utilities for retargeting."""

from __future__ import annotations

import math
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


def quat_from_aim_up(
    aim0: np.ndarray,
    up0: np.ndarray,
    aim1: np.ndarray,
    up1: np.ndarray,
) -> np.ndarray:
    basis0 = _basis_from_aim_up(aim0, up0)
    basis1 = _basis_from_aim_up(aim1, up1)
    if basis0 is None or basis1 is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot = basis1 @ basis0.T
    return quat_normalize(quat_from_matrix(rot))


def rotate_vector_around_axis(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = _safe_normalize(axis)
    if axis_n is None:
        return vec
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        vec * cos_a
        + np.cross(axis_n, vec) * sin_a
        + axis_n * np.dot(axis_n, vec) * (1.0 - cos_a)
    )


def _safe_normalize(v: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return v / norm


def _basis_from_aim_up(aim: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    aim_n = _safe_normalize(aim)
    up_n = _safe_normalize(up)
    if aim_n is None or up_n is None:
        return None
    right = np.cross(up_n, aim_n)
    right_n = _safe_normalize(right)
    if right_n is None:
        return None
    up2 = np.cross(aim_n, right_n)
    return np.stack([right_n, aim_n, up2], axis=1)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _orthogonal(v: np.ndarray) -> np.ndarray:
    if abs(v[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(v, other)
    return axis / (np.linalg.norm(axis) + 1e-8)
