"""Quaternion utilities based on SciPy Rotation."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

EPS = 1e-9
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _norm(vec: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vec))
    if length < EPS:
        return None
    return vec / length


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    length = float(np.linalg.norm(quat))
    if length < EPS:
        return IDENTITY_QUAT.copy()
    return quat / length


def mul(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    return (Rotation.from_quat(quat_a) * Rotation.from_quat(quat_b)).as_quat()


def inv(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).inv().as_quat()


def apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).apply(vector)


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_n = _norm(np.asarray(axis, dtype=np.float64))
    if axis_n is None:
        return IDENTITY_QUAT.copy()
    return Rotation.from_rotvec(axis_n * float(angle)).as_quat()


def from_matrix(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix).as_quat()


def from_two_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    vec_a_norm = _norm(np.asarray(vec_a, dtype=np.float64))
    vec_b_norm = _norm(np.asarray(vec_b, dtype=np.float64))
    if vec_a_norm is None or vec_b_norm is None:
        return IDENTITY_QUAT.copy()

    dot = float(np.clip(np.dot(vec_a_norm, vec_b_norm), -1.0, 1.0))
    if dot > 0.999999:
        return IDENTITY_QUAT.copy()
    if dot < -0.999999:
        axis = np.cross(vec_a_norm, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(vec_a_norm, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        return from_axis_angle(axis, np.pi)

    axis = np.cross(vec_a_norm, vec_b_norm)
    angle = float(np.arctan2(np.linalg.norm(axis), dot))
    return from_axis_angle(axis, angle)


def make_basis(
    right: np.ndarray,
    up: np.ndarray,
    forward_hint: np.ndarray | None = None,
) -> np.ndarray:
    right_axis = _norm(np.asarray(right, dtype=np.float64))
    if right_axis is None:
        return np.eye(3, dtype=np.float64)

    up_axis = up - np.dot(up, right_axis) * right_axis
    if np.linalg.norm(up_axis) < 1e-6:
        temp_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(temp_axis, right_axis)) > 0.9:
            temp_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up_axis = temp_axis - np.dot(temp_axis, right_axis) * right_axis
    up_axis = _norm(up_axis)
    if up_axis is None:
        return np.eye(3, dtype=np.float64)

    forward_axis = np.cross(right_axis, up_axis)
    if np.linalg.norm(forward_axis) < 1e-6:
        forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_axis = _norm(forward_axis)
    if forward_axis is None:
        return np.eye(3, dtype=np.float64)

    if forward_hint is not None and np.dot(forward_axis, forward_hint) < 0:
        up_axis = -up_axis
        forward_axis = -forward_axis

    return np.stack([right_axis, up_axis, forward_axis], axis=1)
