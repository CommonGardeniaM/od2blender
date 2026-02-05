"""Babylon-compatible quaternion math for MiKaPo retargeting."""

from __future__ import annotations

import numpy as np

EPS = 1e-9
BABYLON_EPS = 0.001


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    length = float(np.linalg.norm(quat))
    if length < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / length


def quat_mul(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    quat_a = np.asarray(quat_a, dtype=np.float64)
    quat_b = np.asarray(quat_b, dtype=np.float64)
    ax, ay, az, aw = quat_a
    bx, by, bz, bw = quat_b
    x = ax * bw + ay * bz - az * by + aw * bx
    y = -ax * bz + ay * bw + az * bx + aw * by
    z = ax * by - ay * bx + az * bw + aw * bz
    w = -ax * bx - ay * by - az * bz + aw * bw
    return np.array([x, y, z, w], dtype=np.float64)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quat_inv_unit(quat: np.ndarray) -> np.ndarray:
    return quat_conjugate(quat)


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat)
    vec = np.asarray(vec, dtype=np.float64)
    vec_quat = np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)
    return quat_mul(quat_mul(quat, vec_quat), quat_conjugate(quat))[:3]


def quat_from_unit_vectors(
    vec_from: np.ndarray, vec_to: np.ndarray, epsilon: float = BABYLON_EPS
) -> np.ndarray:
    vec_from = np.asarray(vec_from, dtype=np.float64)
    vec_to = np.asarray(vec_to, dtype=np.float64)
    from_norm = float(np.linalg.norm(vec_from))
    to_norm = float(np.linalg.norm(vec_to))
    if from_norm < EPS or to_norm < EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    v_from = vec_from / from_norm
    v_to = vec_to / to_norm
    r = float(np.dot(v_from, v_to) + 1.0)
    if r < epsilon:
        if abs(float(v_from[0])) > abs(float(v_from[2])):
            quat = np.array([-v_from[1], v_from[0], 0.0, 0.0], dtype=np.float64)
        else:
            quat = np.array([0.0, -v_from[2], v_from[1], 0.0], dtype=np.float64)
    else:
        cross = np.cross(v_from, v_to)
        quat = np.array([cross[0], cross[1], cross[2], r], dtype=np.float64)
    return quat_normalize(quat)


def quat_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3")

    m11 = float(matrix[0, 0])
    m12 = float(matrix[0, 1])
    m13 = float(matrix[0, 2])
    m21 = float(matrix[1, 0])
    m22 = float(matrix[1, 1])
    m23 = float(matrix[1, 2])
    m31 = float(matrix[2, 0])
    m32 = float(matrix[2, 1])
    m33 = float(matrix[2, 2])

    trace = m11 + m22 + m33
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m32 - m23) * s
        y = (m13 - m31) * s
        z = (m21 - m12) * s
    elif m11 > m22 and m11 > m33:
        s = 2.0 * np.sqrt(1.0 + m11 - m22 - m33)
        w = (m32 - m23) / s
        x = 0.25 * s
        y = (m12 + m21) / s
        z = (m13 + m31) / s
    elif m22 > m33:
        s = 2.0 * np.sqrt(1.0 + m22 - m11 - m33)
        w = (m13 - m31) / s
        x = (m12 + m21) / s
        y = 0.25 * s
        z = (m23 + m32) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m33 - m11 - m22)
        w = (m21 - m12) / s
        x = (m13 + m31) / s
        y = (m23 + m32) / s
        z = 0.25 * s

    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))

