from __future__ import annotations

import numpy as np

from mmd_trace.retarget.mikapo_math import (
    quat_from_rotation_matrix,
    quat_from_unit_vectors,
)


def _quat_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return np.allclose(a, b, atol=tol) or np.allclose(a, -b, atol=tol)


def test_quat_from_unit_vectors_identity() -> None:
    quat = quat_from_unit_vectors(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert _quat_equal(quat, np.array([0.0, 0.0, 0.0, 1.0]))


def test_quat_from_unit_vectors_opposite() -> None:
    quat = quat_from_unit_vectors(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert _quat_equal(quat, np.array([0.0, 1.0, 0.0, 0.0]))


def test_quat_from_rotation_matrix_z_90() -> None:
    angle = np.deg2rad(90.0)
    cos = float(np.cos(angle))
    sin = float(np.sin(angle))
    matrix = np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    quat = quat_from_rotation_matrix(matrix)
    expected = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float64)
    assert _quat_equal(quat, expected)
