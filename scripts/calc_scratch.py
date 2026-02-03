from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

left_shoulder = np.array([0.59662926197052, 0.2969486117362976, -0.15011295676231384])
right_shoulder = np.array([0.4050024449825287, 0.29932692646980286, -0.10676739364862442])
left_hip = np.array([0.5633628368377686, 0.5334545373916626, -0.016372714191675186])
right_hip = np.array([0.43847575783729553, 0.535342812538147, 0.01641947217285633])


def _norm(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length < eps:
        return None
    return vector / length


def solve_upper_body_np_scipy(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_hip: np.ndarray | None = None,
    right_hip: np.ndarray | None = None,
    use_hip_center: bool = True,
    use_rows_like_babylon: bool = True,
    flip_z: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    left_shoulder_vec = np.asarray(left_shoulder, dtype=float)
    right_shoulder_vec = np.asarray(right_shoulder, dtype=float)

    shoulder_center = 0.5 * (left_shoulder_vec + right_shoulder_vec)

    axis_x = _norm(left_shoulder_vec - right_shoulder_vec)
    if axis_x is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    if use_hip_center:
        if left_hip is None or right_hip is None:
            raise ValueError("use_hip_center=True requires left_hip/right_hip")
        left_hip_vec = np.asarray(left_hip, dtype=float)
        right_hip_vec = np.asarray(right_hip, dtype=float)
        hip_center = 0.5 * (left_hip_vec + right_hip_vec)
        axis_y = _norm(shoulder_center - hip_center)
    else:
        axis_y = _norm(shoulder_center)

    if axis_y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    axis_y = axis_y - axis_x * np.dot(axis_y, axis_x)
    axis_y = _norm(axis_y)
    if axis_y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    axis_z = _norm(np.cross(axis_x, axis_y))
    if axis_z is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    if flip_z:
        axis_z = -axis_z

    axis_y = _norm(np.cross(axis_z, axis_x))
    if axis_y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    if use_rows_like_babylon:
        matrix = np.vstack([axis_x, axis_y, axis_z])
    else:
        matrix = np.column_stack([axis_x, axis_y, axis_z])

    quat = Rotation.from_matrix(matrix).as_quat()
    return quat, matrix


if __name__ == "__main__":
    quat, matrix = solve_upper_body_np_scipy(left_shoulder, right_shoulder, left_hip, right_hip, use_hip_center=True)
    print("quat (x,y,z,w):", quat)
    print("M:\n", matrix)
    print(
        "dot(x,y), dot(y,z), dot(z,x):",
        np.dot(matrix[0], matrix[1]),
        np.dot(matrix[1], matrix[2]),
        np.dot(matrix[2], matrix[0]),
    )
