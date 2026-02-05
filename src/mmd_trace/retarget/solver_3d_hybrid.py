"""Hybrid 3D solver using basis alignment."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmd_trace.io.pmx import PmxModel

from .bone_resolver import BoneResolver
from .indices import POSE_IDX
from .quat import IDENTITY_QUAT, apply, from_matrix, from_two_vectors, inv, mul

EPS = 1e-9
ROLL_BASE = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass
class Solver3DHybrid:
    """Solve bone rotations from 3D landmarks with twist-aware bases."""

    model: PmxModel
    vis_th: float = 0.2
    resolver: BoneResolver = field(init=False)
    bone_local: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    bone_world: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    centers: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    rest_local_cache: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    rest_local_quat_cache: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    hip_right: np.ndarray | None = field(init=False, default=None)
    shoulder_right: np.ndarray | None = field(init=False, default=None)
    torso_forward: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.resolver = BoneResolver(self.model)

    def solve(
        self, points: np.ndarray, vis: np.ndarray | None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
        self.bone_local = {}
        self.bone_world = {}
        self.centers = {}
        self.hip_right = None
        self.shoulder_right = None
        self.torso_forward = None
        self._compute_centers(points, vis)
        self._solve_torso(points, vis)
        self._solve_neck_head(points, vis)
        self._solve_arm(points, vis, "L")
        self._solve_arm(points, vis, "R")
        self._solve_leg(points, vis, "L")
        self._solve_leg(points, vis, "R")
        return self.bone_local, self.bone_world, self.centers, dict(self.resolver.bones)

    def _pose_pt(
        self, points: np.ndarray, vis: np.ndarray | None, name: str
    ) -> np.ndarray | None:
        idx = POSE_IDX[name]
        if vis is not None and float(vis[idx]) < self.vis_th:
            return None
        return points[idx]

    def _dir(self, start: np.ndarray | None, end: np.ndarray | None) -> np.ndarray | None:
        if start is None or end is None:
            return None
        delta = end - start
        if np.linalg.norm(delta) < EPS:
            return None
        return delta

    def _safe_norm(self, vec: np.ndarray | None) -> np.ndarray | None:
        if vec is None:
            return None
        length = float(np.linalg.norm(vec))
        if length < EPS:
            return None
        return vec / length

    def _safe_cross(
        self, vec_a: np.ndarray | None, vec_b: np.ndarray | None
    ) -> np.ndarray | None:
        if vec_a is None or vec_b is None:
            return None
        cross = np.cross(vec_a, vec_b)
        if np.linalg.norm(cross) < EPS:
            return None
        return cross

    def _first_vec(self, *vecs: np.ndarray | None) -> np.ndarray | None:
        for vec in vecs:
            if vec is not None:
                return vec
        return None

    def _get_parent_world(self, bone_name: str) -> np.ndarray:
        parent = self.resolver.parent_name(bone_name)
        if parent is None:
            return IDENTITY_QUAT.copy()
        parent_world = self.bone_world.get(parent)
        if parent_world is not None:
            return parent_world
        parent_bone = self.resolver.get_bone(parent)
        if parent_bone is None:
            return IDENTITY_QUAT.copy()
        return from_matrix(self.model.get_bone_rest_matrix(parent_bone.index))

    def _rest_local_matrix(self, bone_name: str) -> np.ndarray:
        cached = self.rest_local_cache.get(bone_name)
        if cached is not None:
            return cached
        bone = self.resolver.get_bone(bone_name)
        if bone is None:
            matrix = np.eye(3, dtype=np.float64)
        else:
            matrix = self.model.compute_bone_local_basis(bone.index)
        self.rest_local_cache[bone_name] = matrix
        return matrix

    def _rest_local_quat(self, bone_name: str) -> np.ndarray:
        cached = self.rest_local_quat_cache.get(bone_name)
        if cached is not None:
            return cached
        matrix = self._rest_local_matrix(bone_name)
        quat = from_matrix(matrix)
        self.rest_local_quat_cache[bone_name] = quat
        return quat

    def _set_local(self, bone_name: str, quat_local: np.ndarray) -> None:
        parent_world = self._get_parent_world(bone_name)
        rest_local = self._rest_local_quat(bone_name)
        quat_world = mul(parent_world, mul(rest_local, quat_local))
        self.bone_local[bone_name] = quat_local
        self.bone_world[bone_name] = quat_world

    def _build_basis(
        self,
        target_y: np.ndarray | None,
        target_x_hint: np.ndarray | None,
        rest_local: np.ndarray,
    ) -> np.ndarray | None:
        y_axis = self._safe_norm(target_y)
        if y_axis is None:
            return None

        rest_x = rest_local[:, 0]
        x_axis = None
        if target_x_hint is not None:
            x_proj = target_x_hint - np.dot(target_x_hint, y_axis) * y_axis
            x_axis = self._safe_norm(x_proj)
        if x_axis is None:
            x_proj = rest_x - np.dot(rest_x, y_axis) * y_axis
            x_axis = self._safe_norm(x_proj)
        if x_axis is None:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(axis, y_axis))) > 0.9:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            x_proj = axis - np.dot(axis, y_axis) * y_axis
            x_axis = self._safe_norm(x_proj)
        if x_axis is None:
            return None

        z_axis = self._safe_norm(np.cross(x_axis, y_axis))
        if z_axis is None:
            return None
        x_axis = self._safe_norm(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None

        if float(np.dot(x_axis, rest_x)) < 0.0:
            x_axis = -x_axis
            z_axis = -z_axis

        return np.column_stack([x_axis, y_axis, z_axis])

    def _align_dir(self, bone_key: str, target_world: np.ndarray | None) -> bool:
        bone_name = self.resolver.bone(bone_key)
        if bone_name is None or target_world is None:
            return False
        parent_world = self._get_parent_world(bone_name)
        target_local = apply(inv(parent_world), target_world)
        rest_local = self._rest_local_quat(bone_name)
        target_bone = apply(inv(rest_local), target_local)
        quat_local = from_two_vectors(ROLL_BASE, target_bone)
        self._set_local(bone_name, quat_local)
        return True

    def _align_basis(
        self,
        bone_key: str,
        target_y_world: np.ndarray | None,
        target_x_hint_world: np.ndarray | None,
    ) -> bool:
        bone_name = self.resolver.bone(bone_key)
        if bone_name is None or target_y_world is None:
            return False

        parent_world = self._get_parent_world(bone_name)
        target_y_local = apply(inv(parent_world), target_y_world)
        target_x_hint_local = None
        if target_x_hint_world is not None:
            target_x_hint_local = apply(inv(parent_world), target_x_hint_world)

        rest_local = self._rest_local_matrix(bone_name)
        target_local = self._build_basis(target_y_local, target_x_hint_local, rest_local)
        if target_local is None:
            return False

        delta_matrix = rest_local.T @ target_local
        quat_local = from_matrix(delta_matrix)
        self._set_local(bone_name, quat_local)
        return True

    def _compute_centers(self, points: np.ndarray, vis: np.ndarray | None) -> None:
        lhip = self._pose_pt(points, vis, "left_hip")
        rhip = self._pose_pt(points, vis, "right_hip")
        lsho = self._pose_pt(points, vis, "left_shoulder")
        rsho = self._pose_pt(points, vis, "right_shoulder")
        lear = self._pose_pt(points, vis, "left_ear")
        rear = self._pose_pt(points, vis, "right_ear")
        leye = self._pose_pt(points, vis, "left_eye")
        reye = self._pose_pt(points, vis, "right_eye")
        nose = self._pose_pt(points, vis, "nose")

        if lhip is not None and rhip is not None:
            self.centers["hip_center"] = 0.5 * (lhip + rhip)
        if lsho is not None and rsho is not None:
            self.centers["shoulder_center"] = 0.5 * (lsho + rsho)
        if lear is not None and rear is not None:
            self.centers["ear_center"] = 0.5 * (lear + rear)
        if leye is not None and reye is not None:
            self.centers["eye_center"] = 0.5 * (leye + reye)
        if nose is not None:
            self.centers["nose"] = nose

    def _solve_torso(self, points: np.ndarray, vis: np.ndarray | None) -> None:
        lhip = self._pose_pt(points, vis, "left_hip")
        rhip = self._pose_pt(points, vis, "right_hip")
        lsho = self._pose_pt(points, vis, "left_shoulder")
        rsho = self._pose_pt(points, vis, "right_shoulder")

        hip_center = self.centers.get("hip_center")
        shoulder_center = self.centers.get("shoulder_center")
        ear_center = self.centers.get("ear_center")

        self.hip_right = self._dir(rhip, lhip)
        self.shoulder_right = self._dir(rsho, lsho)

        spine_dir = self._dir(hip_center, shoulder_center)
        if self.shoulder_right is not None and spine_dir is not None:
            self.torso_forward = self._safe_cross(self.shoulder_right, spine_dir)
        elif self.hip_right is not None and spine_dir is not None:
            self.torso_forward = self._safe_cross(self.hip_right, spine_dir)

        lower_body_dir = spine_dir if spine_dir is not None else self.hip_right
        if not self._align_basis("lower_body", lower_body_dir, self.hip_right):
            self._align_dir("lower_body", lower_body_dir)

        if not self._align_basis("upper_body", spine_dir, self.shoulder_right):
            self._align_dir("upper_body", spine_dir)

        upper2_dir = self._dir(shoulder_center, ear_center)
        if upper2_dir is None:
            upper2_dir = spine_dir
        if not self._align_basis("upper_body2", upper2_dir, self.shoulder_right):
            self._align_dir("upper_body2", upper2_dir)

    def _solve_neck_head(self, points: np.ndarray, vis: np.ndarray | None) -> None:
        ear_center = self.centers.get("ear_center")
        eye_center = self.centers.get("eye_center")
        shoulder_center = self.centers.get("shoulder_center")
        nose = self.centers.get("nose")

        leye = self._pose_pt(points, vis, "left_eye")
        reye = self._pose_pt(points, vis, "right_eye")
        lear = self._pose_pt(points, vis, "left_ear")
        rear = self._pose_pt(points, vis, "right_ear")

        neck_dir = self._dir(shoulder_center, ear_center)
        if neck_dir is None and nose is not None and shoulder_center is not None:
            neck_dir = self._dir(shoulder_center, nose)

        neck_x = self._dir(rear, lear)
        if neck_x is None:
            neck_x = self.shoulder_right

        if not self._align_basis("neck", neck_dir, neck_x):
            self._align_dir("neck", neck_dir)

        head_dir = self._dir(ear_center, eye_center)
        if head_dir is None and nose is not None and ear_center is not None:
            head_dir = self._dir(ear_center, nose)

        head_x = self._dir(reye, leye)
        if head_x is None:
            head_x = self._dir(rear, lear)
        if head_x is None:
            head_x = self.shoulder_right

        if not self._align_basis("head", head_dir, head_x):
            self._align_dir("head", head_dir)

    def _solve_arm(self, points: np.ndarray, vis: np.ndarray | None, side: str) -> None:
        shoulder_point = self._pose_pt(points, vis, "left_shoulder" if side == "L" else "right_shoulder")
        elbow_point = self._pose_pt(points, vis, "left_elbow" if side == "L" else "right_elbow")
        wrist_point = self._pose_pt(points, vis, "left_wrist" if side == "L" else "right_wrist")
        index_point = self._pose_pt(points, vis, "left_index" if side == "L" else "right_index")
        pinky_point = self._pose_pt(points, vis, "left_pinky" if side == "L" else "right_pinky")

        shoulder_center = self.centers.get("shoulder_center")

        shoulder_dir = self._dir(shoulder_center, shoulder_point)
        shoulder_x = self._first_vec(self.torso_forward, self.shoulder_right, self.hip_right)
        if not self._align_basis(f"shoulder_{side}", shoulder_dir, shoulder_x):
            self._align_dir(f"shoulder_{side}", shoulder_dir)

        upper_arm_dir = self._dir(shoulder_point, elbow_point)
        arm_x = self._first_vec(self.torso_forward, self.shoulder_right, self.hip_right)
        if not self._align_basis(f"arm_{side}", upper_arm_dir, arm_x):
            self._align_dir(f"arm_{side}", upper_arm_dir)

        forearm_dir = self._dir(elbow_point, wrist_point)
        plane_n = self._safe_cross(upper_arm_dir, forearm_dir)
        elbow_x = self._first_vec(plane_n, self.torso_forward, self.shoulder_right, self.hip_right)
        if not self._align_basis(f"elbow_{side}", forearm_dir, elbow_x):
            self._align_dir(f"elbow_{side}", forearm_dir)

        hand_dir = None
        across = None
        if wrist_point is not None and index_point is not None and pinky_point is not None:
            hand_center = 0.5 * (index_point + pinky_point)
            hand_dir = self._dir(wrist_point, hand_center)
            across = self._dir(pinky_point, index_point)
        elif wrist_point is not None and index_point is not None:
            hand_dir = self._dir(wrist_point, index_point)
        elif wrist_point is not None and pinky_point is not None:
            hand_dir = self._dir(wrist_point, pinky_point)

        wrist_dir = hand_dir if hand_dir is not None else forearm_dir
        wrist_x = self._first_vec(across, plane_n, self.torso_forward, self.shoulder_right, self.hip_right)
        if not self._align_basis(f"wrist_{side}", wrist_dir, wrist_x):
            self._align_dir(f"wrist_{side}", wrist_dir)

    def _solve_leg(self, points: np.ndarray, vis: np.ndarray | None, side: str) -> None:
        hip = self._pose_pt(points, vis, "left_hip" if side == "L" else "right_hip")
        knee = self._pose_pt(points, vis, "left_knee" if side == "L" else "right_knee")
        ankle = self._pose_pt(points, vis, "left_ankle" if side == "L" else "right_ankle")
        heel = self._pose_pt(points, vis, "left_heel" if side == "L" else "right_heel")
        foot = self._pose_pt(points, vis, "left_foot_index" if side == "L" else "right_foot_index")

        thigh_dir = self._dir(hip, knee)
        shin_dir = self._dir(knee, ankle)
        plane_n = self._safe_cross(thigh_dir, shin_dir)

        leg_x = self._first_vec(self.hip_right, self.torso_forward, self.shoulder_right)
        if not self._align_basis(f"leg_{side}", thigh_dir, leg_x):
            self._align_dir(f"leg_{side}", thigh_dir)

        knee_x = self._first_vec(plane_n, self.hip_right, self.torso_forward, self.shoulder_right)
        if not self._align_basis(f"knee_{side}", shin_dir, knee_x):
            self._align_dir(f"knee_{side}", shin_dir)

        toe_dir = self._dir(ankle, foot)
        foot_n = None
        if ankle is not None and heel is not None and foot is not None:
            heel_vec = heel - ankle
            toe_vec = foot - ankle
            foot_n = self._safe_cross(heel_vec, toe_vec)
        ankle_x = None
        if toe_dir is not None and foot_n is not None:
            ankle_x = self._safe_cross(toe_dir, foot_n)
        ankle_dir = toe_dir if toe_dir is not None else shin_dir
        ankle_x = self._first_vec(ankle_x, plane_n, self.hip_right, self.torso_forward, self.shoulder_right)
        if not self._align_basis(f"ankle_{side}", ankle_dir, ankle_x):
            self._align_dir(f"ankle_{side}", ankle_dir)


def solve(
    model: PmxModel, points: np.ndarray, vis: np.ndarray | None, vis_th: float = 0.2
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
    solver = Solver3DHybrid(model, vis_th=vis_th)
    return solver.solve(points, vis)
