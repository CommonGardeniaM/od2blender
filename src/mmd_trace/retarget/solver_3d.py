"""3D solver using rest-direction alignment."""
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
class Solver3D:
    """Solve bone rotations from 3D landmarks."""

    model: PmxModel
    vis_th: float = 0.2
    resolver: BoneResolver = field(init=False)
    bone_local: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    bone_world: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    centers: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    rest_local_cache: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.resolver = BoneResolver(self.model)

    def solve(
        self, points: np.ndarray, vis: np.ndarray | None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
        self.bone_local = {}
        self.bone_world = {}
        self.centers = {}
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

    def _rest_local_quat(self, bone_name: str) -> np.ndarray:
        cached = self.rest_local_cache.get(bone_name)
        if cached is not None:
            return cached
        bone = self.resolver.get_bone(bone_name)
        if bone is None:
            quat = IDENTITY_QUAT.copy()
        else:
            quat = from_matrix(self.model.compute_bone_local_basis(bone.index))
        self.rest_local_cache[bone_name] = quat
        return quat

    def _set_local(self, bone_name: str, quat_local: np.ndarray) -> None:
        parent_world = self._get_parent_world(bone_name)
        rest_local = self._rest_local_quat(bone_name)
        quat_world = mul(parent_world, mul(rest_local, quat_local))
        self.bone_local[bone_name] = quat_local
        self.bone_world[bone_name] = quat_world

    def _align(self, bone_key: str, target_world: np.ndarray | None) -> None:
        bone_name = self.resolver.bone(bone_key)
        if bone_name is None or target_world is None:
            return
        parent_world = self._get_parent_world(bone_name)
        target_local = apply(inv(parent_world), target_world)
        rest_local = self._rest_local_quat(bone_name)
        target_bone = apply(inv(rest_local), target_local)
        quat_local = from_two_vectors(ROLL_BASE, target_bone)
        self._set_local(bone_name, quat_local)

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
        hip_dir = self._dir(rhip, lhip)
        hip_center = self.centers.get("hip_center")
        shoulder_center = self.centers.get("shoulder_center")
        spine_dir = self._dir(hip_center, shoulder_center)
        lower_body_dir = spine_dir if spine_dir is not None else hip_dir
        self._align("lower_body", lower_body_dir)
        self._align("upper_body", spine_dir)

        ear_center = self.centers.get("ear_center")
        upper2_dir = self._dir(shoulder_center, ear_center)
        self._align("upper_body2", upper2_dir)

    def _solve_neck_head(self, points: np.ndarray, vis: np.ndarray | None) -> None:
        shoulder_center = self.centers.get("shoulder_center")
        ear_center = self.centers.get("ear_center")
        eye_center = self.centers.get("eye_center")
        nose = self.centers.get("nose")

        neck_dir = self._dir(shoulder_center, ear_center)
        self._align("neck", neck_dir)

        head_dir = self._dir(ear_center, eye_center)
        if head_dir is None and nose is not None and ear_center is not None:
            head_dir = self._dir(ear_center, nose)
        self._align("head", head_dir)

    def _solve_arm(self, points: np.ndarray, vis: np.ndarray | None, side: str) -> None:
        shoulder_point = self._pose_pt(points, vis, "left_shoulder" if side == "L" else "right_shoulder")
        elbow_point = self._pose_pt(points, vis, "left_elbow" if side == "L" else "right_elbow")
        wrist_point = self._pose_pt(points, vis, "left_wrist" if side == "L" else "right_wrist")

        shoulder_center = self.centers.get("shoulder_center")
        self._align(f"shoulder_{side}", self._dir(shoulder_center, shoulder_point))
        self._align(f"arm_{side}", self._dir(shoulder_point, elbow_point))
        self._align(f"elbow_{side}", self._dir(elbow_point, wrist_point))
        self._align(f"wrist_{side}", self._dir(elbow_point, wrist_point))

    def _solve_leg(self, points: np.ndarray, vis: np.ndarray | None, side: str) -> None:
        hip = self._pose_pt(points, vis, "left_hip" if side == "L" else "right_hip")
        knee = self._pose_pt(points, vis, "left_knee" if side == "L" else "right_knee")
        ankle = self._pose_pt(points, vis, "left_ankle" if side == "L" else "right_ankle")
        foot = self._pose_pt(points, vis, "left_foot_index" if side == "L" else "right_foot_index")

        self._align(f"leg_{side}", self._dir(hip, knee))
        self._align(f"knee_{side}", self._dir(knee, ankle))
        self._align(f"ankle_{side}", self._dir(ankle, foot))


def solve(
    model: PmxModel, points: np.ndarray, vis: np.ndarray | None, vis_th: float = 0.2
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
    solver = Solver3D(model, vis_th=vis_th)
    return solver.solve(points, vis)
