"""MiKaPo pose-only solver port."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmd_trace.io.pmx import PmxModel

from .bone_resolver import BoneResolver
from .indices import POSE_IDX
from .mikapo_math import (
    quat_apply,
    quat_from_rotation_matrix,
    quat_from_unit_vectors,
    quat_inv_unit,
    quat_mul,
    quat_normalize,
)

EPS = 1e-9
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

_REF_LOWER_BODY = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_REF_NECK = np.array([0.0, 0.9758578206707508, -0.21840676233975218], dtype=np.float64)
_REF_LEG = np.array(
    [-0.009540689177369048, -0.998440855265296, 0.05499848895310636],
    dtype=np.float64,
)
_REF_KNEE_L = np.array(
    [-0.0007085292291306043, -0.9908517790187175, 0.1349527695224302],
    dtype=np.float64,
)
_REF_KNEE_R = np.array(
    [0.0007079817891811808, -0.9908517794028981, 0.13495276957475513],
    dtype=np.float64,
)
_REF_ANKLE = np.array([0.0, -0.65728916525082, -0.7536384764884819], dtype=np.float64)
_REF_ARM_L = np.array(
    [0.8012514930735141, -0.5966378711527615, -0.04493657256361681],
    dtype=np.float64,
)
_REF_ARM_R = np.array(
    [-0.8020376176381924, -0.5972232450219962, -0.007749548286792409],
    dtype=np.float64,
)
_REF_ELBOW_L = np.array(
    [0.7991214493734219, -0.600241324846603, -0.03339552511514752],
    dtype=np.float64,
)
_REF_ELBOW_R = np.array(
    [-0.7991213083626819, -0.6002415122251716, -0.03339553147285845],
    dtype=np.float64,
)
_REF_HEAD_H = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_REF_HEAD_V = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _safe_norm(vec: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vec))
    if length < EPS:
        return None
    return vec / length


def _unit_ref(vec: np.ndarray) -> np.ndarray:
    vec_n = _safe_norm(np.asarray(vec, dtype=np.float64))
    if vec_n is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return vec_n


_REF_NECK = _unit_ref(_REF_NECK)
_REF_LEG = _unit_ref(_REF_LEG)
_REF_KNEE_L = _unit_ref(_REF_KNEE_L)
_REF_KNEE_R = _unit_ref(_REF_KNEE_R)
_REF_ANKLE = _unit_ref(_REF_ANKLE)
_REF_ARM_L = _unit_ref(_REF_ARM_L)
_REF_ARM_R = _unit_ref(_REF_ARM_R)
_REF_ELBOW_L = _unit_ref(_REF_ELBOW_L)
_REF_ELBOW_R = _unit_ref(_REF_ELBOW_R)

_SOLVE_ORDER = (
    "upper_body",
    "neck",
    "head",
    "lower_body",
    "leg_L",
    "leg_R",
    "knee_L",
    "knee_R",
    "ankle_L",
    "ankle_R",
    "arm_L",
    "arm_R",
    "elbow_L",
    "elbow_R",
)


@dataclass
class SolverMiKaPoPose:
    """Solve bone rotations from pose world landmarks using MiKaPo logic (pose-only)."""

    model: PmxModel
    vis_th: float = 0.2
    resolver: BoneResolver = field(init=False)
    bone_local: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    bone_world: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    centers: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.resolver = BoneResolver(self.model)

    def solve(
        self, points: np.ndarray, vis: np.ndarray | None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
        self.bone_local = {}
        self.bone_world = {}
        self.centers = {}
        self._init_identity_bones()
        self._compute_centers(points)

        self._solve_upper_body(points)
        self._solve_neck(points)
        self._solve_head(points)
        self._solve_lower_body(points)
        self._solve_leg(points, "L")
        self._solve_leg(points, "R")
        self._solve_knee(points, "L")
        self._solve_knee(points, "R")
        self._solve_ankle(points, "L")
        self._solve_ankle(points, "R")
        self._solve_arm(points, "L")
        self._solve_arm(points, "R")
        self._solve_elbow(points, "L")
        self._solve_elbow(points, "R")

        return self.bone_local, self.bone_world, self.centers, dict(self.resolver.bones)

    def _init_identity_bones(self) -> None:
        for key in _SOLVE_ORDER:
            bone_name = self.resolver.bone(key)
            if bone_name is None:
                continue
            self.bone_local[bone_name] = IDENTITY_QUAT.copy()
            self.bone_world[bone_name] = IDENTITY_QUAT.copy()

    def _pose_pt(self, points: np.ndarray, name: str) -> np.ndarray:
        return points[POSE_IDX[name]]

    def _world_quat_key(self, bone_key: str) -> np.ndarray:
        bone_name = self.resolver.bone(bone_key)
        if bone_name is None:
            return IDENTITY_QUAT.copy()
        return self.bone_world.get(bone_name, IDENTITY_QUAT.copy())

    def _set_local(self, bone_key: str, quat_local: np.ndarray, parent_key: str | None) -> None:
        bone_name = self.resolver.bone(bone_key)
        if bone_name is None:
            return
        parent_world = IDENTITY_QUAT.copy() if parent_key is None else self._world_quat_key(parent_key)
        quat_local = quat_normalize(quat_local)
        quat_world = quat_mul(parent_world, quat_local)
        self.bone_local[bone_name] = quat_local
        self.bone_world[bone_name] = quat_world

    def _compute_centers(self, points: np.ndarray) -> None:
        lhip = self._pose_pt(points, "left_hip")
        rhip = self._pose_pt(points, "right_hip")
        lsho = self._pose_pt(points, "left_shoulder")
        rsho = self._pose_pt(points, "right_shoulder")
        lear = self._pose_pt(points, "left_ear")
        rear = self._pose_pt(points, "right_ear")
        leye = self._pose_pt(points, "left_eye")
        reye = self._pose_pt(points, "right_eye")
        nose = self._pose_pt(points, "nose")

        self.centers["hip_center"] = 0.5 * (lhip + rhip)
        self.centers["shoulder_center"] = 0.5 * (lsho + rsho)
        self.centers["ear_center"] = 0.5 * (lear + rear)
        self.centers["eye_center"] = 0.5 * (leye + reye)
        self.centers["nose"] = nose

    def _solve_lower_body(self, points: np.ndarray) -> None:
        left_hip = self._pose_pt(points, "left_hip")
        right_hip = self._pose_pt(points, "right_hip")
        hip_dir = _safe_norm(left_hip - right_hip)
        if hip_dir is None:
            return
        quat_local = quat_from_unit_vectors(_REF_LOWER_BODY, hip_dir)
        self._set_local("lower_body", quat_local, parent_key=None)

    def _solve_upper_body(self, points: np.ndarray) -> None:
        left_shoulder = self._pose_pt(points, "left_shoulder")
        right_shoulder = self._pose_pt(points, "right_shoulder")
        shoulder_center = 0.5 * (left_shoulder + right_shoulder)

        shoulder_x = _safe_norm(left_shoulder - right_shoulder)
        spine_y = _safe_norm(shoulder_center)
        if shoulder_x is None or spine_y is None:
            return

        upper_body_z = _safe_norm(np.cross(shoulder_x, spine_y))
        if upper_body_z is None:
            return

        matrix = np.vstack([shoulder_x, spine_y, upper_body_z])
        quat_local = quat_from_rotation_matrix(matrix)
        self._set_local("upper_body", quat_local, parent_key=None)

    def _solve_neck(self, points: np.ndarray) -> None:
        world_left_ear = self._pose_pt(points, "left_ear")
        world_right_ear = self._pose_pt(points, "right_ear")
        world_left_shoulder = self._pose_pt(points, "left_shoulder")
        world_right_shoulder = self._pose_pt(points, "right_shoulder")

        world_to_upper = quat_inv_unit(self._world_quat_key("upper_body"))
        local_left_ear = quat_apply(world_to_upper, world_left_ear)
        local_right_ear = quat_apply(world_to_upper, world_right_ear)
        local_left_shoulder = quat_apply(world_to_upper, world_left_shoulder)
        local_right_shoulder = quat_apply(world_to_upper, world_right_shoulder)

        ear_center = 0.5 * (local_left_ear + local_right_ear)
        shoulder_center = 0.5 * (local_left_shoulder + local_right_shoulder)
        neck_dir = _safe_norm(ear_center - shoulder_center)
        if neck_dir is None:
            return

        quat_local = quat_from_unit_vectors(_REF_NECK, neck_dir)
        self._set_local("neck", quat_local, parent_key="upper_body")

    def _solve_head(self, points: np.ndarray) -> None:
        world_left_ear = self._pose_pt(points, "left_ear")
        world_right_ear = self._pose_pt(points, "right_ear")
        world_left_eye = self._pose_pt(points, "left_eye")
        world_right_eye = self._pose_pt(points, "right_eye")

        world_to_full = quat_inv_unit(self._world_quat_key("neck"))
        local_left_ear = quat_apply(world_to_full, world_left_ear)
        local_right_ear = quat_apply(world_to_full, world_right_ear)
        local_left_eye = quat_apply(world_to_full, world_left_eye)
        local_right_eye = quat_apply(world_to_full, world_right_eye)

        ear_center = 0.5 * (local_left_ear + local_right_ear)
        eye_center = 0.5 * (local_left_eye + local_right_eye)

        ear_dir = _safe_norm(local_left_ear - local_right_ear)
        bend_dir = _safe_norm(eye_center - ear_center)
        if ear_dir is None or bend_dir is None:
            return

        horizontal = quat_from_unit_vectors(_REF_HEAD_H, ear_dir)
        vertical = quat_from_unit_vectors(_REF_HEAD_V, bend_dir)
        quat_local = quat_mul(horizontal, vertical)
        self._set_local("head", quat_local, parent_key="neck")

    def _solve_leg(self, points: np.ndarray, side: str) -> None:
        hip = self._pose_pt(points, "left_hip" if side == "L" else "right_hip")
        knee = self._pose_pt(points, "left_knee" if side == "L" else "right_knee")

        world_to_lower = quat_inv_unit(self._world_quat_key("lower_body"))
        local_hip = quat_apply(world_to_lower, hip)
        local_knee = quat_apply(world_to_lower, knee)
        leg_dir = _safe_norm(local_knee - local_hip)
        if leg_dir is None:
            return

        quat_local = quat_from_unit_vectors(_REF_LEG, leg_dir)
        self._set_local(f"leg_{side}", quat_local, parent_key="lower_body")

    def _solve_knee(self, points: np.ndarray, side: str) -> None:
        knee = self._pose_pt(points, "left_knee" if side == "L" else "right_knee")
        ankle = self._pose_pt(points, "left_ankle" if side == "L" else "right_ankle")

        world_to_leg = quat_inv_unit(self._world_quat_key(f"leg_{side}"))
        local_knee = quat_apply(world_to_leg, knee)
        local_ankle = quat_apply(world_to_leg, ankle)
        knee_dir = _safe_norm(local_ankle - local_knee)
        if knee_dir is None:
            return

        ref = _REF_KNEE_L if side == "L" else _REF_KNEE_R
        quat_local = quat_from_unit_vectors(ref, knee_dir)
        self._set_local(f"knee_{side}", quat_local, parent_key=f"leg_{side}")

    def _solve_ankle(self, points: np.ndarray, side: str) -> None:
        heel = self._pose_pt(points, "left_heel" if side == "L" else "right_heel")
        foot = self._pose_pt(points, "left_foot_index" if side == "L" else "right_foot_index")

        world_to_knee = quat_inv_unit(self._world_quat_key(f"knee_{side}"))
        local_heel = quat_apply(world_to_knee, heel)
        local_foot = quat_apply(world_to_knee, foot)
        ankle_dir = _safe_norm(local_foot - local_heel)
        if ankle_dir is None:
            return

        quat_local = quat_from_unit_vectors(_REF_ANKLE, ankle_dir)
        self._set_local(f"ankle_{side}", quat_local, parent_key=f"knee_{side}")

    def _solve_arm(self, points: np.ndarray, side: str) -> None:
        shoulder = self._pose_pt(points, "left_shoulder" if side == "L" else "right_shoulder")
        elbow = self._pose_pt(points, "left_elbow" if side == "L" else "right_elbow")

        world_to_upper = quat_inv_unit(self._world_quat_key("upper_body"))
        local_shoulder = quat_apply(world_to_upper, shoulder)
        local_elbow = quat_apply(world_to_upper, elbow)
        arm_dir = _safe_norm(local_elbow - local_shoulder)
        if arm_dir is None:
            return

        ref = _REF_ARM_L if side == "L" else _REF_ARM_R
        quat_local = quat_from_unit_vectors(ref, arm_dir)
        self._set_local(f"arm_{side}", quat_local, parent_key="upper_body")

    def _solve_elbow(self, points: np.ndarray, side: str) -> None:
        elbow = self._pose_pt(points, "left_elbow" if side == "L" else "right_elbow")
        wrist = self._pose_pt(points, "left_wrist" if side == "L" else "right_wrist")

        world_to_arm = quat_inv_unit(self._world_quat_key(f"arm_{side}"))
        local_elbow = quat_apply(world_to_arm, elbow)
        local_wrist = quat_apply(world_to_arm, wrist)
        elbow_dir = _safe_norm(local_wrist - local_elbow)
        if elbow_dir is None:
            return

        ref = _REF_ELBOW_L if side == "L" else _REF_ELBOW_R
        quat_local = quat_from_unit_vectors(ref, elbow_dir)
        self._set_local(f"elbow_{side}", quat_local, parent_key=f"arm_{side}")


def solve(
    model: PmxModel, points: np.ndarray, vis: np.ndarray | None, vis_th: float = 0.2
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str | None]]:
    solver = SolverMiKaPoPose(model, vis_th=vis_th)
    return solver.solve(points, vis)

