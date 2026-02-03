"""Pose processing utilities for bone rotation computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmd_trace.io.pmx import PmxModel, PmxBone


MP = {
    "NOSE": 0,
    "L_EAR": 7,
    "R_EAR": 8,
    "L_SHO": 11,
    "R_SHO": 12,
    "L_ELB": 13,
    "R_ELB": 14,
    "L_WRI": 15,
    "R_WRI": 16,
    "L_PNK": 17,
    "R_PNK": 18,
    "L_IDX": 19,
    "R_IDX": 20,
    "L_THM": 21,
    "R_THM": 22,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNE": 25,
    "R_KNE": 26,
    "L_ANK": 27,
    "R_ANK": 28,
    "L_HEE": 29,
    "R_HEE": 30,
    "L_FOO": 31,
    "R_FOO": 32,
}

POSE_IDX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

HAND_IDX = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
EPS = 1e-9


BONE_CANDIDATES: Dict[str, List[str]] = {
    "center": ["センター", "center"],
    "groove": ["グルーブ", "groove"],
    "lower_body": ["下半身", "lower body"],
    "upper_body": ["上半身", "upper body"],
    "upper_body2": ["上半身2", "上半身２", "upper body2", "upper body 2"],
    "neck": ["首", "neck"],
    "head": ["頭", "head"],
    "shoulder_L": ["左肩", "shoulder_L", "shoulder.L"],
    "shoulder_R": ["右肩", "shoulder_R", "shoulder.R"],
    "arm_L": ["左腕", "arm_L", "arm.L"],
    "arm_R": ["右腕", "arm_R", "arm.R"],
    "elbow_L": ["左ひじ", "左肘", "elbow_L", "elbow.L"],
    "elbow_R": ["右ひじ", "右肘", "elbow_R", "elbow.R"],
    "wrist_L": ["左手首", "wrist_L", "wrist.L"],
    "wrist_R": ["右手首", "wrist_R", "wrist.R"],
    "wrist_twist_L": ["左手首捩", "左手首捩り", "wrist_twist_L", "wrist_twist.L"],
    "wrist_twist_R": ["右手首捩", "右手首捩り", "wrist_twist_R", "wrist_twist.R"],
    "leg_L": ["左足", "leg_L", "leg.L"],
    "leg_R": ["右足", "leg_R", "leg.R"],
    "knee_L": ["左ひざ", "左膝", "knee_L", "knee.L"],
    "knee_R": ["右ひざ", "右膝", "knee_R", "knee.R"],
    "ankle_L": ["左足首", "ankle_L", "ankle.L"],
    "ankle_R": ["右足首", "ankle_R", "ankle.R"],
    "toe_L": ["左つま先", "L toe", "toe_L", "toe.L"],
    "toe_R": ["右つま先", "R toe", "toe_R", "toe.R"],
}

FINGER_CANDIDATES: Dict[str, List[List[str]]] = {
    "thumb": [
        ["親指１", "親指０", "thumb0", "thumb1"],
        ["親指２", "thumb1", "thumb2"],
        ["親指３", "thumb2", "thumb3"],
    ],
    "index": [
        ["人指１", "人差指１", "index1", "fore1"],
        ["人指２", "人差指２", "index2", "fore2"],
        ["人指３", "人差指３", "index3", "fore3"],
    ],
    "middle": [
        ["中指１", "middle1"],
        ["中指２", "middle2"],
        ["中指３", "middle3"],
    ],
    "ring": [
        ["薬指１", "ring1", "third1"],
        ["薬指２", "ring2", "third2"],
        ["薬指３", "ring3", "third3"],
    ],
    "pinky": [
        ["小指１", "little1"],
        ["小指２", "little2"],
        ["小指３", "little3"],
    ],
}

FINGER_RATIO_2 = 0.7
FINGER_RATIO_3 = 0.4


def _norm(vec: np.ndarray, eps: float = EPS) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length < eps:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return vec / length


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64) / (np.dot(q, q) + EPS)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _norm(axis)
    s = np.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)], dtype=np.float64)


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _norm(a)
    b = _norm(b)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return IDENTITY_QUAT.copy()
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + d], dtype=np.float64)
    return q / (np.linalg.norm(q) + EPS)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_inv(q))[:3]


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + EPS)


def make_basis(right: np.ndarray, up: np.ndarray, forward_hint: Optional[np.ndarray] = None) -> np.ndarray:
    r = _norm(right)
    u = up - np.dot(up, r) * r
    if np.linalg.norm(u) < 1e-6:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, r)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = tmp - np.dot(tmp, r) * r
    u = _norm(u)
    f = np.cross(r, u)
    if np.linalg.norm(f) < 1e-6:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    f = _norm(f)

    if forward_hint is not None and np.dot(f, forward_hint) < 0:
        u = -u
        f = -f
    return np.stack([r, u, f], axis=1)


def rot_from_basis(rest_R: np.ndarray, pose_R: np.ndarray) -> np.ndarray:
    return quat_from_matrix(pose_R @ rest_R.T)


def map_points(
    points: np.ndarray,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    axis_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map MediaPipe points into MMD coordinate space (axis scale + Y flip)."""
    if axis_matrix is not None:
        M = np.asarray(axis_matrix, dtype=np.float64)
        if M.shape != (3, 3):
            raise ValueError("axis_matrix must be 3x3")
    else:
        M = np.diag([axis_x, axis_y, axis_z]).astype(np.float64)
    mapped = np.asarray(points, dtype=np.float64) @ M.T
    mapped[:, 1] *= -1.0
    return mapped


def parent_name(bones: List[PmxBone], bone: PmxBone) -> Optional[str]:
    """Get parent bone name."""
    if bone.parent_index < 0:
        return None
    return bones[bone.parent_index].name


def rest_dir(bones: List[PmxBone], bone: PmxBone) -> np.ndarray:
    """Calculate rest direction from bone position to tail."""
    origin = bone.position
    if bone.tail_index is not None and bone.tail_index >= 0:
        target = bones[bone.tail_index].position
        direction = target - origin
    elif bone.tail_offset is not None:
        direction = bone.tail_offset
    else:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _norm(direction)


def extract_bend_angle(q: np.ndarray, bend_axis: np.ndarray) -> float:
    """Extract signed bend angle (radians) from quaternion around bend_axis."""
    qn = q / (np.linalg.norm(q) + EPS)
    w = float(np.clip(abs(qn[3]), -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    if angle < 1e-6:
        return 0.0
    s = np.sqrt(max(1.0 - w * w, 0.0))
    axis = qn[:3] / (s + EPS)
    if qn[3] < 0.0:
        axis = -axis
    if np.dot(axis, bend_axis) < 0.0:
        angle = -angle
    return angle


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _expand_side_candidates(names: List[str], side: str) -> List[str]:
    prefix = "左" if side == "L" else "右"
    suffix = "L" if side == "L" else "R"
    expanded: List[str] = []
    for name in names:
        if _has_non_ascii(name):
            expanded.append(f"{prefix}{name}")
            continue
        if name.endswith(("_L", "_R", ".L", ".R")):
            expanded.append(name)
            continue
        expanded.append(f"{name}_{suffix}")
        expanded.append(f"{name}.{suffix}")
    return expanded


def find_bone_name(model: PmxModel, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        bone = model.get_bone(name)
        if bone is not None:
            return bone.name
    return None


@dataclass
class BoneResolver:
    model: PmxModel
    bones: Dict[str, Optional[str]] = field(init=False)
    fingers: Dict[str, List[Optional[str]]] = field(init=False)
    name_to_bone: Dict[str, PmxBone] = field(init=False)

    def __post_init__(self) -> None:
        self.name_to_bone = {bone.name: bone for bone in self.model.bones}
        self.bones = {}
        for key, candidates in BONE_CANDIDATES.items():
            self.bones[key] = find_bone_name(self.model, candidates)
        self.fingers = {}
        for side in ["L", "R"]:
            for finger, chains in FINGER_CANDIDATES.items():
                resolved: List[Optional[str]] = []
                for joint_names in chains:
                    expanded = _expand_side_candidates(joint_names, side)
                    resolved.append(find_bone_name(self.model, expanded))
                self.fingers[f"{side}_{finger}"] = resolved

    def bone(self, key: str) -> Optional[str]:
        return self.bones.get(key)

    def finger_chain(self, side: str, finger: str) -> List[Optional[str]]:
        return self.fingers.get(f"{side}_{finger}", [None, None, None])

    def get_bone(self, name: Optional[str]) -> Optional[PmxBone]:
        if name is None:
            return None
        return self.name_to_bone.get(name)


class PoseSolver:
    """Solve MMD bone rotations from MediaPipe pose/hand landmarks."""

    def __init__(self, model: PmxModel, vis_th: float = 0.2) -> None:
        self.model = model
        self.vis_th = vis_th
        self.resolver = BoneResolver(model)
        self.bones = model.bones
        self.pose: Optional[np.ndarray] = None
        self.vis: Optional[np.ndarray] = None
        self.left_hand: Optional[np.ndarray] = None
        self.right_hand: Optional[np.ndarray] = None
        self.bone_local: Dict[str, np.ndarray] = {}
        self.bone_world: Dict[str, np.ndarray] = {}
        self.hip_center: Optional[np.ndarray] = None
        self.shoulder_center: Optional[np.ndarray] = None
        self.ear_center: Optional[np.ndarray] = None
        self.eye_center: Optional[np.ndarray] = None
        self.nose: Optional[np.ndarray] = None

    def solve(
        self,
        pose_points: np.ndarray,
        vis: Optional[np.ndarray],
        left_hand: Optional[np.ndarray] = None,
        right_hand: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        self._reset()
        if not self._adopt_landmarks(pose_points, vis, left_hand, right_hand):
            return {}
        self._compute_centers()
        self._solve_torso()
        self._solve_neck_head()
        self._solve_leg("L")
        self._solve_leg("R")
        self._solve_arm("L")
        self._solve_arm("R")
        self._solve_fingers("L")
        self._solve_fingers("R")
        return self.bone_local

    def _reset(self) -> None:
        self.pose = None
        self.vis = None
        self.left_hand = None
        self.right_hand = None
        self.bone_local = {}
        self.bone_world = {}
        self.hip_center = None
        self.shoulder_center = None
        self.ear_center = None
        self.eye_center = None
        self.nose = None

    def _adopt_landmarks(
        self,
        pose_points: np.ndarray,
        vis: Optional[np.ndarray],
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
    ) -> bool:
        if pose_points is None or len(pose_points) != 33:
            return False
        self.pose = np.asarray(pose_points, dtype=np.float64)
        if vis is not None and len(vis) == 33:
            self.vis = np.asarray(vis, dtype=np.float64)
        if left_hand is not None and len(left_hand) == 21:
            self.left_hand = np.asarray(left_hand, dtype=np.float64)
        if right_hand is not None and len(right_hand) == 21:
            self.right_hand = np.asarray(right_hand, dtype=np.float64)
        return True

    def _pose_pt(self, name: str) -> Optional[np.ndarray]:
        if self.pose is None:
            return None
        idx = POSE_IDX[name]
        if self.vis is not None and self.vis[idx] < self.vis_th:
            return None
        return self.pose[idx]

    def _hand_pt(self, side: str, name: str) -> Optional[np.ndarray]:
        data = self.left_hand if side == "L" else self.right_hand
        if data is None:
            return None
        return data[HAND_IDX[name]]

    def _compute_centers(self) -> None:
        lhip = self._pose_pt("left_hip")
        rhip = self._pose_pt("right_hip")
        lsho = self._pose_pt("left_shoulder")
        rsho = self._pose_pt("right_shoulder")
        lear = self._pose_pt("left_ear")
        rear = self._pose_pt("right_ear")
        leye = self._pose_pt("left_eye")
        reye = self._pose_pt("right_eye")
        self.nose = self._pose_pt("nose")

        self.hip_center = self._mid(lhip, rhip)
        self.shoulder_center = self._mid(lsho, rsho)
        self.ear_center = self._mid(lear, rear)
        self.eye_center = self._mid(leye, reye)

    @staticmethod
    def _mid(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None or b is None:
            return None
        return 0.5 * (a + b)

    def _get_parent_world(self, bone_name: str) -> np.ndarray:
        bone = self.resolver.get_bone(bone_name)
        if bone is None or bone.parent_index < 0:
            return IDENTITY_QUAT.copy()
        parent = self.bones[bone.parent_index].name
        return self.bone_world.get(parent, IDENTITY_QUAT.copy())

    def _set_local(self, bone_name: Optional[str], local_q: np.ndarray) -> None:
        if bone_name is None:
            return
        parent_world = self._get_parent_world(bone_name)
        local_q = local_q / (np.linalg.norm(local_q) + EPS)
        self.bone_local[bone_name] = local_q
        self.bone_world[bone_name] = quat_mul(parent_world, local_q)

    def _rest_matrix(self, bone_name: Optional[str]) -> Optional[np.ndarray]:
        bone = self.resolver.get_bone(bone_name)
        if bone is None:
            return None
        return self.model.get_bone_rest_matrix(bone.index)

    def _rest_dir(self, bone_name: Optional[str]) -> Optional[np.ndarray]:
        bone = self.resolver.get_bone(bone_name)
        if bone is None:
            return None
        return rest_dir(self.bones, bone)

    def _rest_dir_or(self, bone_name: Optional[str], fallback: np.ndarray) -> np.ndarray:
        ref = self._rest_dir(bone_name)
        if ref is None or np.linalg.norm(ref) < 1e-6:
            return fallback
        return ref

    @staticmethod
    def _dir(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None or b is None:
            return None
        d = b - a
        if np.linalg.norm(d) < 1e-6:
            return None
        return _norm(d)

    def _to_local(self, parent_world: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
        return quat_rotate(quat_inv(parent_world), vec_world)

    def _solve_torso(self) -> None:
        lhip = self._pose_pt("left_hip")
        rhip = self._pose_pt("right_hip")
        lsho = self._pose_pt("left_shoulder")
        rsho = self._pose_pt("right_shoulder")

        lower_body = self.resolver.bone("lower_body")
        if lower_body is not None and lhip is not None and rhip is not None:
            hip_dir_world = lhip - rhip
            parent_world = self._get_parent_world(lower_body)
            hip_dir_local = self._to_local(parent_world, hip_dir_world)
            rest_R = self._rest_matrix(lower_body)
            ref_right = rest_R[:, 0] if rest_R is not None else np.array([1.0, 0.0, 0.0])
            q_lower = quat_from_two_vectors(ref_right, hip_dir_local)
            self._set_local(lower_body, q_lower)

        upper_body = self.resolver.bone("upper_body")
        if upper_body is not None and lsho is not None and rsho is not None and self.shoulder_center is not None:
            right_world = lsho - rsho
            up_world = self.shoulder_center
            fwd_world = None
            if self.nose is not None:
                fwd_world = self.nose - self.shoulder_center
            parent_world = self._get_parent_world(upper_body)
            right_local = self._to_local(parent_world, right_world)
            up_local = self._to_local(parent_world, up_world)
            fwd_local = self._to_local(parent_world, fwd_world) if fwd_world is not None else None
            pose_R = make_basis(right_local, up_local, fwd_local)
            rest_R = self._rest_matrix(upper_body)
            q_upper = rot_from_basis(rest_R, pose_R) if rest_R is not None else quat_from_matrix(pose_R)
            self._set_local(upper_body, q_upper)

        upper_body2 = self.resolver.bone("upper_body2")
        if (
            upper_body2 is not None
            and lsho is not None
            and rsho is not None
            and self.shoulder_center is not None
            and self.ear_center is not None
        ):
            right_world = lsho - rsho
            up_world = self.ear_center - self.shoulder_center
            fwd_world = None
            if self.nose is not None:
                fwd_world = self.nose - self.ear_center
            parent_world = self._get_parent_world(upper_body2)
            right_local = self._to_local(parent_world, right_world)
            up_local = self._to_local(parent_world, up_world)
            fwd_local = self._to_local(parent_world, fwd_world) if fwd_world is not None else None
            pose_R = make_basis(right_local, up_local, fwd_local)
            rest_R = self._rest_matrix(upper_body2)
            q_upper2 = rot_from_basis(rest_R, pose_R) if rest_R is not None else quat_from_matrix(pose_R)
            self._set_local(upper_body2, q_upper2)

    def _solve_neck_head(self) -> None:
        neck = self.resolver.bone("neck")
        if neck is not None and self.shoulder_center is not None and self.ear_center is not None:
            dir_world = self.ear_center - self.shoulder_center
            parent_world = self._get_parent_world(neck)
            dir_local = self._to_local(parent_world, dir_world)
            ref = self._rest_dir_or(neck, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            q_neck = quat_from_two_vectors(ref, dir_local)
            self._set_local(neck, q_neck)

        head = self.resolver.bone("head")
        lear = self._pose_pt("left_ear")
        rear = self._pose_pt("right_ear")
        if head is None or lear is None or rear is None or self.ear_center is None:
            return

        up_world = None
        if self.eye_center is not None:
            up_world = self.eye_center - self.ear_center
        elif self.nose is not None:
            up_world = self.nose - self.ear_center
        if up_world is None:
            return

        right_world = rear - lear
        fwd_world = None
        if self.nose is not None:
            fwd_world = self.nose - self.ear_center

        parent_world = self._get_parent_world(head)
        right_local = self._to_local(parent_world, right_world)
        up_local = self._to_local(parent_world, up_world)
        fwd_local = self._to_local(parent_world, fwd_world) if fwd_world is not None else None
        pose_R = make_basis(right_local, up_local, fwd_local)
        rest_R = self._rest_matrix(head)
        q_head = rot_from_basis(rest_R, pose_R) if rest_R is not None else quat_from_matrix(pose_R)
        self._set_local(head, q_head)

    def _solve_leg(self, side: str) -> None:
        hip = self._pose_pt("left_hip" if side == "L" else "right_hip")
        knee = self._pose_pt("left_knee" if side == "L" else "right_knee")
        ankle = self._pose_pt("left_ankle" if side == "L" else "right_ankle")
        foot = self._pose_pt("left_foot_index" if side == "L" else "right_foot_index")

        leg = self.resolver.bone("leg_L" if side == "L" else "leg_R")
        knee_bone = self.resolver.bone("knee_L" if side == "L" else "knee_R")
        ankle_bone = self.resolver.bone("ankle_L" if side == "L" else "ankle_R")

        if leg is not None:
            dir_world = self._dir(hip, knee)
            if dir_world is not None:
                parent_world = self._get_parent_world(leg)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(leg, np.array([0.0, -1.0, 0.0], dtype=np.float64))
                q_leg = quat_from_two_vectors(ref, dir_local)
                self._set_local(leg, q_leg)

        if knee_bone is not None:
            dir_world = self._dir(knee, ankle)
            if dir_world is not None:
                parent_world = self._get_parent_world(knee_bone)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(knee_bone, np.array([0.0, -1.0, 0.0], dtype=np.float64))
                q_knee = quat_from_two_vectors(ref, dir_local)
                self._set_local(knee_bone, q_knee)

        if ankle_bone is not None and foot is not None:
            dir_world = self._dir(ankle, foot)
            if dir_world is not None:
                parent_world = self._get_parent_world(ankle_bone)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(ankle_bone, np.array([0.0, -1.0, 0.0], dtype=np.float64))
                q_ankle = quat_from_two_vectors(ref, dir_local)
                self._set_local(ankle_bone, q_ankle)

    def _solve_arm(self, side: str) -> None:
        sho = self._pose_pt("left_shoulder" if side == "L" else "right_shoulder")
        elb = self._pose_pt("left_elbow" if side == "L" else "right_elbow")
        wri = self._pose_pt("left_wrist" if side == "L" else "right_wrist")
        if sho is None or elb is None or wri is None:
            return

        shoulder = self.resolver.bone("shoulder_L" if side == "L" else "shoulder_R")
        arm = self.resolver.bone("arm_L" if side == "L" else "arm_R")
        elbow = self.resolver.bone("elbow_L" if side == "L" else "elbow_R")
        wrist = self.resolver.bone("wrist_L" if side == "L" else "wrist_R")
        twist = self.resolver.bone("wrist_twist_L" if side == "L" else "wrist_twist_R")

        if shoulder is not None and self.shoulder_center is not None:
            dir_world = self._dir(self.shoulder_center, sho)
            if dir_world is not None:
                parent_world = self._get_parent_world(shoulder)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(shoulder, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                q_sho = quat_from_two_vectors(ref, dir_local)
                self._set_local(shoulder, q_sho)

        if arm is not None:
            dir_world = self._dir(sho, elb)
            if dir_world is not None:
                parent_world = self._get_parent_world(arm)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(arm, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                q_arm = quat_from_two_vectors(ref, dir_local)
                self._set_local(arm, q_arm)

        if elbow is not None:
            dir_world = self._dir(elb, wri)
            if dir_world is not None:
                parent_world = self._get_parent_world(elbow)
                dir_local = self._to_local(parent_world, dir_world)
                ref = self._rest_dir_or(elbow, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                q_elb = quat_from_two_vectors(ref, dir_local)
                self._set_local(elbow, q_elb)

        if wrist is None:
            return

        twist_q = self._compute_wrist_twist(side, elb, wri)
        if twist is not None and twist_q is not None:
            self._set_local(twist, twist_q)

        wrist_dir_world = self._wrist_target_direction(side, wri)
        if wrist_dir_world is None:
            return

        if twist_q is not None and twist is None:
            parent_world = self._get_parent_world(wrist)
            parent_world = quat_mul(parent_world, twist_q)
            dir_local = self._to_local(parent_world, wrist_dir_world)
            ref = self._rest_dir_or(wrist, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            q_wrist_local = quat_from_two_vectors(ref, dir_local)
            q_wrist_local = quat_mul(twist_q, q_wrist_local)
        else:
            parent_world = self._get_parent_world(wrist)
            dir_local = self._to_local(parent_world, wrist_dir_world)
            ref = self._rest_dir_or(wrist, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            q_wrist_local = quat_from_two_vectors(ref, dir_local)

        self._set_local(wrist, q_wrist_local)

    def _compute_wrist_twist(
        self,
        side: str,
        elbow_pt: np.ndarray,
        wrist_pt: np.ndarray,
    ) -> Optional[np.ndarray]:
        hand = self.left_hand if side == "L" else self.right_hand
        if hand is None:
            return None
        idx = self._hand_pt(side, "index_mcp")
        ring = self._hand_pt(side, "ring_mcp")
        if idx is None:
            return None
        if ring is None:
            ring = self._hand_pt(side, "pinky_mcp")
        if ring is None:
            return None

        forearm_world = wrist_pt - elbow_pt
        if np.linalg.norm(forearm_world) < 1e-6:
            return None

        elbow_bone = self.resolver.bone("elbow_L" if side == "L" else "elbow_R")
        if elbow_bone is None:
            return None
        parent_world = self._get_parent_world(elbow_bone)
        axis_local = _norm(self._to_local(parent_world, forearm_world))
        palm_side_local = self._to_local(parent_world, idx - ring)

        rest_idx = self._finger_base_pos(side, "index")
        rest_ring = self._finger_base_pos(side, "ring")
        if rest_idx is None:
            return None
        if rest_ring is None:
            rest_ring = self._finger_base_pos(side, "pinky")
        if rest_ring is None:
            return None

        rest_side = rest_idx - rest_ring
        ref = rest_side - np.dot(rest_side, axis_local) * axis_local
        tgt = palm_side_local - np.dot(palm_side_local, axis_local) * axis_local
        if np.linalg.norm(ref) < 1e-6 or np.linalg.norm(tgt) < 1e-6:
            return None
        ref = _norm(ref)
        tgt = _norm(tgt)
        angle = float(np.arctan2(np.dot(axis_local, np.cross(ref, tgt)), np.dot(ref, tgt)))
        return quat_from_axis_angle(axis_local, angle)

    def _wrist_target_direction(self, side: str, wrist_pt: np.ndarray) -> Optional[np.ndarray]:
        mid = self._hand_pt(side, "middle_mcp")
        if mid is not None:
            return mid - wrist_pt

        idx = self._pose_pt("left_index" if side == "L" else "right_index")
        pnk = self._pose_pt("left_pinky" if side == "L" else "right_pinky")
        thm = self._pose_pt("left_thumb" if side == "L" else "right_thumb")
        if idx is None or pnk is None or thm is None:
            return None
        center = (idx + pnk + thm) / 3.0
        return center - wrist_pt

    def _finger_base_pos(self, side: str, finger: str) -> Optional[np.ndarray]:
        chain = self.resolver.finger_chain(side, finger)
        base_name = chain[0]
        bone = self.resolver.get_bone(base_name)
        if bone is None:
            return None
        return bone.position

    def _solve_fingers(self, side: str) -> None:
        hand = self.left_hand if side == "L" else self.right_hand
        if hand is None:
            return

        wrist_name = self.resolver.bone("wrist_L" if side == "L" else "wrist_R")
        if wrist_name is None or wrist_name not in self.bone_world:
            return

        parent_world = self.bone_world[wrist_name]
        wrist_pt = self._hand_pt(side, "wrist")
        idx = self._hand_pt(side, "index_mcp")
        pnk = self._hand_pt(side, "pinky_mcp")
        if wrist_pt is None or idx is None or pnk is None:
            return

        palm_normal = np.cross(idx - wrist_pt, pnk - wrist_pt)
        if np.linalg.norm(palm_normal) < 1e-6:
            return
        palm_normal_local = _norm(self._to_local(parent_world, palm_normal))

        finger_map = {
            "thumb": ("thumb_mcp", "thumb_ip"),
            "index": ("index_mcp", "index_pip"),
            "middle": ("middle_mcp", "middle_pip"),
            "ring": ("ring_mcp", "ring_pip"),
            "pinky": ("pinky_mcp", "pinky_pip"),
        }

        for finger, (mcp_name, pip_name) in finger_map.items():
            mcp = self._hand_pt(side, mcp_name)
            pip = self._hand_pt(side, pip_name)
            if mcp is None or pip is None:
                continue
            chain = self.resolver.finger_chain(side, finger)
            base_name, mid_name, tip_name = chain
            if base_name is None:
                continue
            dir_world = pip - mcp
            if np.linalg.norm(dir_world) < 1e-6:
                continue
            dir_local = self._to_local(parent_world, dir_world)
            ref = self._rest_dir_or(base_name, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            q_base = quat_from_two_vectors(ref, dir_local)
            self._set_local(base_name, q_base)

            bend_axis_parent = np.cross(palm_normal_local, _norm(dir_local))
            if np.linalg.norm(bend_axis_parent) < 1e-6:
                continue
            bend_axis_parent = _norm(bend_axis_parent)
            bend_angle = extract_bend_angle(q_base, bend_axis_parent)
            axis_in_base = quat_rotate(quat_inv(q_base), bend_axis_parent)

            if mid_name is not None:
                q_mid = quat_from_axis_angle(axis_in_base, bend_angle * FINGER_RATIO_2)
                self._set_local(mid_name, q_mid)

            if tip_name is not None:
                axis_in_tip = axis_in_base
                if mid_name is not None:
                    q_mid = self.bone_local.get(mid_name)
                    if q_mid is not None:
                        axis_in_tip = quat_rotate(quat_inv(q_mid), axis_in_base)
                q_tip = quat_from_axis_angle(axis_in_tip, bend_angle * FINGER_RATIO_3)
                self._set_local(tip_name, q_tip)


def compute_scale(points: np.ndarray, model: PmxModel) -> float:
    """Compute scale factor from MediaPipe to PMX model."""
    if points is None or len(points) != 33:
        return 1.0

    resolver = BoneResolver(model)
    name_to_bone = resolver.name_to_bone

    left_leg = (
        np.linalg.norm(points[POSE_IDX["left_knee"]] - points[POSE_IDX["left_hip"]])
        + np.linalg.norm(points[POSE_IDX["left_ankle"]] - points[POSE_IDX["left_knee"]])
    )
    right_leg = (
        np.linalg.norm(points[POSE_IDX["right_knee"]] - points[POSE_IDX["right_hip"]])
        + np.linalg.norm(points[POSE_IDX["right_ankle"]] - points[POSE_IDX["right_knee"]])
    )
    mp_leg = (left_leg + right_leg) * 0.5

    leg_l = resolver.bone("leg_L")
    knee_l = resolver.bone("knee_L")
    ankle_l = resolver.bone("ankle_L")
    leg_r = resolver.bone("leg_R")
    knee_r = resolver.bone("knee_R")
    ankle_r = resolver.bone("ankle_R")
    if not all([leg_l, knee_l, ankle_l, leg_r, knee_r, ankle_r]):
        return 1.0

    pmx_left = (
        np.linalg.norm(name_to_bone[knee_l].position - name_to_bone[leg_l].position)
        + np.linalg.norm(name_to_bone[ankle_l].position - name_to_bone[knee_l].position)
    )
    pmx_right = (
        np.linalg.norm(name_to_bone[knee_r].position - name_to_bone[leg_r].position)
        + np.linalg.norm(name_to_bone[ankle_r].position - name_to_bone[knee_r].position)
    )
    pmx_leg = (pmx_left + pmx_right) * 0.5

    return float(pmx_leg / (mp_leg + EPS))


def build_rotations(
    points: np.ndarray,
    vis: np.ndarray,
    model: PmxModel,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    axis_matrix: Optional[np.ndarray] = None,
    vis_th: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float, float]]]:
    """Build bone rotations from MediaPipe landmarks.

    Args:
        points: MediaPipe 3D landmarks (33 points)
        vis: Visibility scores for each landmark
        model: PMX model data
        axis_x, axis_y, axis_z: Axis scaling factors
        axis_matrix: Optional 3x3 transform matrix (overrides axis_x/y/z)
        vis_th: Visibility threshold

    Returns:
        Tuple of (bone_quaternions, bone_translations)
    """
    points_mapped = map_points(points, axis_x, axis_y, axis_z, axis_matrix)
    solver = PoseSolver(model, vis_th=vis_th)
    bone_quat_local = solver.solve(points_mapped, vis)
    bone_trans: Dict[str, Tuple[float, float, float]] = {}
    return bone_quat_local, bone_trans
