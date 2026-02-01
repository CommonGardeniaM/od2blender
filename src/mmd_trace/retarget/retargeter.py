"""Pose to VMD retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pypmxvmd.common.models.vmd import VmdBoneFrame

import numpy as np

from .kinematics import (
    quat_from_two_vectors,
    quat_from_aim_up,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    rotate_vector_around_axis,
)
from .mapping import BoneMapping
from .center_groove import compute_center_groove, apply_axis_map
from ..io_pose import PoseSequence, PoseFrame
from ..mmd_io.pmx_adapter import PmxModelData
from ..mmd_io.vmd_writer import make_bone_frame


@dataclass
class RetargetResult:
    bone_frames: List[VmdBoneFrame]


def retarget_to_vmd(
    seq: PoseSequence,
    pmx: PmxModelData,
    bone_map: Dict[str, str],
    axis_map: Dict[str, object],
    min_confidence: float = 0.2,
    joint_min_confidence: Optional[Dict[str, float]] = None,
    center_pattern: str = "A",
    center_lowpass: float = 0.1,
    root_joint: str = "pelvis",
    include_upper_body2: bool = True,
    facing_mode: str = "auto",
    facing_yaw_offset_deg: float = 0.0,
    facing_source: str = "hips",
) -> RetargetResult:
    """Compute VMD bone frames from pose sequence."""

    mapping = BoneMapping(bone_map)
    rest_vectors = _compute_rest_vectors(seq, mapping, axis_map, min_confidence, joint_min_confidence)
    model_rest_vectors = _compute_model_rest_vectors(pmx, mapping, bone_map)
    rest_forward = None
    if facing_mode == "auto":
        rest_forward = _compute_model_forward(pmx, bone_map)
        if rest_forward is None:
            rest_forward = _compute_rest_forward(seq, axis_map, min_confidence, joint_min_confidence, facing_source)
    parent_map = _build_parent_map(pmx)
    prev_local: Dict[str, np.ndarray] = {}
    bone_frames = []
    joint_min_confidence = joint_min_confidence or {}

    center_positions, groove_positions = compute_center_groove(
        seq,
        axis_map=axis_map,
        pattern=center_pattern,
        lowpass_alpha=center_lowpass,
        root_joint=root_joint,
        scale=_estimate_scale(seq, pmx),
    )

    for frame_idx, frame in enumerate(seq.frames):
        global_rotations: Dict[str, np.ndarray] = {}
        frame_forward = None
        if facing_mode == "auto":
            frame_forward = _compute_frame_forward(frame, axis_map, min_confidence, joint_min_confidence, facing_source)
            if frame_forward is not None:
                yaw_offset_rad = np.radians(facing_yaw_offset_deg)
                frame_forward = rotate_vector_around_axis(
                    frame_forward,
                    np.array([0.0, 1.0, 0.0]),
                    yaw_offset_rad,
                )
        for bone_key, parent_joint, child_joint in mapping.vector_pairs():
            if bone_key == "upper_body2" and not include_upper_body2:
                continue
            bone_name = bone_map.get(bone_key)
            if not bone_name:
                continue
            if bone_name not in pmx.bone_index_map:
                continue
            if parent_joint not in frame.joints or child_joint not in frame.joints:
                continue

            parent = frame.joints[parent_joint]
            child = frame.joints[child_joint]
            parent_min = float(joint_min_confidence.get(parent_joint, min_confidence))
            child_min = float(joint_min_confidence.get(child_joint, min_confidence))
            if parent.c < parent_min or child.c < child_min:
                if bone_key in prev_local:
                    bone_frames.append(
                        make_bone_frame(
                            bone_name,
                            frame_idx,
                            [0.0, 0.0, 0.0],
                            _quat_to_euler_deg(prev_local[bone_key]),
                        )
                    )
                continue

            v = np.array([child.x - parent.x, child.y - parent.y, child.z - parent.z], dtype=np.float64)
            v = apply_axis_map(v, axis_map)
            v0 = rest_vectors.get(bone_key)
            if v0 is None:
                v0 = model_rest_vectors.get(bone_key)
            if v0 is None or np.linalg.norm(v) < 1e-6:
                continue

            q_global = _rotation_for_bone(
                bone_key,
                v0,
                v,
                rest_vectors,
                model_rest_vectors,
                frame,
                axis_map,
                min_confidence,
                joint_min_confidence,
                frame_forward,
                rest_forward,
                facing_mode,
            )
            parent_bone = parent_map.get(bone_name)
            if parent_bone and parent_bone in global_rotations:
                q_local = quat_multiply(quat_inverse(global_rotations[parent_bone]), q_global)
            else:
                q_local = q_global
            q_local = quat_normalize(q_local)

            global_rotations[bone_name] = q_global
            prev_local[bone_key] = q_local

            bone_frames.append(
                make_bone_frame(
                    bone_name,
                    frame_idx,
                    [0.0, 0.0, 0.0],
                    _quat_to_euler_deg(q_local),
                )
            )

        center_name = bone_map.get("center")
        groove_name = bone_map.get("groove")
        if center_name and center_name in pmx.bone_index_map:
            pos = center_positions[frame_idx] if frame_idx < len(center_positions) else np.zeros(3)
            bone_frames.append(
                make_bone_frame(
                    center_name,
                    frame_idx,
                    [float(pos[0]), float(pos[1]), float(pos[2])],
                    [0.0, 0.0, 0.0],
                )
            )
        if groove_name and groove_name in pmx.bone_index_map:
            pos = groove_positions[frame_idx] if frame_idx < len(groove_positions) else np.zeros(3)
            bone_frames.append(
                make_bone_frame(
                    groove_name,
                    frame_idx,
                    [float(pos[0]), float(pos[1]), float(pos[2])],
                    [0.0, 0.0, 0.0],
                )
            )

    return RetargetResult(bone_frames=bone_frames)


def _compute_rest_vectors(
    seq: PoseSequence,
    mapping: BoneMapping,
    axis_map: Dict[str, object],
    min_confidence: float,
    joint_min_confidence: Optional[Dict[str, float]],
) -> Dict[str, np.ndarray]:
    rest: Dict[str, np.ndarray] = {}
    joint_min_confidence = joint_min_confidence or {}
    if not seq.frames:
        return rest
    target = len(mapping.vector_pairs())
    for frame in seq.frames:
        for bone_key, parent_joint, child_joint in mapping.vector_pairs():
            if bone_key in rest:
                continue
            if parent_joint not in frame.joints or child_joint not in frame.joints:
                continue
            parent = frame.joints[parent_joint]
            child = frame.joints[child_joint]
            parent_min = float(joint_min_confidence.get(parent_joint, min_confidence))
            child_min = float(joint_min_confidence.get(child_joint, min_confidence))
            if parent.c < parent_min or child.c < child_min:
                continue
            v0 = np.array([child.x - parent.x, child.y - parent.y, child.z - parent.z], dtype=np.float64)
            v0 = apply_axis_map(v0, axis_map)
            if np.linalg.norm(v0) < 1e-6:
                continue
            rest[bone_key] = v0
        if len(rest) >= target:
            break
    return rest


def _rotation_for_bone(
    bone_key: str,
    v0: np.ndarray,
    v: np.ndarray,
    rest_vectors: Dict[str, np.ndarray],
    model_rest_vectors: Dict[str, np.ndarray],
    frame: PoseFrame,
    axis_map: Dict[str, object],
    min_confidence: float,
    joint_min_confidence: Dict[str, float],
    frame_forward: Optional[np.ndarray],
    rest_forward: Optional[np.ndarray],
    facing_mode: str,
) -> np.ndarray:
    if bone_key == "lower_body" and facing_mode == "auto":
        if frame_forward is None or rest_forward is None:
            return quat_from_two_vectors(v0, v)
        up0 = v0
        up1 = v
        return quat_from_aim_up(rest_forward, up0, frame_forward, up1)

    if bone_key in {"upper_body", "upper_body2", "neck", "head", "left_arm", "left_elbow", "right_arm", "right_elbow"}:
        if frame_forward is None or rest_forward is None:
            return quat_from_two_vectors(v0, v)
        return quat_from_aim_up(v0, rest_forward, v, frame_forward)

    return quat_from_two_vectors(v0, v)


def _compute_frame_forward(
    frame: PoseFrame,
    axis_map: Dict[str, object],
    min_confidence: float,
    joint_min_confidence: Dict[str, float],
    source: str,
) -> Optional[np.ndarray]:
    joints = frame.joints
    if source == "shoulders":
        required = ("l_shoulder", "r_shoulder", "spine", "pelvis")
    else:
        required = ("l_hip", "r_hip", "spine", "pelvis")
    if not all(k in joints for k in required):
        return None
    if source == "shoulders":
        left = joints["l_shoulder"]
        right = joints["r_shoulder"]
    else:
        left = joints["l_hip"]
        right = joints["r_hip"]
    spine = joints["spine"]
    pelvis = joints["pelvis"]
    checks = [("spine", spine), ("pelvis", pelvis)]
    if source == "shoulders":
        checks += [("l_shoulder", left), ("r_shoulder", right)]
    else:
        checks += [("l_hip", left), ("r_hip", right)]
    for name, joint in checks:
        if joint.c < float(joint_min_confidence.get(name, min_confidence)):
            return None
    side = np.array([left.x - right.x, left.y - right.y, left.z - right.z], dtype=np.float64)
    up = np.array([spine.x - pelvis.x, spine.y - pelvis.y, spine.z - pelvis.z], dtype=np.float64)
    forward = np.cross(side, up)
    forward = apply_axis_map(forward, axis_map)
    if np.linalg.norm(forward) < 1e-6:
        return None
    return forward


def _compute_rest_forward(
    seq: PoseSequence,
    axis_map: Dict[str, object],
    min_confidence: float,
    joint_min_confidence: Optional[Dict[str, float]],
    source: str,
) -> Optional[np.ndarray]:
    joint_min_confidence = joint_min_confidence or {}
    for frame in seq.frames:
        forward = _compute_frame_forward(frame, axis_map, min_confidence, joint_min_confidence, source)
        if forward is not None:
            return forward
    return None


def _compute_model_rest_vectors(
    pmx: PmxModelData,
    mapping: BoneMapping,
    bone_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    rest: Dict[str, np.ndarray] = {}
    for bone_key, parent_key, child_key in mapping.bone_pairs():
        parent_name = bone_map.get(parent_key)
        child_name = bone_map.get(child_key)
        if not parent_name or not child_name:
            continue
        parent_pos = _pmx_bone_pos(pmx, parent_name)
        child_pos = _pmx_bone_pos(pmx, child_name)
        if parent_pos is None or child_pos is None:
            continue
        v0 = np.array(
            [child_pos[0] - parent_pos[0], child_pos[1] - parent_pos[1], child_pos[2] - parent_pos[2]],
            dtype=np.float64,
        )
        if np.linalg.norm(v0) < 1e-6:
            continue
        rest[bone_key] = v0
    return rest


def _compute_model_forward(pmx: PmxModelData, bone_map: Dict[str, str]) -> Optional[np.ndarray]:
    left_name = bone_map.get("left_leg", "左足")
    right_name = bone_map.get("right_leg", "右足")
    upper_name = bone_map.get("upper_body", "上半身")
    lower_name = bone_map.get("lower_body", "下半身")
    left = _pmx_bone_pos(pmx, left_name)
    right = _pmx_bone_pos(pmx, right_name)
    upper = _pmx_bone_pos(pmx, upper_name)
    lower = _pmx_bone_pos(pmx, lower_name)
    if left is None or right is None or upper is None or lower is None:
        return None
    right_vec = np.array([right[0] - left[0], right[1] - left[1], right[2] - left[2]], dtype=np.float64)
    up_vec = np.array([upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]], dtype=np.float64)
    if np.linalg.norm(right_vec) < 1e-6 or np.linalg.norm(up_vec) < 1e-6:
        return None
    forward = np.cross(right_vec, up_vec)
    if np.linalg.norm(forward) < 1e-6:
        return None
    return forward / np.linalg.norm(forward)


def _compute_model_basis(pmx: PmxModelData, bone_map: Dict[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    left_name = bone_map.get("left_leg", "左足")
    right_name = bone_map.get("right_leg", "右足")
    upper_name = bone_map.get("upper_body", "上半身")
    lower_name = bone_map.get("lower_body", "下半身")
    left = _pmx_bone_pos(pmx, left_name)
    right = _pmx_bone_pos(pmx, right_name)
    upper = _pmx_bone_pos(pmx, upper_name)
    lower = _pmx_bone_pos(pmx, lower_name)
    if left is None or right is None or upper is None or lower is None:
        return None
    right_vec = np.array([right[0] - left[0], right[1] - left[1], right[2] - left[2]], dtype=np.float64)
    up_vec = np.array([upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]], dtype=np.float64)
    if np.linalg.norm(right_vec) < 1e-6 or np.linalg.norm(up_vec) < 1e-6:
        return None
    right_n = right_vec / np.linalg.norm(right_vec)
    up_n = up_vec / np.linalg.norm(up_vec)
    forward = np.cross(right_n, up_n)
    if np.linalg.norm(forward) < 1e-6:
        return None
    forward_n = forward / np.linalg.norm(forward)
    return right_n, up_n, forward_n


def _build_parent_map(pmx: PmxModelData) -> Dict[str, Optional[str]]:
    parent_map = {}
    for idx, bone in enumerate(pmx.bones):
        parent_idx = bone.parent_index
        if parent_idx is None or parent_idx < 0 or parent_idx >= len(pmx.bones):
            parent_map[bone.name] = None
        else:
            parent_map[bone.name] = pmx.bones[parent_idx].name
    return parent_map


def _estimate_scale(seq: PoseSequence, pmx: PmxModelData) -> float:
    if not seq.frames:
        return 1.0
    heights = []
    for frame in seq.frames[: min(len(seq.frames), 30)]:
        if "head" not in frame.joints or "l_ankle" not in frame.joints or "r_ankle" not in frame.joints:
            continue
        head = frame.joints["head"]
        ankle_y = min(frame.joints["l_ankle"].y, frame.joints["r_ankle"].y)
        heights.append(head.y - ankle_y)
    if not heights:
        return 1.0
    h_est = float(np.mean(heights))
    pmx_head = _pmx_bone_pos(pmx, "頭")
    pmx_ankle = _pmx_bone_pos(pmx, "左足首") or _pmx_bone_pos(pmx, "右足首")
    if pmx_head is None or pmx_ankle is None:
        return 1.0
    h_pmx = float(pmx_head[1] - pmx_ankle[1])
    if abs(h_est) < 1e-6:
        return 1.0
    return h_pmx / h_est


def _pmx_bone_pos(pmx: PmxModelData, name: str) -> Optional[List[float]]:
    idx = pmx.bone_index_map.get(name)
    if idx is None:
        return None
    return pmx.bones[idx].position


def _quat_to_euler_deg(q: np.ndarray) -> List[float]:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.degrees(np.sign(sinp) * (np.pi / 2))
    else:
        pitch = np.degrees(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return [float(roll), float(pitch), float(yaw)]
