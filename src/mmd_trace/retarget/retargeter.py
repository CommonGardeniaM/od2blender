"""Pose to VMD retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pypmxvmd.common.models.vmd import VmdBoneFrame

import numpy as np

from .kinematics import quat_from_two_vectors, quat_inverse, quat_multiply, quat_normalize
from .mapping import BoneMapping
from .center_groove import compute_center_groove, apply_axis_map
from ..io_pose import PoseSequence
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
    center_pattern: str = "A",
    center_lowpass: float = 0.1,
    root_joint: str = "pelvis",
    include_upper_body2: bool = True,
) -> RetargetResult:
    """Compute VMD bone frames from pose sequence."""

    mapping = BoneMapping(bone_map)
    rest_vectors = _compute_rest_vectors(seq, mapping, axis_map, min_confidence)
    parent_map = _build_parent_map(pmx)
    prev_local: Dict[str, np.ndarray] = {}
    bone_frames = []

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
            if parent.c < min_confidence or child.c < min_confidence:
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
            if v0 is None or np.linalg.norm(v) < 1e-6:
                continue

            q_global = quat_from_two_vectors(v0, v)
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
) -> Dict[str, np.ndarray]:
    rest: Dict[str, np.ndarray] = {}
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
            if parent.c < min_confidence or child.c < min_confidence:
                continue
            v0 = np.array([child.x - parent.x, child.y - parent.y, child.z - parent.z], dtype=np.float64)
            v0 = apply_axis_map(v0, axis_map)
            if np.linalg.norm(v0) < 1e-6:
                continue
            rest[bone_key] = v0
        if len(rest) >= target:
            break
    return rest


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
