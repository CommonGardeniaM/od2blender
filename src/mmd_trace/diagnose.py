"""Diagnostics helpers for pose->VMD pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import math

import numpy as np

from .io_pose import PoseSequence
from .mmd_io.pmx_adapter import PmxModelData
from .retarget.retargeter import _compute_model_forward, _compute_model_basis
from .retarget.center_groove import apply_axis_map


@dataclass
class Diagnostics:
    meta: Dict[str, object]
    pose_confidence: Dict[str, Dict[str, float]]
    vmd_counts: Dict[str, int]
    bone_map_missing: List[str]
    pose_pair_valid_counts: Dict[str, int]
    yaw_stats: Dict[str, float]
    forward_stats: Dict[str, float]


def collect_diagnostics(
    pose_seq: PoseSequence,
    pmx: PmxModelData,
    bone_frames: Sequence[object],
    bone_map: Dict[str, str],
    axis_map: Dict[str, object],
    min_confidence: float,
    joint_min_confidence: Dict[str, float],
) -> Diagnostics:
    model_forward = _compute_model_forward(pmx, bone_map)
    model_basis = _compute_model_basis(pmx, bone_map)
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames": len(pose_seq.frames),
        "pmx_name": pmx.name,
        "pmx_bones": len(pmx.bones),
        "min_confidence": min_confidence,
        "model_forward": (
            [float(model_forward[0]), float(model_forward[1]), float(model_forward[2])]
            if model_forward is not None
            else None
        ),
        "model_basis": (
            [
                [float(model_basis[0][0]), float(model_basis[0][1]), float(model_basis[0][2])],
                [float(model_basis[1][0]), float(model_basis[1][1]), float(model_basis[1][2])],
                [float(model_basis[2][0]), float(model_basis[2][1]), float(model_basis[2][2])],
            ]
            if model_basis is not None
            else None
        ),
    }

    bone_names = set(pmx.bone_index_map.keys())
    bone_map_missing = sorted([name for name in bone_map.values() if name not in bone_names])

    joints = [
        "pelvis",
        "spine",
        "chest",
        "neck",
        "head",
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
        "l_hip",
        "r_hip",
        "l_knee",
        "r_knee",
        "l_ankle",
        "r_ankle",
    ]
    pose_confidence = _summarize_confidence(pose_seq, joints)

    vmd_counts = _count_vmd_frames(bone_frames)

    pair_map = {
        "lower_body": ("pelvis", "spine"),
        "upper_body": ("spine", "chest"),
        "upper_body2": ("chest", "neck"),
        "neck": ("neck", "head"),
        "head": ("neck", "head"),
        "left_arm": ("l_shoulder", "l_elbow"),
        "left_elbow": ("l_elbow", "l_wrist"),
        "right_arm": ("r_shoulder", "r_elbow"),
        "right_elbow": ("r_elbow", "r_wrist"),
        "left_leg": ("l_hip", "l_knee"),
        "left_knee": ("l_knee", "l_ankle"),
        "right_leg": ("r_hip", "r_knee"),
        "right_knee": ("r_knee", "r_ankle"),
    }
    pair_counts = _count_valid_pairs(pose_seq, pair_map, min_confidence, joint_min_confidence)

    yaw_stats, forward_stats = _summarize_yaw(pose_seq, axis_map)

    return Diagnostics(
        meta=meta,
        pose_confidence=pose_confidence,
        vmd_counts=vmd_counts,
        bone_map_missing=bone_map_missing,
        pose_pair_valid_counts=pair_counts,
        yaw_stats=yaw_stats,
        forward_stats=forward_stats,
    )


def write_diagnostics(diag: Diagnostics, path: Path) -> None:
    lines = []
    lines.append("# Diagnostic Report")
    lines.append("")
    lines.append(f"Generated: {diag.meta.get('generated')}")
    lines.append("")
    lines.append("## Meta")
    for key in ["frames", "pmx_name", "pmx_bones", "min_confidence", "model_forward", "model_basis"]:
        lines.append(f"- {key}: {diag.meta.get(key)}")
    lines.append("")

    lines.append("## Bone Map Missing")
    if diag.bone_map_missing:
        for name in diag.bone_map_missing:
            lines.append(f"- {name}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Pose Confidence")
    lines.append("joint | min | mean | max | low<0.2")
    lines.append("--- | --- | --- | --- | ---")
    for name, stats in diag.pose_confidence.items():
        lines.append(
            f"{name} | {stats['min']:.3f} | {stats['mean']:.3f} | {stats['max']:.3f} | {int(stats['low'])}"
        )
    lines.append("")

    lines.append("## VMD Bone Frame Counts")
    for name in sorted(diag.vmd_counts.keys()):
        lines.append(f"- {name}: {diag.vmd_counts[name]}")
    lines.append("")

    lines.append("## Valid Pair Counts")
    for name in sorted(diag.pose_pair_valid_counts.keys()):
        lines.append(f"- {name}: {diag.pose_pair_valid_counts[name]}")
    lines.append("")

    lines.append("## Yaw Stats (deg)")
    if diag.yaw_stats:
        for key in ["min", "max", "mean", "p50", "p95"]:
            lines.append(f"- {key}: {diag.yaw_stats.get(key):.2f}")
    else:
        lines.append("- (insufficient data)")
    lines.append("")

    lines.append("## Forward Alignment")
    if diag.forward_stats:
        for key in ["first_yaw"]:
            lines.append(f"- {key}: {diag.forward_stats.get(key):.2f}")
    else:
        lines.append("- (insufficient data)")

    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize_confidence(seq: PoseSequence, joints: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name in joints:
        values: List[float] = []
        for frame in seq.frames:
            joint = frame.joints.get(name)
            if joint is None:
                continue
            values.append(float(joint.c))
        if not values:
            continue
        stats[name] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
            "low": sum(1 for v in values if v < 0.2),
        }
    return stats


def _count_vmd_frames(bone_frames: Sequence[object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for frame in bone_frames:
        name = getattr(frame, "bone_name", None)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _count_valid_pairs(
    seq: PoseSequence,
    pair_map: Dict[str, Tuple[str, str]],
    min_conf: float,
    joint_min_confidence: Dict[str, float],
) -> Dict[str, int]:
    counts = {k: 0 for k in pair_map.keys()}
    for frame in seq.frames:
        for name, (parent, child) in pair_map.items():
            p = frame.joints.get(parent)
            c = frame.joints.get(child)
            if p is None or c is None:
                continue
            p_min = float(joint_min_confidence.get(parent, min_conf))
            c_min = float(joint_min_confidence.get(child, min_conf))
            if p.c >= p_min and c.c >= c_min:
                counts[name] += 1
    return counts


def _summarize_yaw(seq: PoseSequence, axis_map: Dict[str, object]) -> Tuple[Dict[str, float], Dict[str, float]]:
    yaws: List[float] = []
    first_yaw = None
    for frame in seq.frames:
        joints = frame.joints
        if not all(k in joints for k in ("l_hip", "r_hip", "spine", "pelvis")):
            continue
        lh = joints["l_hip"]
        rh = joints["r_hip"]
        sp = joints["spine"]
        pe = joints["pelvis"]
        hip_vec = np.array([lh.x - rh.x, lh.y - rh.y, lh.z - rh.z], dtype=np.float64)
        spine_vec = np.array([sp.x - pe.x, sp.y - pe.y, sp.z - pe.z], dtype=np.float64)
        forward = np.cross(hip_vec, spine_vec)
        forward = apply_axis_map(forward, axis_map)
        yaw = math.degrees(math.atan2(forward[0], forward[2]))
        yaws.append(yaw)
        if first_yaw is None:
            first_yaw = yaw
    if not yaws:
        return {}, {}
    yaws_sorted = sorted(yaws)
    p50 = yaws_sorted[int(0.5 * (len(yaws_sorted) - 1))]
    p95 = yaws_sorted[int(0.95 * (len(yaws_sorted) - 1))]
    yaw_stats = {
        "min": min(yaws),
        "max": max(yaws),
        "mean": sum(yaws) / len(yaws),
        "p50": p50,
        "p95": p95,
    }
    forward_stats = {}
    if first_yaw is not None:
        forward_stats = {
            "first_yaw": first_yaw,
        }
    return yaw_stats, forward_stats
