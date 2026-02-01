"""Axis auto-inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .io_pose import PoseSequence
from .retarget.center_groove import apply_axis_map


@dataclass
class AxisMapInference:
    swap: List[str]
    invert: Dict[str, bool]
    score: float
    determinant: float


def infer_axis_map(
    seq: PoseSequence,
    min_confidence: float,
    joint_min_confidence: Optional[Dict[str, float]] = None,
    source: str = "hips",
    max_frames: int = 30,
    rest_basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Optional[AxisMapInference]:
    joint_min_confidence = joint_min_confidence or {}
    basis = _estimate_pose_basis(seq, min_confidence, joint_min_confidence, source, max_frames)
    if basis is None:
        return None
    right, up, forward = basis
    if rest_basis is None:
        rest_basis = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
    rest_right, rest_up, rest_forward = rest_basis

    best: Optional[AxisMapInference] = None
    axes = ["x", "y", "z"]
    signs = [-1.0, 1.0]
    for swap in _permutations(axes):
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    invert = {"x": sx < 0.0, "y": sy < 0.0, "z": sz < 0.0}
                    candidate = {"swap": list(swap), "invert": invert}
                    det = _axis_map_determinant(candidate)
                    if det < 0.0:
                        continue
                    mapped_right = apply_axis_map(right, candidate)
                    mapped_up = apply_axis_map(up, candidate)
                    mapped_forward = apply_axis_map(forward, candidate)
                    score = (
                        float(np.dot(mapped_right, rest_right))
                        + float(np.dot(mapped_up, rest_up))
                        + float(np.dot(mapped_forward, rest_forward))
                    )
                    if best is None or score > best.score:
                        best = AxisMapInference(swap=list(swap), invert=invert, score=score, determinant=det)
    return best


def _estimate_pose_basis(
    seq: PoseSequence,
    min_confidence: float,
    joint_min_confidence: Dict[str, float],
    source: str,
    max_frames: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rights = []
    ups = []
    forwards = []
    for frame in seq.frames[: max_frames if max_frames > 0 else None]:
        joints = frame.joints
        if source == "shoulders":
            required = ("l_shoulder", "r_shoulder", "spine", "pelvis")
        else:
            required = ("l_hip", "r_hip", "spine", "pelvis")
        if not all(k in joints for k in required):
            continue
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
        if any(j.c < float(joint_min_confidence.get(name, min_confidence)) for name, j in checks):
            continue
        right_vec = np.array([right.x - left.x, right.y - left.y, right.z - left.z], dtype=np.float64)
        up_vec = np.array([spine.x - pelvis.x, spine.y - pelvis.y, spine.z - pelvis.z], dtype=np.float64)
        right_n = _safe_normalize(right_vec)
        up_n = _safe_normalize(up_vec)
        if right_n is None or up_n is None:
            continue
        forward_vec = np.cross(right_n, up_n)
        forward_n = _safe_normalize(forward_vec)
        if forward_n is None:
            continue
        rights.append(right_n)
        ups.append(up_n)
        forwards.append(forward_n)
    if not rights:
        return None
    right = _safe_normalize(np.mean(np.stack(rights), axis=0))
    up = _safe_normalize(np.mean(np.stack(ups), axis=0))
    forward = _safe_normalize(np.mean(np.stack(forwards), axis=0))
    if right is None or up is None or forward is None:
        return None
    return right, up, forward


def _safe_normalize(v: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return None
    return v / norm


def _axis_map_determinant(axis_map: Dict[str, Any]) -> float:
    swap_raw = axis_map.get("swap", ["x", "y", "z"])
    if not isinstance(swap_raw, list):
        swap_raw = ["x", "y", "z"]
    invert_raw = axis_map.get("invert", {"x": False, "y": False, "z": False})
    if not isinstance(invert_raw, dict):
        invert_raw = {"x": False, "y": False, "z": False}
    swap = [str(s) for s in swap_raw]
    invert = {str(k): bool(v) for k, v in invert_raw.items()}
    axis_index = {"x": 0, "y": 1, "z": 2}
    mat = np.zeros((3, 3), dtype=np.float64)
    for out_idx, axis in enumerate(swap):
        in_idx = axis_index[axis]
        sign = -1.0 if invert.get(axis, False) else 1.0
        mat[out_idx, in_idx] = sign
    return float(np.linalg.det(mat))


def _permutations(items: List[str]) -> List[Tuple[str, str, str]]:
    return [
        (items[0], items[1], items[2]),
        (items[0], items[2], items[1]),
        (items[1], items[0], items[2]),
        (items[1], items[2], items[0]),
        (items[2], items[0], items[1]),
        (items[2], items[1], items[0]),
    ]
