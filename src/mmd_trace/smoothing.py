"""Pose smoothing and gap filling."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .io_pose import PoseFrame, PoseJoint, PoseSequence


def smooth_pose_sequence(
    seq: PoseSequence,
    max_gap: int = 5,
    ema_alpha: float = 0.3,
    min_confidence: float = 0.2,
) -> PoseSequence:
    """Fill short gaps and apply EMA smoothing to joint positions."""

    joint_names = _collect_joint_names(seq.frames)
    filled_frames = _fill_gaps(seq.frames, joint_names, max_gap, min_confidence)
    smoothed_frames = _ema_smooth(filled_frames, joint_names, ema_alpha)
    return PoseSequence(meta=seq.meta, frames=smoothed_frames)


def _collect_joint_names(frames: List[PoseFrame]) -> List[str]:
    names = set()
    for frame in frames:
        names.update(frame.joints.keys())
    return sorted(names)


def _fill_gaps(
    frames: List[PoseFrame],
    joint_names: List[str],
    max_gap: int,
    min_confidence: float,
) -> List[PoseFrame]:
    filled = [PoseFrame(f=frame.f, t=frame.t, joints=dict(frame.joints)) for frame in frames]
    for name in joint_names:
        indices = [i for i, frame in enumerate(frames) if name in frame.joints and frame.joints[name].c >= min_confidence]
        if not indices:
            continue
        for idx, frame in enumerate(frames):
            if name in frame.joints and frame.joints[name].c >= min_confidence:
                continue
            prev_idx = max([i for i in indices if i < idx], default=None)
            next_idx = min([i for i in indices if i > idx], default=None)
            if prev_idx is None:
                continue
            if next_idx is None:
                filled[idx].joints[name] = frames[prev_idx].joints[name]
                continue
            gap = next_idx - prev_idx - 1
            if gap <= max_gap:
                t = (idx - prev_idx) / (gap + 1)
                filled[idx].joints[name] = _lerp_joint(frames[prev_idx].joints[name], frames[next_idx].joints[name], t)
            else:
                filled[idx].joints[name] = frames[prev_idx].joints[name]
    return filled


def _ema_smooth(frames: List[PoseFrame], joint_names: List[str], alpha: float) -> List[PoseFrame]:
    smoothed: Dict[str, np.ndarray] = {}
    out_frames: List[PoseFrame] = []
    for frame in frames:
        joints = dict(frame.joints)
        for name in joint_names:
            if name not in joints:
                continue
            pos = np.array([joints[name].x, joints[name].y, joints[name].z], dtype=np.float64)
            if name not in smoothed:
                smoothed[name] = pos
            else:
                smoothed[name] = alpha * pos + (1.0 - alpha) * smoothed[name]
            joints[name] = PoseJoint(
                x=float(smoothed[name][0]),
                y=float(smoothed[name][1]),
                z=float(smoothed[name][2]),
                c=joints[name].c,
            )
        out_frames.append(PoseFrame(f=frame.f, t=frame.t, joints=joints))
    return out_frames


def _lerp_joint(a: PoseJoint, b: PoseJoint, t: float) -> PoseJoint:
    return PoseJoint(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t,
        c=a.c + (b.c - a.c) * t,
    )
