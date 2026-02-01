"""Pose JSON schema and I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass
class PoseJoint:
    x: float
    y: float
    z: float
    c: float
    u: float = 0.0  # Normalized 2D x (0-1) for visualization
    v: float = 0.0  # Normalized 2D y (0-1) for visualization


@dataclass
class PoseFrame:
    f: int
    t: Optional[float]
    joints: Dict[str, PoseJoint]


@dataclass
class PoseMeta:
    fps: int
    frame_count: int
    units: str
    coord: str
    axis_map: Dict[str, object]


@dataclass
class PoseSequence:
    meta: PoseMeta
    frames: List[PoseFrame]


def pose_to_dict(seq: PoseSequence) -> Dict[str, object]:
    return {
        "meta": {
            "fps": int(seq.meta.fps),
            "frame_count": int(seq.meta.frame_count),
            "units": seq.meta.units,
            "coord": seq.meta.coord,
            "axis_map": seq.meta.axis_map,
        },
        "frames": [
            {
                "f": int(frame.f),
                "t": frame.t,
                "joints": {
                    name: {"x": joint.x, "y": joint.y, "z": joint.z, "c": joint.c, "u": joint.u, "v": joint.v}
                    for name, joint in frame.joints.items()
                },
            }
            for frame in seq.frames
        ],
    }


def pose_from_dict(data: Dict[str, object]) -> PoseSequence:
    meta = data.get("meta", {})
    frames_data = data.get("frames", [])
    pose_meta = PoseMeta(
        fps=int(meta.get("fps", 30)),
        frame_count=int(meta.get("frame_count", len(frames_data))),
        units=str(meta.get("units", "m")),
        coord=str(meta.get("coord", "provider_world")),
        axis_map=dict(meta.get("axis_map", {})),
    )
    frames: List[PoseFrame] = []
    for frame in frames_data:
        joints: Dict[str, PoseJoint] = {}
        for name, joint in frame.get("joints", {}).items():
            joints[str(name)] = PoseJoint(
                x=float(joint.get("x", 0.0)),
                y=float(joint.get("y", 0.0)),
                z=float(joint.get("z", 0.0)),
                c=float(joint.get("c", 0.0)),
                u=float(joint.get("u", 0.0)),
                v=float(joint.get("v", 0.0)),
            )
        frames.append(PoseFrame(f=int(frame.get("f", 0)), t=frame.get("t"), joints=joints))
    return PoseSequence(meta=pose_meta, frames=frames)


def write_pose_json(seq: PoseSequence, path: str) -> None:
    data = pose_to_dict(seq)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_pose_json(path: str) -> PoseSequence:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Pose JSON root must be a mapping")
    return pose_from_dict(data)
