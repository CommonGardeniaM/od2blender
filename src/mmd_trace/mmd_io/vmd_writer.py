"""VMD writing using pypmxvmd."""

from __future__ import annotations

from typing import Dict, List
import contextlib
import io

import pypmxvmd
from pypmxvmd.common.models.vmd import VmdMotion, VmdHeader, VmdBoneFrame


def write_vmd_motion(
    path: str,
    model_name: str,
    bone_frames: List[VmdBoneFrame],
) -> None:
    motion = VmdMotion()
    motion.header = VmdHeader(version=2, model_name=model_name)
    motion.bone_frames = bone_frames
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pypmxvmd.save_vmd(motion, path)


def make_bone_frame(
    bone_name: str,
    frame_number: int,
    position: List[float],
    rotation_euler_deg: List[float],
) -> VmdBoneFrame:
    return VmdBoneFrame(
        bone_name=bone_name,
        frame_number=frame_number,
        position=position,
        rotation=rotation_euler_deg,
        interpolation=[20, 20, 107, 107] * 4,
        physics_disabled=False,
    )
