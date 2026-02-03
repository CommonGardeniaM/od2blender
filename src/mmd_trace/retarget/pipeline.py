"""Retargeting pipeline entry point."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mmd_trace.io.pmx import PmxModel
from mmd_trace.pose_provider import PoseBundleNp

from .coords import AxisTransform, center_points, drop_z
from .solver_2d_roll import solve as solve_2d_roll
from .solver_3d import solve as solve_3d


class SolveMode(str, Enum):
    """Solver mode."""

    ROLL_2D = "2d_roll"
    WORLD_3D = "3d"


@dataclass(frozen=True)
class SolveResult:
    bone_quat_local: dict[str, np.ndarray]
    bone_quat_world: dict[str, np.ndarray]
    bone_trans: dict[str, tuple[float, float, float]]
    centers: dict[str, np.ndarray]
    pose_points: np.ndarray
    pose_vis: np.ndarray | None
    key_to_name: dict[str, str | None]


def build_rotations(
    bundle: PoseBundleNp,
    model: PmxModel,
    mode: SolveMode,
    axis: AxisTransform,
    vis_th: float = 0.2,
) -> SolveResult:
    points = axis.apply(bundle.world_points)
    points_centered, _ = center_points(points, bundle.world_vis, vis_th)

    if mode == SolveMode.ROLL_2D:
        points_use = drop_z(points_centered)
        bone_local, bone_world, centers, key_to_name = solve_2d_roll(
            model, points_use, bundle.world_vis, vis_th
        )
    elif mode == SolveMode.WORLD_3D:
        points_use = points_centered
        bone_local, bone_world, centers, key_to_name = solve_3d(
            model, points_use, bundle.world_vis, vis_th
        )
    else:
        raise ValueError(f"Unknown solver mode: {mode}")

    bone_trans: dict[str, tuple[float, float, float]] = {}
    return SolveResult(
        bone_quat_local=bone_local,
        bone_quat_world=bone_world,
        bone_trans=bone_trans,
        centers=centers,
        pose_points=points_use,
        pose_vis=bundle.world_vis,
        key_to_name=key_to_name,
    )
