"""Retargeting pipeline entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mmd_trace.io.pmx import PmxModel
from mmd_trace.pose_provider import PoseBundleNp

from .coords import AxisTransform, center_points, drop_z
from .indices import POSE_IDX
from .solver_2d_roll import solve as solve_2d_roll
from .solver_3d import solve as solve_3d
from .solver_3d_hybrid import solve as solve_3d_hybrid
from .solver_mikapo import solve as solve_mikapo

LOG = logging.getLogger(__name__)

EPS = 1e-9


class SolveMode(str, Enum):
    """Solver mode."""

    ROLL_2D = "2d_roll"
    WORLD_3D = "3d"
    HYBRID_3D = "3d_hybrid"
    MIKAPO = "mikapo"


class LegMode(str, Enum):
    """Leg output mode."""

    IK = "ik"
    FK = "fk"


class LegIkAxes(str, Enum):
    """Leg IK offset axes."""

    X = "x"
    XZ = "xz"


@dataclass(frozen=True)
class SolveResult:
    bone_quat_local: dict[str, np.ndarray]
    bone_quat_world: dict[str, np.ndarray]
    bone_trans: dict[str, tuple[float, float, float]]
    centers: dict[str, np.ndarray]
    pose_points: np.ndarray
    pose_vis: np.ndarray | None
    key_to_name: dict[str, str | None]


def _pose_point(
    points: np.ndarray,
    vis: np.ndarray | None,
    name: str,
    vis_th: float,
) -> np.ndarray | None:
    idx = POSE_IDX[name]
    if vis is not None and float(vis[idx]) < vis_th:
        return None
    return points[idx]


def _pair_distance(
    points: np.ndarray,
    vis: np.ndarray | None,
    name_a: str,
    name_b: str,
    vis_th: float,
) -> float | None:
    point_a = _pose_point(points, vis, name_a, vis_th)
    point_b = _pose_point(points, vis, name_b, vis_th)
    if point_a is None or point_b is None:
        return None
    length = float(np.linalg.norm(point_b - point_a))
    if length < EPS:
        return None
    return length


def _strip_leg_fk(
    bone_local: dict[str, np.ndarray],
    bone_world: dict[str, np.ndarray],
    key_to_name: dict[str, str | None],
) -> None:
    for key in ("leg_L", "knee_L", "leg_R", "knee_R"):
        bone_name = key_to_name.get(key)
        if bone_name is None:
            continue
        bone_local.pop(bone_name, None)
        bone_world.pop(bone_name, None)


def _apply_leg_ik_trans(
    bone_trans: dict[str, tuple[float, float, float]],
    points: np.ndarray,
    vis: np.ndarray | None,
    vis_th: float,
    model: PmxModel,
    key_to_name: dict[str, str | None],
    leg_ik_axes: LegIkAxes,
    leg_ik_scale: float,
) -> None:
    leg_ik_l = key_to_name.get("leg_ik_L")
    leg_ik_r = key_to_name.get("leg_ik_R")
    leg_l = key_to_name.get("leg_L")
    leg_r = key_to_name.get("leg_R")

    if leg_ik_l is None or leg_ik_r is None or leg_l is None or leg_r is None:
        LOG.warning("Leg IK bones not found; skipping IK translation output.")
        return

    leg_l_bone = model.get_bone(leg_l)
    leg_r_bone = model.get_bone(leg_r)
    if leg_l_bone is None or leg_r_bone is None:
        LOG.warning("Leg FK bones not found in model; skipping IK translation output.")
        return

    pose_width = _pair_distance(points, vis, "left_hip", "right_hip", vis_th)
    if pose_width is None:
        pose_width = _pair_distance(points, vis, "left_shoulder", "right_shoulder", vis_th)
    if pose_width is None:
        LOG.warning("Pose hip/shoulder width unavailable; skipping IK translation output.")
        return

    model_width = float(np.linalg.norm(leg_l_bone.position - leg_r_bone.position))
    if model_width < EPS:
        LOG.warning("Model leg width too small; skipping IK translation output.")
        return

    scale = (model_width / pose_width) * float(leg_ik_scale)

    ankle_l = _pose_point(points, vis, "left_ankle", vis_th)
    ankle_r = _pose_point(points, vis, "right_ankle", vis_th)

    if leg_ik_axes == LegIkAxes.XZ:
        if ankle_l is None or ankle_r is None:
            LOG.warning("Pose ankle points unavailable; skipping IK translation output.")
            return
        mean_ankle_z = 0.5 * (float(ankle_l[2]) + float(ankle_r[2]))
    else:
        mean_ankle_z = 0.0

    if ankle_l is not None:
        offset_x = scale * float(ankle_l[0])
        offset_z = (
            scale * (float(ankle_l[2]) - mean_ankle_z)
            if leg_ik_axes == LegIkAxes.XZ
            else 0.0
        )
        bone_trans[leg_ik_l] = (offset_x, 0.0, offset_z)

    if ankle_r is not None:
        offset_x = scale * float(ankle_r[0])
        offset_z = (
            scale * (float(ankle_r[2]) - mean_ankle_z)
            if leg_ik_axes == LegIkAxes.XZ
            else 0.0
        )
        bone_trans[leg_ik_r] = (offset_x, 0.0, offset_z)


def build_rotations(
    bundle: PoseBundleNp,
    model: PmxModel,
    mode: SolveMode,
    axis: AxisTransform,
    vis_th: float = 0.2,
    leg_mode: LegMode = LegMode.IK,
    leg_ik_axes: LegIkAxes = LegIkAxes.XZ,
    leg_ik_scale: float = 1.0,
) -> SolveResult:
    points = axis.apply(bundle.world_points)
    if mode == SolveMode.MIKAPO:
        points_centered = points
        points_use = points
        bone_local, bone_world, centers, key_to_name = solve_mikapo(
            model, points_use, None, vis_th
        )
    else:
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
        elif mode == SolveMode.HYBRID_3D:
            points_use = points_centered
            bone_local, bone_world, centers, key_to_name = solve_3d_hybrid(
                model, points_use, bundle.world_vis, vis_th
            )
        else:
            raise ValueError(f"Unknown solver mode: {mode}")

    bone_trans: dict[str, tuple[float, float, float]] = {}
    if leg_mode == LegMode.IK:
        _apply_leg_ik_trans(
            bone_trans=bone_trans,
            points=points_centered,
            vis=bundle.world_vis,
            vis_th=vis_th,
            model=model,
            key_to_name=key_to_name,
            leg_ik_axes=leg_ik_axes,
            leg_ik_scale=leg_ik_scale,
        )
        _strip_leg_fk(bone_local, bone_world, key_to_name)
    return SolveResult(
        bone_quat_local=bone_local,
        bone_quat_world=bone_world,
        bone_trans=bone_trans,
        centers=centers,
        pose_points=points_use,
        pose_vis=bundle.world_vis,
        key_to_name=key_to_name,
    )
