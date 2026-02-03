from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.pose_provider.base import PoseBundleNp
from mmd_trace.retarget.bone_resolver import BoneResolver
from mmd_trace.retarget.coords import AxisTransform
from mmd_trace.retarget.pipeline import LegIkAxes, LegMode, SolveMode, build_rotations


def test_leg_ik_trans_x_axes() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    points[23] = np.array([0.5, 0.0, 0.0])  # left_hip
    points[24] = np.array([-0.5, 0.0, 0.0])  # right_hip
    points[27] = np.array([0.6, -1.0, 0.2])  # left_ankle
    points[28] = np.array([-0.6, -1.0, -0.2])  # right_ankle

    vis = np.ones(33, dtype=np.float64)
    bundle = PoseBundleNp(
        world_points=points,
        world_vis=vis,
        image_points=points,
        image_vis=vis,
    )

    result = build_rotations(
        bundle=bundle,
        model=model,
        mode=SolveMode.WORLD_3D,
        axis=AxisTransform(),
        vis_th=0.1,
        leg_mode=LegMode.IK,
        leg_ik_axes=LegIkAxes.X,
        leg_ik_scale=1.0,
    )

    leg_ik_l = resolver.bone("leg_ik_L")
    leg_ik_r = resolver.bone("leg_ik_R")
    assert leg_ik_l is not None
    assert leg_ik_r is not None

    assert leg_ik_l in result.bone_trans
    assert leg_ik_r in result.bone_trans

    _, ty_l, tz_l = result.bone_trans[leg_ik_l]
    _, ty_r, tz_r = result.bone_trans[leg_ik_r]
    assert abs(ty_l) < 1e-6
    assert abs(ty_r) < 1e-6
    assert abs(tz_l) < 1e-6
    assert abs(tz_r) < 1e-6

    for key in ("leg_L", "knee_L", "leg_R", "knee_R"):
        bone_name = resolver.bone(key)
        assert bone_name is not None
        assert bone_name not in result.bone_quat_local
