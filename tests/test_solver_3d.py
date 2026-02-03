from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.retarget.bone_resolver import BoneResolver
from mmd_trace.retarget.solver_3d import solve


def _quat_angle_deg(quat: np.ndarray) -> float:
    w = float(np.clip(quat[3], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def test_solver_3d_leg_straight() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    points[23] = np.array([-1.0, 0.0, 0.0])
    points[24] = np.array([1.0, 0.0, 0.0])
    points[11] = np.array([-1.0, 1.0, 0.0])
    points[12] = np.array([1.0, 1.0, 0.0])
    points[25] = np.array([-1.0, -1.0, 0.0])
    points[27] = np.array([-1.0, -2.0, 0.0])
    points[26] = np.array([1.0, -1.0, 0.0])
    points[28] = np.array([1.0, -2.0, 0.0])

    vis = np.ones(33, dtype=np.float64)
    bone_local, _, _, _ = solve(model, points, vis, vis_th=0.1)
    assert len(bone_local) > 0

    for key in ("leg_L", "knee_L", "leg_R", "knee_R"):
        bone_name = resolver.bone(key)
        assert bone_name is not None
        quat = bone_local[bone_name]
        assert _quat_angle_deg(quat) < 10.0
