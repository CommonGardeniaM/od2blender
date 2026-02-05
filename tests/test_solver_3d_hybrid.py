from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.retarget.bone_resolver import BoneResolver
from mmd_trace.retarget.quat import IDENTITY_QUAT
from mmd_trace.retarget.solver_3d_hybrid import solve


def _quat_angle_deg(quat: np.ndarray) -> float:
    w = float(np.clip(quat[3], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def test_solver_3d_hybrid_leg_straight() -> None:
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


def test_solver_3d_hybrid_wrist_ankle() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    points[23] = np.array([-0.5, 0.0, 0.0])
    points[24] = np.array([0.5, 0.0, 0.0])
    points[11] = np.array([-0.5, 1.0, 0.0])
    points[12] = np.array([0.5, 1.0, 0.0])

    points[13] = np.array([-0.5, 2.0, 0.0])
    points[15] = np.array([-0.5, 3.0, 0.0])
    points[19] = np.array([-0.3, 3.2, 0.2])
    points[17] = np.array([-0.7, 3.2, -0.2])

    points[25] = np.array([-0.5, -1.0, 0.5])
    points[27] = np.array([-0.5, -2.0, 1.0])
    points[29] = np.array([-0.5, -2.0, 0.6])
    points[31] = np.array([-0.5, -2.0, 1.4])

    vis = np.ones(33, dtype=np.float64)
    bone_local, _, _, _ = solve(model, points, vis, vis_th=0.1)

    wrist_name = resolver.bone("wrist_L")
    ankle_name = resolver.bone("ankle_L")
    assert wrist_name is not None
    assert ankle_name is not None
    assert wrist_name in bone_local
    assert ankle_name in bone_local

    assert not np.allclose(bone_local[wrist_name], IDENTITY_QUAT)
    assert not np.allclose(bone_local[ankle_name], IDENTITY_QUAT)
