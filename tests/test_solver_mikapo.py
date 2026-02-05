from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.retarget.bone_resolver import BoneResolver
from mmd_trace.retarget.solver_mikapo import solve


def _quat_angle_deg(quat: np.ndarray) -> float:
    w = float(np.clip(quat[3], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def test_solver_mikapo_outputs_identity_for_missing() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    vis = np.zeros(33, dtype=np.float64)

    bone_local, bone_world, _, _ = solve(model, points, vis, vis_th=0.1)

    for key in (
        "upper_body",
        "neck",
        "head",
        "lower_body",
        "leg_L",
        "leg_R",
        "knee_L",
        "knee_R",
        "ankle_L",
        "ankle_R",
        "arm_L",
        "arm_R",
        "elbow_L",
        "elbow_R",
    ):
        bone_name = resolver.bone(key)
        if bone_name is None:
            continue
        assert bone_name in bone_local
        assert bone_name in bone_world
        assert np.allclose(bone_local[bone_name], np.array([0.0, 0.0, 0.0, 1.0]))
        assert np.allclose(bone_world[bone_name], np.array([0.0, 0.0, 0.0, 1.0]))


def test_solver_mikapo_upper_body_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    points[11] = np.array([1.0, 1.0, 0.0])
    points[12] = np.array([-1.0, 1.0, 0.0])

    vis = np.zeros(33, dtype=np.float64)
    bone_local, _, _, _ = solve(model, points, vis, vis_th=0.1)

    upper_body = resolver.bone("upper_body")
    assert upper_body is not None
    assert _quat_angle_deg(bone_local[upper_body]) < 1.0


def test_solver_mikapo_vis_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))
    resolver = BoneResolver(model)

    points = np.zeros((33, 3), dtype=np.float64)
    points[11] = np.array([1.0, 1.0, 0.0])
    points[12] = np.array([0.0, 1.0, 0.0])

    vis = np.zeros(33, dtype=np.float64)
    bone_local, _, _, _ = solve(model, points, vis, vis_th=0.1)

    upper_body = resolver.bone("upper_body")
    assert upper_body is not None
    assert _quat_angle_deg(bone_local[upper_body]) > 1.0
