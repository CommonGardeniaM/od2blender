from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.retarget.solver_2d_roll import solve


def test_solver_2d_roll_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    pmx_path = root / "Millial_forMMD_v1.0.0.pmx"
    model = load_pmx(str(pmx_path))

    points = np.zeros((33, 3), dtype=np.float64)
    points[23] = np.array([-1.0, 0.0, 0.0])
    points[24] = np.array([1.0, 0.0, 0.0])
    points[11] = np.array([-1.0, 1.0, 0.0])
    points[12] = np.array([1.0, 1.0, 0.0])
    points[13] = np.array([-2.0, 1.0, 0.0])
    points[15] = np.array([-3.0, 1.0, 0.0])
    points[14] = np.array([2.0, 1.0, 0.0])
    points[16] = np.array([3.0, 1.0, 0.0])
    points[25] = np.array([-1.0, -1.0, 0.0])
    points[27] = np.array([-1.0, -2.0, 0.0])
    points[26] = np.array([1.0, -1.0, 0.0])
    points[28] = np.array([1.0, -2.0, 0.0])

    vis = np.ones(33, dtype=np.float64)
    bone_local, _, _, _ = solve(model, points, vis, vis_th=0.1)
    assert len(bone_local) > 0
    for quat in bone_local.values():
        assert quat.shape == (4,)
