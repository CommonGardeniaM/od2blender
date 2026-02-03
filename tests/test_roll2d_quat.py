from __future__ import annotations

import numpy as np

from mmd_trace.retarget.quat import apply
from mmd_trace.retarget.solver_2d_roll import quat_from_two_vectors_roll2d


def test_quat_from_two_vectors_roll2d() -> None:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tgt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    quat = quat_from_two_vectors_roll2d(ref, tgt)
    rotated = apply(quat, ref)
    assert np.allclose(rotated[:2], np.array([0.0, 1.0]), atol=1e-6)
    assert abs(quat[0]) < 1e-6
    assert abs(quat[1]) < 1e-6
