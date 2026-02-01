import numpy as np

from mmd_trace.retarget.kinematics import quat_from_two_vectors, quat_normalize


def test_quat_identity():
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    q = quat_from_two_vectors(v0, v1)
    q = quat_normalize(q)
    assert np.allclose(q, np.array([0.0, 0.0, 0.0, 1.0]))
