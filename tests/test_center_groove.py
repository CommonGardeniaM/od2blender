import numpy as np

from mmd_trace.io_pose import PoseFrame, PoseJoint, PoseMeta, PoseSequence
from mmd_trace.retarget.center_groove import compute_center_groove


def test_center_groove_pattern_a():
    meta = PoseMeta(fps=30, frame_count=2, units="m", coord="provider_world", axis_map={})
    frames = [
        PoseFrame(f=0, t=0.0, joints={"pelvis": PoseJoint(0.0, 0.0, 0.0, 1.0)}),
        PoseFrame(f=1, t=1.0 / 30.0, joints={"pelvis": PoseJoint(1.0, 2.0, 3.0, 1.0)}),
    ]
    seq = PoseSequence(meta=meta, frames=frames)
    center, groove = compute_center_groove(seq, axis_map={"swap": ["x", "y", "z"], "invert": {"x": False, "y": False, "z": False}})
    assert np.allclose(center[1], np.array([0.0, 2.0, 0.0]))
    assert np.allclose(groove[1], np.array([1.0, 0.0, 3.0]))
