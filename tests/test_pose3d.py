from __future__ import annotations

from od2blender.pose3d import get_pose3d_skeleton


def test_pose3d_skeleton_shape() -> None:
    skeleton = get_pose3d_skeleton()
    landmarks = skeleton["landmarks"]
    bones = skeleton["bones"]

    assert len(landmarks) == 33
    assert landmarks[23] == "left_hip"
    assert landmarks[24] == "right_hip"
    assert bones
