"""Tests for pose skeleton image rendering."""

from __future__ import annotations

import numpy as np

from mmd_trace.viz.pose_skeleton_image import render_pose_skeleton_image


def _make_points(count: int = 33) -> np.ndarray:
    values = np.linspace(-0.5, 0.5, count)
    return np.stack([values, values[::-1], np.zeros(count)], axis=1)


def test_render_pose_skeleton_image_draws_lines() -> None:
    points = _make_points()
    vis = np.ones(points.shape[0], dtype=np.float64)
    image = render_pose_skeleton_image(
        points=points,
        vis=vis,
        width=64,
        height=64,
        margin=0.1,
        thickness=1,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        vis_th=0.2,
    )
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert int(image.sum()) > 0


def test_render_pose_skeleton_image_skips_low_visibility() -> None:
    points = _make_points()
    vis = np.zeros(points.shape[0], dtype=np.float64)
    image = render_pose_skeleton_image(
        points=points,
        vis=vis,
        width=64,
        height=64,
        margin=0.1,
        thickness=1,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        vis_th=0.2,
    )
    assert int(image.sum()) == 0
