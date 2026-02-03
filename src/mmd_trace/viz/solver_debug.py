"""Pose solver debug visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.pose_provider import create_pose_provider
from mmd_trace.retarget.coords import AxisTransform
from mmd_trace.retarget.indices import POSE_CONNECTIONS, POSE_IDX
from mmd_trace.retarget.pipeline import LegIkAxes, LegMode, SolveMode, build_rotations
from mmd_trace.retarget.quat import apply
from mmd_trace.viz.landmarks_overlay import draw_landmarks

LOG = logging.getLogger(__name__)

AXIS_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 255, 0),
    "z": (255, 0, 0),
}


def _pose_point(
    points: np.ndarray, vis: np.ndarray | None, name: str, vis_th: float
) -> np.ndarray | None:
    index = POSE_IDX[name]
    if vis is not None and float(vis[index]) < vis_th:
        return None
    return points[index]


def _origin_from_key(
    key: str,
    points: np.ndarray,
    vis: np.ndarray | None,
    vis_th: float,
) -> np.ndarray | None:
    if key == "lower_body":
        lhip = _pose_point(points, vis, "left_hip", vis_th)
        rhip = _pose_point(points, vis, "right_hip", vis_th)
        if lhip is not None and rhip is not None:
            return 0.5 * (lhip + rhip)
    if key == "upper_body":
        lsho = _pose_point(points, vis, "left_shoulder", vis_th)
        rsho = _pose_point(points, vis, "right_shoulder", vis_th)
        if lsho is not None and rsho is not None:
            return 0.5 * (lsho + rsho)
    if key == "upper_body2":
        lear = _pose_point(points, vis, "left_ear", vis_th)
        rear = _pose_point(points, vis, "right_ear", vis_th)
        if lear is not None and rear is not None:
            return 0.5 * (lear + rear)
    if key == "neck":
        lsho = _pose_point(points, vis, "left_shoulder", vis_th)
        rsho = _pose_point(points, vis, "right_shoulder", vis_th)
        if lsho is not None and rsho is not None:
            return 0.5 * (lsho + rsho)
    if key == "head":
        lear = _pose_point(points, vis, "left_ear", vis_th)
        rear = _pose_point(points, vis, "right_ear", vis_th)
        if lear is not None and rear is not None:
            return 0.5 * (lear + rear)
    if key == "shoulder_L":
        return _pose_point(points, vis, "left_shoulder", vis_th)
    if key == "arm_L":
        return _pose_point(points, vis, "left_shoulder", vis_th)
    if key == "elbow_L":
        return _pose_point(points, vis, "left_elbow", vis_th)
    if key == "wrist_L":
        return _pose_point(points, vis, "left_wrist", vis_th)
    if key == "shoulder_R":
        return _pose_point(points, vis, "right_shoulder", vis_th)
    if key == "arm_R":
        return _pose_point(points, vis, "right_shoulder", vis_th)
    if key == "elbow_R":
        return _pose_point(points, vis, "right_elbow", vis_th)
    if key == "wrist_R":
        return _pose_point(points, vis, "right_wrist", vis_th)
    if key == "leg_L":
        return _pose_point(points, vis, "left_hip", vis_th)
    if key == "knee_L":
        return _pose_point(points, vis, "left_knee", vis_th)
    if key == "ankle_L":
        return _pose_point(points, vis, "left_ankle", vis_th)
    if key == "leg_R":
        return _pose_point(points, vis, "right_hip", vis_th)
    if key == "knee_R":
        return _pose_point(points, vis, "right_knee", vis_th)
    if key == "ankle_R":
        return _pose_point(points, vis, "right_ankle", vis_th)
    return None


def _world_to_image(
    point: np.ndarray, points: np.ndarray, size: tuple[int, int]
) -> tuple[int, int]:
    height, width = size
    min_x, max_x = float(points[:, 0].min()), float(points[:, 0].max())
    min_y, max_y = float(points[:, 1].min()), float(points[:, 1].max())
    delta_x = max(max_x - min_x, 1e-6)
    delta_y = max(max_y - min_y, 1e-6)
    x_norm = (point[0] - min_x) / delta_x
    y_norm = (point[1] - min_y) / delta_y
    x_pixel = int(x_norm * width)
    y_pixel = int((1.0 - y_norm) * height)
    return x_pixel, y_pixel


def _draw_axes_2d(
    image: np.ndarray,
    origin_px: tuple[int, int],
    quat_world: np.ndarray,
    axis_length: int,
) -> None:
    origin_x, origin_y = origin_px
    axes = {
        "x": apply(quat_world, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        "y": apply(quat_world, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
        "z": apply(quat_world, np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    }
    for key, vec in axes.items():
        vec_2d = np.array([vec[0], -vec[1]], dtype=np.float64)
        length = np.linalg.norm(vec_2d)
        if length < 1e-6:
            continue
        vec_2d = vec_2d / length
        delta_x = int(vec_2d[0] * axis_length)
        delta_y = int(vec_2d[1] * axis_length)
        color = AXIS_COLORS[key]
        cv2.line(
            image,
            (origin_x, origin_y),
            (origin_x + delta_x, origin_y + delta_y),
            color,
            2,
        )


def _draw_connections_3d(axis_3d, points: np.ndarray) -> None:
    for start_index, end_index in POSE_CONNECTIONS:
        start = points[start_index]
        end = points[end_index]
        axis_3d.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="gray",
            linewidth=1,
        )


def _set_equal_axis_3d(axis_3d, points: np.ndarray) -> None:
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_min + x_max) * 0.5
    mid_y = (y_min + y_max) * 0.5
    mid_z = (z_min + z_max) * 0.5
    axis_3d.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    axis_3d.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    axis_3d.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def generate_debug_visualization(
    image_path: str,
    pmx_path: str,
    out_dir: str,
    axis: AxisTransform,
    vis_th: float,
    det_conf: float,
    solver: SolveMode,
    leg_mode: LegMode,
    leg_ik_axes: LegIkAxes,
    leg_ik_scale: float,
    mode: str,
    project: str,
    axis_scale: float,
    dpi: int,
) -> dict[str, str]:
    model = load_pmx(pmx_path)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    provider = create_pose_provider(det_conf=det_conf)
    bundle = provider.detect(image)
    result = build_rotations(
        bundle,
        model,
        solver,
        axis,
        vis_th,
        leg_mode=leg_mode,
        leg_ik_axes=leg_ik_axes,
        leg_ik_scale=leg_ik_scale,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_files: dict[str, str] = {}

    if mode in ("2d", "both"):
        output_path_2d = str(Path(out_dir) / "debug_2d.png")
        overlay = draw_landmarks(
            image,
            bundle.image_points,
            bundle.image_vis,
            output_path=None,
            show_labels=False,
            show_confidence=False,
        )
        height, width = overlay.shape[:2]
        axis_length = int(min(height, width) * 0.08 * axis_scale)

        for key, bone_name in result.key_to_name.items():
            if bone_name is None:
                continue
            quat_world = result.bone_quat_world.get(bone_name)
            if quat_world is None:
                continue
            if project == "2d":
                origin = _origin_from_key(
                    key, bundle.image_points, bundle.image_vis, vis_th
                )
                if origin is None:
                    continue
                origin_x = int(origin[0] * width)
                origin_y = int(origin[1] * height)
            else:
                origin_world = _origin_from_key(
                    key, result.pose_points, result.pose_vis, vis_th
                )
                if origin_world is None:
                    continue
                origin_x, origin_y = _world_to_image(
                    origin_world, result.pose_points, (height, width)
                )
            _draw_axes_2d(overlay, (origin_x, origin_y), quat_world, axis_length)

        cv2.imwrite(output_path_2d, overlay)
        output_files["2d"] = output_path_2d

    if mode in ("3d", "both"):
        output_path_3d = str(Path(out_dir) / "debug_3d.png")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for 3D visualization") from exc

        fig = plt.figure(figsize=(12, 10), dpi=dpi)
        axis_3d = fig.add_subplot(111, projection="3d")
        _draw_connections_3d(axis_3d, result.pose_points)

        for key, bone_name in result.key_to_name.items():
            if bone_name is None:
                continue
            quat_world = result.bone_quat_world.get(bone_name)
            if quat_world is None:
                continue
            origin = _origin_from_key(key, result.pose_points, result.pose_vis, vis_th)
            if origin is None:
                continue
            axes = [
                apply(quat_world, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                apply(quat_world, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                apply(quat_world, np.array([0.0, 0.0, 1.0], dtype=np.float64)),
            ]
            colors = ["r", "g", "b"]
            for vec, color in zip(axes, colors, strict=True):
                axis_3d.plot(
                    [origin[0], origin[0] + vec[0] * axis_scale],
                    [origin[1], origin[1] + vec[1] * axis_scale],
                    [origin[2], origin[2] + vec[2] * axis_scale],
                    color=color,
                )

        axis_3d.set_xlabel("X")
        axis_3d.set_ylabel("Y")
        axis_3d.set_zlabel("Z")
        _set_equal_axis_3d(axis_3d, result.pose_points)
        fig.tight_layout()
        fig.savefig(output_path_3d, dpi=dpi)
        output_files["3d"] = output_path_3d
        plt.close(fig)

    return output_files
