"""Pose solver debug visualization with centers and axes."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mmd_trace.pose_processor import HAND_IDX, POSE_IDX, PoseSolver, map_points, quat_rotate

LOG = logging.getLogger(__name__)


@dataclass
class CenterPoint:
    """Center point data."""

    name: str
    position: np.ndarray
    color: Tuple[int, int, int]
    space: str


@dataclass
class BoneAxis:
    """Bone axes data."""

    label: str
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    space: str


class PoseDebugVisualizer:
    """Visualize PoseSolver outputs with centers and axes."""

    CENTER_COLORS = {
        "hip_center": (255, 0, 0),
        "shoulder_center": (0, 0, 255),
        "ear_center": (0, 255, 0),
        "eye_center": (0, 255, 255),
        "nose": (255, 0, 255),
    }

    AXIS_COLORS = {
        "x": (0, 0, 255),
        "y": (0, 255, 0),
        "z": (255, 0, 0),
    }

    def __init__(
        self,
        axis_scale: float = 1.0,
        show_centers: bool = True,
        show_axes: bool = True,
        show_fingers: bool = True,
        project_mode: str = "2d",
        dpi: int = 150,
    ) -> None:
        self.axis_scale = axis_scale
        self.show_centers = show_centers
        self.show_axes = show_axes
        self.show_fingers = show_fingers
        self.project_mode = project_mode
        self.dpi = dpi

    def visualize_2d(
        self,
        image: np.ndarray,
        pose_solver: PoseSolver,
        image_points: Optional[np.ndarray] = None,
        image_vis: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw pose debug overlay on a 2D image."""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        axis_length = int(min(h, w) * 0.08 * self.axis_scale)

        centers: List[CenterPoint] = []
        if self.show_centers:
            centers = self._extract_centers(pose_solver, image_points, image_vis)
            for center in centers:
                self._draw_center(vis_image, center, w, h)

        bone_axes: List[BoneAxis] = []
        if self.show_axes:
            bone_axes = self._extract_bone_axes(pose_solver, image_points, image_vis)
            for bone_axis in bone_axes:
                self._draw_bone_axis_2d(vis_image, bone_axis, axis_length, w, h)

        self._draw_info_panel(vis_image, centers, bone_axes)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            LOG.info("Saved 2D debug visualization to: %s", output_path)

        return vis_image

    def visualize_3d(
        self,
        pose_solver: PoseSolver,
        output_path: Optional[str] = None,
        show_connections: bool = True,
    ) -> Any:
        """Render a 3D matplotlib visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            LOG.error("matplotlib is required for 3D visualization")
            raise exc

        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        centers: List[CenterPoint] = []
        if self.show_centers:
            centers = self._extract_centers(pose_solver, None, None)
            for center in centers:
                ax.scatter(
                    center.position[0],
                    center.position[1],
                    center.position[2],
                    c=[self._bgr_to_rgb(center.color)],
                    s=100,
                    marker="o",
                    label=center.name,
                )

        bone_axes: List[BoneAxis] = []
        if self.show_axes:
            bone_axes = self._extract_bone_axes(pose_solver, None, None)
            axis_length = self._axis_length_3d(pose_solver)
            for bone_axis in bone_axes:
                self._draw_bone_axis_3d(ax, bone_axis, axis_length)

        if show_connections and pose_solver.pose is not None:
            self._draw_skeleton_3d(ax, pose_solver)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Pose Debug Visualization (3D)")
        if centers:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        self._set_equal_axis_3d(ax, pose_solver)

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            LOG.info("Saved 3D debug visualization to: %s", output_path)

        return fig

    def visualize_both(
        self,
        image: np.ndarray,
        pose_solver: PoseSolver,
        output_dir: str,
        base_name: str = "debug",
    ) -> Tuple[np.ndarray, Any]:
        """Generate both 2D and 3D outputs."""
        output_path_2d = str(Path(output_dir) / f"{base_name}_2d.png")
        output_path_3d = str(Path(output_dir) / f"{base_name}_3d.png")

        vis_2d = self.visualize_2d(image, pose_solver, output_path=output_path_2d)
        fig_3d = self.visualize_3d(pose_solver, output_path_3d)

        return vis_2d, fig_3d

    def _extract_centers(
        self,
        pose_solver: PoseSolver,
        image_points: Optional[np.ndarray],
        image_vis: Optional[np.ndarray],
    ) -> List[CenterPoint]:
        if self.project_mode == "2d" and image_points is not None:
            return self._extract_centers_from_points(image_points, image_vis)

        centers: List[CenterPoint] = []
        center_data = [
            ("hip_center", pose_solver.hip_center),
            ("shoulder_center", pose_solver.shoulder_center),
            ("ear_center", pose_solver.ear_center),
            ("eye_center", pose_solver.eye_center),
            ("nose", pose_solver.nose),
        ]

        for name, position in center_data:
            if position is None:
                continue
            color = self.CENTER_COLORS.get(name, (128, 128, 128))
            centers.append(CenterPoint(name, position, color, "world"))

        return centers

    def _extract_centers_from_points(
        self, image_points: np.ndarray, image_vis: Optional[np.ndarray]
    ) -> List[CenterPoint]:
        centers: List[CenterPoint] = []
        center_indices = {
            "hip_center": ("left_hip", "right_hip"),
            "shoulder_center": ("left_shoulder", "right_shoulder"),
            "ear_center": ("left_ear", "right_ear"),
            "eye_center": ("left_eye", "right_eye"),
        }

        for name, (left_key, right_key) in center_indices.items():
            left = self._pose_point_from_points(image_points, image_vis, left_key)
            right = self._pose_point_from_points(image_points, image_vis, right_key)
            if left is None or right is None:
                continue
            position = 0.5 * (left + right)
            color = self.CENTER_COLORS.get(name, (128, 128, 128))
            centers.append(CenterPoint(name, position, color, "image"))

        nose = self._pose_point_from_points(image_points, image_vis, "nose")
        if nose is not None:
            centers.append(CenterPoint("nose", nose, self.CENTER_COLORS["nose"], "image"))

        return centers

    def _extract_bone_axes(
        self,
        pose_solver: PoseSolver,
        image_points: Optional[np.ndarray],
        image_vis: Optional[np.ndarray],
    ) -> List[BoneAxis]:
        label_map, origin_map, finger_bones = self._build_label_and_origin_map(
            pose_solver, image_points, image_vis
        )
        bone_axes: List[BoneAxis] = []

        for bone_name, world_q in pose_solver.bone_world.items():
            if not self.show_fingers and bone_name in finger_bones:
                continue
            bone = pose_solver.resolver.get_bone(bone_name)
            if bone is None:
                continue

            origin = origin_map.get(bone_name, bone.position.copy())
            label = label_map.get(bone_name, bone_name)

            x_axis = quat_rotate(world_q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            y_axis = quat_rotate(world_q, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            z_axis = quat_rotate(world_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))

            space = "image" if self.project_mode == "2d" and image_points is not None else "world"
            bone_axes.append(
                BoneAxis(
                    label=label,
                    origin=origin,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    z_axis=z_axis,
                    space=space,
                )
            )

        return bone_axes

    def _build_label_and_origin_map(
        self,
        pose_solver: PoseSolver,
        image_points: Optional[np.ndarray],
        image_vis: Optional[np.ndarray],
    ) -> Tuple[Dict[str, str], Dict[str, np.ndarray], List[str]]:
        label_map: Dict[str, str] = {}
        origin_map: Dict[str, np.ndarray] = {}
        finger_bones: List[str] = []

        for key, bone_name in pose_solver.resolver.bones.items():
            if bone_name is None:
                continue
            label_map[bone_name] = key
            origin = self._origin_for_bone_key(pose_solver, image_points, image_vis, key)
            if origin is not None:
                origin_map[bone_name] = origin

        finger_joint_map = {
            "thumb": ["thumb_mcp", "thumb_ip", "thumb_tip"],
            "index": ["index_mcp", "index_pip", "index_dip"],
            "middle": ["middle_mcp", "middle_pip", "middle_dip"],
            "ring": ["ring_mcp", "ring_pip", "ring_dip"],
            "pinky": ["pinky_mcp", "pinky_pip", "pinky_dip"],
        }

        for finger_key, chain in pose_solver.resolver.fingers.items():
            if "_" not in finger_key:
                continue
            side, finger = finger_key.split("_", 1)
            joints = finger_joint_map.get(finger)
            if joints is None:
                continue
            for idx, bone_name in enumerate(chain):
                if bone_name is None:
                    continue
                finger_bones.append(bone_name)
                if not self.show_fingers:
                    continue
                label_map[bone_name] = f"{finger}_{side}{idx + 1}"
                joint_name = joints[min(idx, len(joints) - 1)]
                origin = self._hand_point(pose_solver, side, joint_name)
                if origin is not None:
                    origin_map[bone_name] = origin

        return label_map, origin_map, finger_bones

    def _origin_for_bone_key(
        self,
        pose_solver: PoseSolver,
        image_points: Optional[np.ndarray],
        image_vis: Optional[np.ndarray],
        key: str,
    ) -> Optional[np.ndarray]:
        if self.project_mode == "2d" and image_points is not None:
            return self._origin_for_bone_key_from_points(image_points, image_vis, key)
        if key in {"center", "groove", "lower_body"}:
            return pose_solver.hip_center
        if key == "upper_body":
            return pose_solver.shoulder_center
        if key == "upper_body2":
            if pose_solver.ear_center is not None:
                return pose_solver.ear_center
            return pose_solver.shoulder_center
        if key == "neck":
            return pose_solver.shoulder_center
        if key == "head":
            if pose_solver.ear_center is not None:
                return pose_solver.ear_center
            return pose_solver.eye_center
        if key in {"shoulder_L", "arm_L"}:
            return self._pose_point(pose_solver, "left_shoulder")
        if key == "elbow_L":
            return self._pose_point(pose_solver, "left_elbow")
        if key in {"wrist_L", "wrist_twist_L"}:
            return self._pose_point(pose_solver, "left_wrist")
        if key in {"shoulder_R", "arm_R"}:
            return self._pose_point(pose_solver, "right_shoulder")
        if key == "elbow_R":
            return self._pose_point(pose_solver, "right_elbow")
        if key in {"wrist_R", "wrist_twist_R"}:
            return self._pose_point(pose_solver, "right_wrist")
        if key == "leg_L":
            return self._pose_point(pose_solver, "left_hip")
        if key == "knee_L":
            return self._pose_point(pose_solver, "left_knee")
        if key == "ankle_L":
            return self._pose_point(pose_solver, "left_ankle")
        if key == "toe_L":
            return self._pose_point(pose_solver, "left_foot_index")
        if key == "leg_R":
            return self._pose_point(pose_solver, "right_hip")
        if key == "knee_R":
            return self._pose_point(pose_solver, "right_knee")
        if key == "ankle_R":
            return self._pose_point(pose_solver, "right_ankle")
        if key == "toe_R":
            return self._pose_point(pose_solver, "right_foot_index")
        return None

    def _origin_for_bone_key_from_points(
        self,
        image_points: np.ndarray,
        image_vis: Optional[np.ndarray],
        key: str,
    ) -> Optional[np.ndarray]:
        if key in {"center", "groove", "lower_body"}:
            return self._center_from_points(image_points, image_vis, "left_hip", "right_hip")
        if key == "upper_body":
            return self._center_from_points(
                image_points, image_vis, "left_shoulder", "right_shoulder"
            )
        if key == "upper_body2":
            center = self._center_from_points(image_points, image_vis, "left_ear", "right_ear")
            if center is not None:
                return center
            return self._center_from_points(
                image_points, image_vis, "left_shoulder", "right_shoulder"
            )
        if key == "neck":
            return self._center_from_points(
                image_points, image_vis, "left_shoulder", "right_shoulder"
            )
        if key == "head":
            center = self._center_from_points(image_points, image_vis, "left_ear", "right_ear")
            if center is not None:
                return center
            return self._center_from_points(image_points, image_vis, "left_eye", "right_eye")
        if key in {"shoulder_L", "arm_L"}:
            return self._pose_point_from_points(image_points, image_vis, "left_shoulder")
        if key == "elbow_L":
            return self._pose_point_from_points(image_points, image_vis, "left_elbow")
        if key in {"wrist_L", "wrist_twist_L"}:
            return self._pose_point_from_points(image_points, image_vis, "left_wrist")
        if key in {"shoulder_R", "arm_R"}:
            return self._pose_point_from_points(image_points, image_vis, "right_shoulder")
        if key == "elbow_R":
            return self._pose_point_from_points(image_points, image_vis, "right_elbow")
        if key in {"wrist_R", "wrist_twist_R"}:
            return self._pose_point_from_points(image_points, image_vis, "right_wrist")
        if key == "leg_L":
            return self._pose_point_from_points(image_points, image_vis, "left_hip")
        if key == "knee_L":
            return self._pose_point_from_points(image_points, image_vis, "left_knee")
        if key == "ankle_L":
            return self._pose_point_from_points(image_points, image_vis, "left_ankle")
        if key == "toe_L":
            return self._pose_point_from_points(image_points, image_vis, "left_foot_index")
        if key == "leg_R":
            return self._pose_point_from_points(image_points, image_vis, "right_hip")
        if key == "knee_R":
            return self._pose_point_from_points(image_points, image_vis, "right_knee")
        if key == "ankle_R":
            return self._pose_point_from_points(image_points, image_vis, "right_ankle")
        if key == "toe_R":
            return self._pose_point_from_points(image_points, image_vis, "right_foot_index")
        return None

    @staticmethod
    def _pose_point(pose_solver: PoseSolver, name: str) -> Optional[np.ndarray]:
        if pose_solver.pose is None:
            return None
        idx = POSE_IDX[name]
        if pose_solver.vis is not None and pose_solver.vis[idx] < pose_solver.vis_th:
            return None
        return pose_solver.pose[idx]

    @staticmethod
    def _pose_point_from_points(
        image_points: np.ndarray,
        image_vis: Optional[np.ndarray],
        name: str,
        vis_th: float = 0.2,
    ) -> Optional[np.ndarray]:
        idx = POSE_IDX[name]
        if image_vis is not None and image_vis[idx] < vis_th:
            return None
        return image_points[idx]

    def _center_from_points(
        self,
        image_points: np.ndarray,
        image_vis: Optional[np.ndarray],
        left_key: str,
        right_key: str,
    ) -> Optional[np.ndarray]:
        left = self._pose_point_from_points(image_points, image_vis, left_key)
        right = self._pose_point_from_points(image_points, image_vis, right_key)
        if left is None or right is None:
            return None
        return 0.5 * (left + right)

    @staticmethod
    def _hand_point(pose_solver: PoseSolver, side: str, name: str) -> Optional[np.ndarray]:
        data = pose_solver.left_hand if side == "L" else pose_solver.right_hand
        if data is None:
            return None
        return data[HAND_IDX[name]]

    def _draw_center(
        self,
        image: np.ndarray,
        center: CenterPoint,
        img_width: int,
        img_height: int,
    ) -> None:
        if center.space == "image":
            x = int(center.position[0] * img_width)
            y = int(center.position[1] * img_height)
        else:
            x = int(center.position[0] * img_width * 0.5 + img_width * 0.5)
            y = int(-center.position[1] * img_height * 0.5 + img_height * 0.5)

        cv2.circle(image, (x, y), 8, center.color, -1)
        cv2.circle(image, (x, y), 10, (255, 255, 255), 2)

        label = center.name
        if center.space == "image":
            coord_text = f"({center.position[0]:.3f}, {center.position[1]:.3f})"
        else:
            coord_text = (
                f"({center.position[0]:.2f}, {center.position[1]:.2f}, {center.position[2]:.2f})"
            )

        cv2.putText(
            image,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            center.color,
            2,
        )
        cv2.putText(
            image,
            coord_text,
            (x + 12, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    def _draw_bone_axis_2d(
        self,
        image: np.ndarray,
        bone_axis: BoneAxis,
        axis_length: int,
        img_width: int,
        img_height: int,
    ) -> None:
        if bone_axis.space == "image":
            x0 = int(bone_axis.origin[0] * img_width)
            y0 = int(bone_axis.origin[1] * img_height)
        else:
            x0 = int(bone_axis.origin[0] * img_width * 0.5 + img_width * 0.5)
            y0 = int(-bone_axis.origin[1] * img_height * 0.5 + img_height * 0.5)

        axes = [
            (bone_axis.x_axis, self.AXIS_COLORS["x"]),
            (bone_axis.y_axis, self.AXIS_COLORS["y"]),
            (bone_axis.z_axis, self.AXIS_COLORS["z"]),
        ]

        for axis, color in axes:
            scale = max(0.3, min(1.0, 1.0 / (1.0 + abs(axis[2]) * 0.5)))
            x1 = int(x0 + axis[0] * axis_length * scale)
            y1 = int(y0 - axis[1] * axis_length * scale)

            cv2.arrowedLine(
                image,
                (x0, y0),
                (x1, y1),
                color,
                2,
                tipLength=0.3,
            )

        cv2.putText(
            image,
            bone_axis.label,
            (x0 + 5, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    def _draw_bone_axis_3d(self, ax: Any, bone_axis: BoneAxis, axis_length: float) -> None:
        origin = bone_axis.origin

        axes_data = [
            (bone_axis.x_axis, "red"),
            (bone_axis.y_axis, "green"),
            (bone_axis.z_axis, "blue"),
        ]

        for axis, color in axes_data:
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                axis[0],
                axis[1],
                axis[2],
                length=axis_length,
                normalize=True,
                color=color,
                alpha=0.7,
                arrow_length_ratio=0.3,
            )

    def _draw_skeleton_3d(self, ax: Any, pose_solver: PoseSolver) -> None:
        if pose_solver.pose is None:
            return

        connections = [
            (11, 12),
            (11, 23),
            (12, 24),
            (23, 24),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (9, 10),
            (0, 9),
            (0, 10),
        ]

        for start_idx, end_idx in connections:
            if start_idx >= len(pose_solver.pose) or end_idx >= len(pose_solver.pose):
                continue
            start_pt = pose_solver.pose[start_idx]
            end_pt = pose_solver.pose[end_idx]

            ax.plot3D(
                [start_pt[0], end_pt[0]],
                [start_pt[1], end_pt[1]],
                [start_pt[2], end_pt[2]],
                "gray",
                alpha=0.5,
                linewidth=2,
            )

    def _draw_info_panel(
        self,
        image: np.ndarray,
        centers: List[CenterPoint],
        bone_axes: List[BoneAxis],
    ) -> None:
        h, w = image.shape[:2]
        panel_height = 30 + len(centers) * 25 + 40 + len(bone_axes) * 20 + 20
        panel_bottom = min(panel_height, h - 10)

        cv2.rectangle(image, (w - 350, 10), (w - 10, panel_bottom), (0, 0, 0), -1)
        cv2.rectangle(image, (w - 350, 10), (w - 10, panel_bottom), (255, 255, 255), 1)

        y_pos = 35
        cv2.putText(
            image,
            "=== Center Points ===",
            (w - 340, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_pos += 25

        for center in centers:
            if center.space == "image":
                text = f"{center.name}: ({center.position[0]:.3f}, {center.position[1]:.3f})"
            else:
                text = (
                    f"{center.name}: ({center.position[0]:.2f}, "
                    f"{center.position[1]:.2f}, {center.position[2]:.2f})"
                )
            cv2.putText(
                image,
                text,
                (w - 340, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                center.color,
                1,
            )
            y_pos += 25

        if not bone_axes:
            return

        y_pos += 15
        cv2.putText(
            image,
            "=== Bone Axes ===",
            (w - 340, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_pos += 25

        for bone_axis in bone_axes[:10]:
            cv2.putText(
                image,
                bone_axis.label,
                (w - 340, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
            y_pos += 20

        if len(bone_axes) > 10:
            cv2.putText(
                image,
                f"... and {len(bone_axes) - 10} more bones",
                (w - 340, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )

    def _set_equal_axis_3d(self, ax: Any, pose_solver: PoseSolver) -> None:
        if pose_solver.pose is None:
            return

        points = pose_solver.pose
        x_range = [points[:, 0].min(), points[:, 0].max()]
        y_range = [points[:, 1].min(), points[:, 1].max()]
        z_range = [points[:, 2].min(), points[:, 2].max()]

        max_range = max(
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            z_range[1] - z_range[0],
        ) * 0.6

        x_mid = (x_range[0] + x_range[1]) * 0.5
        y_mid = (y_range[0] + y_range[1]) * 0.5
        z_mid = (z_range[0] + z_range[1]) * 0.5

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

    def _axis_length_3d(self, pose_solver: PoseSolver) -> float:
        if pose_solver.pose is None:
            return 0.1 * self.axis_scale

        points = pose_solver.pose
        ranges = [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
        max_range = max(ranges)
        if max_range <= 0:
            return 0.1 * self.axis_scale
        return max_range * 0.2 * self.axis_scale

    @staticmethod
    def _bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return (color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)


def generate_debug_visualization(
    image_path: str,
    pmx_path: str,
    out_dir: str,
    task_model: Optional[str],
    model_type: str,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    axis_matrix: Optional[np.ndarray],
    vis_th: float,
    det_conf: float,
    mode: str,
    project: str,
    show_fingers: bool,
    axis_scale: float,
    dpi: int,
) -> Dict[str, str]:
    """Generate debug visualization from an image and PMX model."""
    from mmd_trace.io.pmx import load_pmx
    from mmd_trace.pose_provider.mediapipe_provider import MediaPipePoseProvider

    model = load_pmx(pmx_path)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    provider = MediaPipePoseProvider(
        model_asset_path=task_model,
        model_type=model_type,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=det_conf,
        min_tracking_confidence=det_conf,
    )

    result = provider.detect_pose_bundle_np(image)
    if result is None:
        raise RuntimeError("PoseLandmarker detected no poses.")
    points, vis, image_points, image_vis = result

    points_mapped = map_points(points, axis_x, axis_y, axis_z, axis_matrix)

    solver = PoseSolver(model, vis_th=vis_th)
    solver.solve(points_mapped, vis)

    visualizer = PoseDebugVisualizer(
        axis_scale=axis_scale,
        show_centers=True,
        show_axes=True,
        show_fingers=show_fingers,
        project_mode=project,
        dpi=dpi,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, str] = {}

    if mode in ("2d", "both"):
        output_path_2d = str(Path(out_dir) / "debug_2d.png")
        visualizer.visualize_2d(
            image,
            solver,
            image_points=image_points,
            image_vis=image_vis,
            output_path=output_path_2d,
        )
        output_files["2d"] = output_path_2d
        print(f"Generated 2D visualization: {output_path_2d}")

    if mode in ("3d", "both"):
        output_path_3d = str(Path(out_dir) / "debug_3d.png")
        visualizer.visualize_3d(solver, output_path_3d)
        output_files["3d"] = output_path_3d
        print(f"Generated 3D visualization: {output_path_3d}")

    return output_files
