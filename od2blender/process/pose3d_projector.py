from __future__ import annotations

from od2blender.pose3d import LEFT_HIP_INDEX, RIGHT_HIP_INDEX
from od2blender.process.base import FrameProcessor, ProcessorContext


def _midpoint(coords: list, left_index: int, right_index: int) -> tuple[float, float, float]:
    if left_index >= len(coords) or right_index >= len(coords):
        return (0.0, 0.0, 0.0)
    left = coords[left_index]
    right = coords[right_index]
    left_x = float(left[0]) if len(left) > 0 else 0.0
    left_y = float(left[1]) if len(left) > 1 else 0.0
    left_z = float(left[2]) if len(left) > 2 else 0.0
    right_x = float(right[0]) if len(right) > 0 else 0.0
    right_y = float(right[1]) if len(right) > 1 else 0.0
    right_z = float(right[2]) if len(right) > 2 else 0.0
    return ((left_x + right_x) * 0.5, (left_y + right_y) * 0.5, (left_z + right_z) * 0.5)


class Pose3DProjectorProcessor(FrameProcessor):
    id = "pose3d_projector"

    def process(self, frames: list[dict], context: ProcessorContext) -> list[dict]:
        plane_width_value = context.settings.get("plane_width", 2.0)
        z_depth_value = context.settings.get("z_depth", 0.01)
        scale_value = context.settings.get("pose_scale", 1.0)
        pose3d_source = context.settings.get("pose3d_source")
        keep_raw = bool(context.settings.get("pose3d_keep_raw")) or pose3d_source == "raw"
        plane_width = float(plane_width_value) if plane_width_value is not None else 2.0
        z_depth = float(z_depth_value) if z_depth_value is not None else 0.01
        scale = float(scale_value) if scale_value is not None else 1.0

        plane_height = (
            plane_width * (context.video_meta.height / context.video_meta.width)
            if context.video_meta.width
            else plane_width
        )

        for frame in frames:
            poses_raw = frame.get("poses3d_raw") or []
            poses_out: list[dict] = []
            for pose in poses_raw:
                world = pose.get("world") or []
                image = pose.get("image") or []
                if not world or not image:
                    continue
                root_world = _midpoint(world, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
                root_image = _midpoint(image, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
                offset_x = (root_image[0] - 0.5) * plane_width
                offset_y = (0.5 - root_image[1]) * plane_height
                landmarks: list[list[float]] = []
                for coords in world:
                    x = float(coords[0]) if len(coords) > 0 else 0.0
                    y = float(coords[1]) if len(coords) > 1 else 0.0
                    z = float(coords[2]) if len(coords) > 2 else 0.0
                    visibility = float(coords[3]) if len(coords) > 3 else 1.0
                    proj_x = (x - root_world[0]) * scale + offset_x
                    proj_y = -(y - root_world[1]) * scale + offset_y
                    proj_z = (z - root_world[2]) * scale + z_depth
                    landmarks.append([proj_x, proj_y, proj_z, visibility])
                poses_out.append({"id": int(pose.get("id", 0)), "landmarks": landmarks})
            frame["poses3d"] = poses_out
            if not keep_raw:
                frame.pop("poses3d_raw", None)
        return frames

    def describe(self, context: ProcessorContext) -> dict | None:
        plane_width_value = context.settings.get("plane_width", 2.0)
        z_depth_value = context.settings.get("z_depth", 0.01)
        scale_value = context.settings.get("pose_scale", 1.0)
        pose3d_source = context.settings.get("pose3d_source")
        keep_raw = bool(context.settings.get("pose3d_keep_raw")) or pose3d_source == "raw"
        return {
            "id": self.id,
            "params": {
                "plane_width": float(plane_width_value)
                if plane_width_value is not None
                else 2.0,
                "z_depth": float(z_depth_value) if z_depth_value is not None else 0.01,
                "scale": float(scale_value) if scale_value is not None else 1.0,
                "keep_raw": keep_raw,
            },
        }
