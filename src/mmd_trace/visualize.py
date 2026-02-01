"""Pose visualization overlay tool."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .io_pose import PoseJoint, PoseSequence, read_pose_json


LOG = logging.getLogger(__name__)


@dataclass
class VisualizeConfig:
    show_skeleton: bool = True
    show_joints: bool = True
    show_labels: bool = False
    show_confidence: bool = True
    show_frame_info: bool = True
    line_thickness: int = 2
    joint_radius: int = 5
    low_confidence_threshold: float = 0.5


# Skeleton connections (parent -> child)
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("pelvis", "spine"),
    ("spine", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("chest", "l_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("chest", "r_shoulder"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("pelvis", "r_hip"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]

# Color scheme: left=red, right=blue, center=green
COLOR_LEFT = (0, 0, 255)  # BGR: red
COLOR_RIGHT = (255, 0, 0)  # BGR: blue
COLOR_CENTER = (0, 255, 0)  # BGR: green
COLOR_LOW_CONFIDENCE = (0, 165, 255)  # BGR: orange
COLOR_TEXT = (255, 255, 255)  # BGR: white
COLOR_TEXT_BG = (0, 0, 0)  # BGR: black


def _get_joint_color(joint_name: str, confidence: float, threshold: float) -> Tuple[int, int, int]:
    """Get color for joint based on side and confidence."""
    if confidence < threshold:
        return COLOR_LOW_CONFIDENCE
    if joint_name.startswith("l_"):
        return COLOR_LEFT
    if joint_name.startswith("r_"):
        return COLOR_RIGHT
    return COLOR_CENTER


def _project_3d_to_2d(
    joint: PoseJoint,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """Convert normalized 2D coordinates to pixel coordinates.
    
    Uses u, v fields (normalized 0-1) from MediaPipe detection.
    Falls back to 3D projection if u=v=0 (legacy data).
    """
    # Use normalized 2D coordinates if available
    if joint.u != 0.0 or joint.v != 0.0:
        x = int(joint.u * frame_width)
        y = int(joint.v * frame_height)
        return x, y
    
    # Fallback: project 3D coordinates (for legacy data)
    scale = 800.0
    offset_x = 0.5
    offset_y = 0.7
    x = int((joint.x * scale) + (frame_width * offset_x))
    y = int((-joint.y * scale) + (frame_height * offset_y))
    return x, y


def _draw_skeleton(
    frame: np.ndarray,
    joints: Dict[str, PoseJoint],
    config: VisualizeConfig,
) -> None:
    """Draw skeleton lines between connected joints."""
    height, width = frame.shape[:2]
    
    for parent_name, child_name in SKELETON_CONNECTIONS:
        if parent_name not in joints or child_name not in joints:
            continue
        
        parent = joints[parent_name]
        child = joints[child_name]
        
        # Skip low confidence connections
        min_conf = min(parent.c, child.c)
        if min_conf < 0.1:  # Very low confidence, skip
            continue
        
        p1 = _project_3d_to_2d(parent, width, height)
        p2 = _project_3d_to_2d(child, width, height)
        
        # Determine color based on child joint side
        color = _get_joint_color(child_name, child.c, config.low_confidence_threshold)
        
        # Adjust alpha based on confidence
        alpha = min(1.0, max(0.3, min(parent.c, child.c)))
        thickness = max(1, int(config.line_thickness * alpha))
        
        cv2.line(frame, p1, p2, color, thickness)


def _draw_joints(
    frame: np.ndarray,
    joints: Dict[str, PoseJoint],
    config: VisualizeConfig,
) -> None:
    """Draw joint markers."""
    height, width = frame.shape[:2]
    
    for name, joint in joints.items():
        if joint.c < 0.1:  # Skip very low confidence
            continue
        
        x, y = _project_3d_to_2d(joint, width, height)
        color = _get_joint_color(name, joint.c, config.low_confidence_threshold)
        
        # Radius based on confidence
        alpha = min(1.0, max(0.5, joint.c))
        radius = max(2, int(config.joint_radius * alpha))
        
        cv2.circle(frame, (x, y), radius, color, -1)  # Filled circle
        cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1)  # Black outline


def _draw_labels(
    frame: np.ndarray,
    joints: Dict[str, PoseJoint],
    config: VisualizeConfig,
) -> None:
    """Draw joint name labels."""
    height, width = frame.shape[:2]
    
    for name, joint in joints.items():
        if joint.c < config.low_confidence_threshold:
            continue
        
        x, y = _project_3d_to_2d(joint, width, height)
        label = name.replace("_", " ")
        
        # Draw text with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(
            frame,
            (x + 5, y - text_size[1] - 5),
            (x + 5 + text_size[0], y),
            COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x + 5, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_TEXT,
            1,
        )


def _draw_frame_info(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    joints: Dict[str, PoseJoint],
    config: VisualizeConfig,
) -> None:
    """Draw frame number, timestamp, and average confidence."""
    if not joints:
        return
    
    avg_conf = sum(j.c for j in joints.values()) / len(joints)
    info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s | Avg Conf: {avg_conf:.2f}"
    
    # Draw at top-left with background
    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (5, 5), (10 + text_size[0], 25), COLOR_TEXT_BG, -1)
    cv2.putText(
        frame,
        info_text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_TEXT,
        2,
    )


def create_overlay_frame(
    original_frame: np.ndarray,
    joints: Dict[str, PoseJoint],
    frame_idx: int,
    timestamp: float,
    config: VisualizeConfig,
) -> np.ndarray:
    """Create a single overlay frame."""
    # Create a copy for overlay
    overlay = original_frame.copy()
    
    # Create semi-transparent overlay layer
    overlay_layer = np.zeros_like(overlay)
    
    if config.show_skeleton:
        _draw_skeleton(overlay_layer, joints, config)
    
    if config.show_joints:
        _draw_joints(overlay_layer, joints, config)
    
    if config.show_labels:
        _draw_labels(overlay_layer, joints, config)
    
    # Blend overlay with original (50% transparency for overlay)
    alpha = 0.6
    result = cv2.addWeighted(overlay, 1.0, overlay_layer, alpha, 0)
    
    if config.show_frame_info:
        _draw_frame_info(result, frame_idx, timestamp, joints, config)
    
    return result


def visualize_pose_on_image(
    image_path: str,
    pose_sequence: PoseSequence,
    output_path: str,
    config: Optional[VisualizeConfig] = None,
) -> None:
    """Generate overlay image from image and pose sequence.
    
    Args:
        image_path: Path to input image
        pose_sequence: PoseSequence with joint data
        output_path: Path for output image
        config: Visualization configuration
    """
    config = config or VisualizeConfig()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        LOG.info("Created output directory: %s", output_dir)
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    # Get first pose frame
    pose_frame = pose_sequence.frames[0] if pose_sequence.frames else None
    
    if pose_frame:
        overlay = create_overlay_frame(
            image,
            pose_frame.joints,
            0,
            0.0,
            config,
        )
        cv2.imwrite(output_path, overlay)
        LOG.info("Wrote overlay image to %s", output_path)
    else:
        LOG.warning("No pose data found to visualize")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmd-trace visualize")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--pose", required=True, help="Pose JSON path")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--no-skeleton", action="store_true", help="Hide skeleton lines")
    parser.add_argument("--no-joints", action="store_true", help="Hide joint markers")
    parser.add_argument("--labels", action="store_true", help="Show joint name labels")
    parser.add_argument("--no-info", action="store_true", help="Hide frame info")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--joint-radius", type=int, default=5)
    parser.add_argument("--low-conf-threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # Load pose data
    LOG.info("Loading pose data from %s", args.pose)
    pose_sequence = read_pose_json(args.pose)
    
    # Create config
    viz_config = VisualizeConfig(
        show_skeleton=not args.no_skeleton,
        show_joints=not args.no_joints,
        show_labels=args.labels,
        show_frame_info=not args.no_info,
        line_thickness=args.line_thickness,
        joint_radius=args.joint_radius,
        low_confidence_threshold=args.low_conf_threshold,
    )
    
    # Generate overlay
    visualize_pose_on_image(
        args.image,
        pose_sequence,
        args.out,
        viz_config,
    )


if __name__ == "__main__":
    main()
