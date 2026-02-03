"""2D overlay for pose landmarks."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from mmd_trace.retarget.indices import LANDMARK_NAMES, POSE_CONNECTIONS

LOG = logging.getLogger(__name__)


LANDMARK_RADIUS = 4
LANDMARK_COLOR = (0, 255, 0)
LANDMARK_COLOR_LOW_CONF = (0, 0, 255)
CONNECTION_COLOR = (255, 255, 255)
CONNECTION_THICKNESS = 2
TEXT_COLOR = (255, 255, 0)
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1


def draw_landmarks(
    image: np.ndarray,
    image_points: np.ndarray,
    image_vis: np.ndarray | None,
    output_path: str | None = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    vis_th: float = 0.5,
) -> np.ndarray:
    vis_image = image.copy()
    height, width = vis_image.shape[:2]

    if image_points is None or len(image_points) == 0:
        cv2.putText(
            vis_image,
            "No pose detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        if output_path:
            cv2.imwrite(output_path, vis_image)
        return vis_image

    for start_index, end_index in POSE_CONNECTIONS:
        start = image_points[start_index]
        end = image_points[end_index]
        x_start = int(start[0] * width)
        y_start = int(start[1] * height)
        x_end = int(end[0] * width)
        y_end = int(end[1] * height)
        cv2.line(
            vis_image,
            (x_start, y_start),
            (x_end, y_end),
            CONNECTION_COLOR,
            CONNECTION_THICKNESS,
        )

    for point_index, point in enumerate(image_points):
        x_pixel = int(point[0] * width)
        y_pixel = int(point[1] * height)
        visibility = float(image_vis[point_index]) if image_vis is not None else 1.0
        color = LANDMARK_COLOR if visibility >= vis_th else LANDMARK_COLOR_LOW_CONF
        cv2.circle(vis_image, (x_pixel, y_pixel), LANDMARK_RADIUS, color, -1)

        if show_labels or show_confidence:
            parts = []
            if show_labels:
                name = (
                    LANDMARK_NAMES[point_index]
                    if point_index < len(LANDMARK_NAMES)
                    else str(point_index)
                )
                parts.append(f"{point_index}:{name}")
            if show_confidence:
                parts.append(f"v={visibility:.2f}")
            label = " ".join(parts)
            cv2.putText(
                vis_image,
                label,
                (x_pixel + 6, y_pixel - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
            )

    if output_path:
        cv2.imwrite(output_path, vis_image)
        LOG.info("Saved visualization to: %s", output_path)

    return vis_image
