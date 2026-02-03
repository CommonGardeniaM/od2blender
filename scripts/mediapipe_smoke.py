from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from mmd_trace.pose_provider import create_pose_provider  # noqa: E402
from mmd_trace.viz.landmarks_overlay import draw_landmarks  # noqa: E402

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    input_image = ROOT / "waking.png"
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    pose_json_path = output_dir / "pose_33points.json"
    debug_image_path = output_dir / "debug_overlay.png"

    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    LOG.info("MediaPipe 33-point Pose Estimation")
    provider = create_pose_provider(det_conf=0.5)

    image = cv2.imread(str(input_image))
    if image is None:
        raise RuntimeError(f"Failed to read image: {input_image}")

    bundle = provider.detect(image)

    pose_data = {
        "frame_count": 1,
        "world_points": bundle.world_points.tolist(),
        "world_vis": bundle.world_vis.tolist(),
        "image_points": bundle.image_points.tolist(),
        "image_vis": bundle.image_vis.tolist(),
    }

    with open(pose_json_path, "w", encoding="utf-8") as file_handle:
        json.dump(pose_data, file_handle, indent=2, ensure_ascii=False)
    LOG.info("Saved pose JSON: %s", pose_json_path)

    draw_landmarks(
        image,
        bundle.image_points,
        bundle.image_vis,
        output_path=str(debug_image_path),
        show_labels=True,
        show_confidence=True,
    )
    LOG.info("Saved debug overlay: %s", debug_image_path)


if __name__ == "__main__":
    main()
