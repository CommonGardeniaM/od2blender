from __future__ import annotations

from typing import Callable
from pathlib import Path

from od2blender.infer.base import InferenceBackend, InferenceResult
from od2blender.pose3d import Pose3DConfig, run_pose3d
from od2blender.video_meta import VideoMeta


class Pose3DBackend(InferenceBackend):
    id = "pose3d"
    mode = "pose3d"
    default_processors = ["pose3d_projector"]

    def infer(
        self,
        video_path: Path,
        video_meta: VideoMeta,
        config: dict,
        log_warn: Callable[[str], None] | None = None,
    ) -> InferenceResult:
        pose_config = Pose3DConfig(
            model_path=config.get("pose_model"),
        )
        frames, detector_meta, skeleton = run_pose3d(
            video_path,
            video_meta,
            config=pose_config,
            log_warn=log_warn,
        )
        return InferenceResult(
            frames=frames,
            detector_meta=detector_meta,
            mode=self.mode,
            skeleton=skeleton,
            units={"distance": "meter"},
            axis="blender_xy_z",
        )
