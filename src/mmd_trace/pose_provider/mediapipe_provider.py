"""MediaPipe-based pose provider (MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast
import importlib
import logging
from pathlib import Path

import numpy as np

cv2 = cast(Any, importlib.import_module("cv2"))

from ..io_pose import PoseFrame, PoseJoint, PoseMeta, PoseSequence
from ..io_video import iter_video_frames
from ..visualize import VisualizeConfig, create_overlay_frame
from .base import PoseProvider

LOG = logging.getLogger(__name__)


@dataclass
class MediaPipeSettings:
    min_confidence: float = 0.2
    model_complexity: int = 1
    model_path: Optional[str] = None


class MediaPipePoseProvider(PoseProvider):
    """MediaPipe Pose inference wrapper."""

    def __init__(self, settings: Optional[MediaPipeSettings] = None) -> None:
        self.settings = settings or MediaPipeSettings()

    def infer(
        self,
        video_path: str,
        fps_override: Optional[int] = None,
        debug_overlay_path: Optional[str] = None,
    ) -> PoseSequence:
        info, frames = iter_video_frames(video_path, fps_override=fps_override)
        pose_module = cast(Any, importlib.import_module("mediapipe.tasks.python.vision"))
        base_module = cast(Any, importlib.import_module("mediapipe.tasks.python.core.base_options"))
        image_module = cast(Any, importlib.import_module("mediapipe.tasks.python.vision.core.image"))
        running_mode = pose_module.RunningMode

        model_path = self._resolve_model_path()
        base_options = base_module.BaseOptions(model_asset_path=model_path)
        options = pose_module.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.settings.min_confidence,
            min_pose_presence_confidence=self.settings.min_confidence,
            min_tracking_confidence=self.settings.min_confidence,
            output_segmentation_masks=False,
        )
        pose = pose_module.PoseLandmarker.create_from_options(options)

        writer = None
        if debug_overlay_path:
            fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
            if fourcc_func is None:
                raise RuntimeError("cv2.VideoWriter_fourcc is unavailable")
            fourcc = fourcc_func(*"mp4v")
            writer = cv2.VideoWriter(debug_overlay_path, fourcc, info.fps, (info.width, info.height))

        pose_frames = []
        for frame_idx, t, frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = image_module.Image(
                image_format=image_module.ImageFormat.SRGB,
                data=rgb,
            )
            timestamp_ms = int(t * 1000)
            result = cast(Any, pose.detect_for_video(mp_image, timestamp_ms))
            joints = self._extract_joints(result, pose_module)
            pose_frames.append(PoseFrame(f=frame_idx, t=t, joints=joints))
            if writer is not None:
                # Use visualize logic for rich overlay
                # Create default config for debug overlay
                viz_config = VisualizeConfig(
                    show_skeleton=True,
                    show_joints=True,
                    show_labels=False,  # Skip labels for debug to be faster? Or keep same. Let's keep defaults.
                    show_frame_info=True,
                )
                overlay = create_overlay_frame(
                    frame,
                    joints,
                    frame_idx,
                    t,
                    viz_config,
                )
                writer.write(overlay)
            if frame_idx % 100 == 0:
                LOG.info("Processed frame %d", frame_idx)

        if writer is not None:
            writer.release()
        pose.close()

        meta = PoseMeta(
            fps=info.fps,
            frame_count=len(pose_frames),
            units="m",
            coord="provider_world",
            axis_map={"swap": ["x", "y", "z"], "invert": {"x": False, "y": False, "z": False}},
        )
        return PoseSequence(meta=meta, frames=pose_frames)

    def _extract_joints(self, result: Any, pose_module: Any) -> Dict[str, PoseJoint]:
        pose_landmarks = getattr(result, "pose_landmarks", None)
        if not pose_landmarks:
            return {}

        landmarks_list = pose_landmarks[0]
        world_landmarks = getattr(result, "pose_world_landmarks", None)
        world_list = world_landmarks[0] if world_landmarks else None

        def _get(idx: int) -> Tuple[float, float, float, float, float, float]:
            """Get joint data: (world_x, world_y, world_z, norm_u, norm_v, confidence)."""
            lm_2d = landmarks_list[idx]
            u = float(lm_2d.x)
            v = float(lm_2d.y)
            conf = float(getattr(lm_2d, "visibility", 1.0))
            if world_list:
                lm_3d = world_list[idx]
                return float(lm_3d.x), float(lm_3d.y), float(lm_3d.z), u, v, conf
            return u, v, 0.0, u, v, conf

        def _avg(
            a: Tuple[float, float, float, float, float, float],
            b: Tuple[float, float, float, float, float, float],
        ) -> Tuple[float, float, float, float, float, float]:
            return (
                (a[0] + b[0]) / 2.0,
                (a[1] + b[1]) / 2.0,
                (a[2] + b[2]) / 2.0,
                (a[3] + b[3]) / 2.0,
                (a[4] + b[4]) / 2.0,
                (a[5] + b[5]) / 2.0,
            )

        mp_pose = pose_module.PoseLandmark

        left_hip = _get(mp_pose.LEFT_HIP.value)
        right_hip = _get(mp_pose.RIGHT_HIP.value)
        left_shoulder = _get(mp_pose.LEFT_SHOULDER.value)
        right_shoulder = _get(mp_pose.RIGHT_SHOULDER.value)
        left_knee = _get(mp_pose.LEFT_KNEE.value)
        right_knee = _get(mp_pose.RIGHT_KNEE.value)
        left_ankle = _get(mp_pose.LEFT_ANKLE.value)
        right_ankle = _get(mp_pose.RIGHT_ANKLE.value)
        left_elbow = _get(mp_pose.LEFT_ELBOW.value)
        right_elbow = _get(mp_pose.RIGHT_ELBOW.value)
        left_wrist = _get(mp_pose.LEFT_WRIST.value)
        right_wrist = _get(mp_pose.RIGHT_WRIST.value)
        left_ear = _get(mp_pose.LEFT_EAR.value)
        right_ear = _get(mp_pose.RIGHT_EAR.value)
        nose = _get(mp_pose.NOSE.value)

        pelvis = _avg(left_hip, right_hip)
        shoulder_center = _avg(left_shoulder, right_shoulder)
        spine = _lerp(pelvis, shoulder_center, 0.4)
        chest = _lerp(pelvis, shoulder_center, 0.8)
        neck = shoulder_center
        head = _avg(_avg(left_ear, right_ear), nose)

        joints = {
            "pelvis": _pose_joint(pelvis),
            "spine": _pose_joint(spine),
            "chest": _pose_joint(chest),
            "neck": _pose_joint(neck),
            "head": _pose_joint(head),
            "l_shoulder": _pose_joint(left_shoulder),
            "r_shoulder": _pose_joint(right_shoulder),
            "l_elbow": _pose_joint(left_elbow),
            "r_elbow": _pose_joint(right_elbow),
            "l_wrist": _pose_joint(left_wrist),
            "r_wrist": _pose_joint(right_wrist),
            "l_hip": _pose_joint(left_hip),
            "r_hip": _pose_joint(right_hip),
            "l_knee": _pose_joint(left_knee),
            "r_knee": _pose_joint(right_knee),
            "l_ankle": _pose_joint(left_ankle),
            "r_ankle": _pose_joint(right_ankle),
        }
        return joints

    def _resolve_model_path(self) -> str:
        candidate = Path(self.settings.model_path) if self.settings.model_path else _default_model_path()
        if not candidate.exists():
            raise FileNotFoundError(
                "Pose landmarker model not found. "
                "Set provider.model_path in config or place model at models/pose_landmarker.task"
            )
        return str(candidate)


def _pose_joint(data: Tuple[float, float, float, float, float, float]) -> PoseJoint:
    return PoseJoint(x=data[0], y=data[1], z=data[2], c=data[5], u=data[3], v=data[4])


def _lerp(
    a: Tuple[float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
        a[4] + (b[4] - a[4]) * t,
        a[5] + (b[5] - a[5]) * t,
    )




def _default_model_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "models" / "pose_landmarker.task"
