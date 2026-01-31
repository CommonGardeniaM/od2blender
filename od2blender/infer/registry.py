from __future__ import annotations

from od2blender.infer.base import InferenceBackend
from od2blender.infer.pose3d_backend import Pose3DBackend
from od2blender.infer.yolo_backend import YoloBackend


_BACKENDS: dict[str, InferenceBackend] = {
    Pose3DBackend.id: Pose3DBackend(),
    YoloBackend.id: YoloBackend(),
}


def get_backend(backend_id: str) -> InferenceBackend | None:
    return _BACKENDS.get(backend_id)


def list_backends() -> list[str]:
    return sorted(_BACKENDS.keys())
