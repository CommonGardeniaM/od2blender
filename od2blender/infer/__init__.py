from __future__ import annotations

from od2blender.infer.base import InferenceBackend, InferenceError, InferenceResult


def get_backend(backend_id: str) -> InferenceBackend | None:
    from od2blender.infer.registry import get_backend as _get_backend

    return _get_backend(backend_id)


def list_backends() -> list[str]:
    from od2blender.infer.registry import list_backends as _list_backends

    return _list_backends()

__all__ = [
    "InferenceBackend",
    "InferenceError",
    "InferenceResult",
    "get_backend",
    "list_backends",
]
