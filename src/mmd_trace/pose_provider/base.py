"""Pose provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..io_pose import PoseSequence


class PoseProvider(ABC):
    """Abstract pose provider."""

    @abstractmethod
    def infer(
        self,
        video_path: str,
        fps_override: Optional[int] = None,
        debug_overlay_path: Optional[str] = None,
    ) -> PoseSequence:
        """Infer pose sequence from video."""

        raise NotImplementedError
