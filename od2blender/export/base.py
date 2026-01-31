from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path


@dataclass(frozen=True)
class ExportContext:
    run_dir: Path
    video_path: Path
    tracks_path: Path
    scene_path: Path
    blender_path: Path | None
    importer_path: Path
    plane_width: float
    z_depth: float
    open_blender: bool
    pose_scale: float
    pose3d_source: str
    model_path: Path | None
    model_object: str | None
    test_motion: bool
    log_info: Callable[[str], None] | None = None
    log_warn: Callable[[str], None] | None = None


class Exporter:
    id: str = ""

    def export(self, payload: dict, context: ExportContext) -> dict:
        raise NotImplementedError
