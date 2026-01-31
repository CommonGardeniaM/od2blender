from __future__ import annotations

from pathlib import Path


def make_blender_relpath(scene_path: Path, media_path: Path) -> str:
    scene_dir = scene_path.resolve().parent
    media_path = media_path.resolve()
    try:
        rel = media_path.relative_to(scene_dir)
    except ValueError:
        return str(media_path)
    return f"//{rel.as_posix()}"
