from __future__ import annotations

from pathlib import Path

from od2blender.paths import make_blender_relpath


def test_make_blender_relpath_relative(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True)
    media_path = media_dir / "source.mp4"
    media_path.write_bytes(b"")
    scene_path = out_dir / "scene.blend"

    rel = make_blender_relpath(scene_path, media_path)
    assert rel.replace("\\", "/") == "//media/source.mp4"
