from __future__ import annotations

import json
from pathlib import Path

import pytest

from od2blender.config import (
    CONFIG_PATH,
    get_config,
    get_default_backend,
    get_default_blender_path,
    get_default_exporters,
    get_default_processors,
    get_default_video_path,
    save_config,
    save_default_paths,
)


def test_get_config_returns_empty_dict_when_no_file() -> None:
    # Ensure config file doesn't exist
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
    config = get_config()
    assert config == {}


def test_save_and_get_config() -> None:
    test_config = {"test_key": "test_value", "number": 42}
    save_config(test_config)
    loaded = get_config()
    assert loaded == test_config


def test_save_default_paths(tmp_path: Path) -> None:
    # Create actual files for the test
    video = tmp_path / "video.mp4"
    blender = tmp_path / "blender.exe"
    video.touch()
    blender.touch()
    save_default_paths(video, blender)
    config = get_config()
    # Paths are normalized via resolve(), so compare normalized paths
    assert config["default_video_path"] == str(video.resolve())
    assert config["default_blender_path"] == str(blender.resolve())


def test_get_default_video_path_returns_none_when_not_set() -> None:
    save_config({})
    assert get_default_video_path() is None


def test_get_default_blender_path_returns_none_when_not_set() -> None:
    save_config({})
    assert get_default_blender_path() is None


def test_get_default_video_path_returns_path_when_exists(tmp_path: Path) -> None:
    video_file = tmp_path / "test_video.mp4"
    video_file.touch()
    save_config({"default_video_path": str(video_file)})
    result = get_default_video_path()
    assert result == video_file


def test_get_default_video_path_returns_none_when_file_missing() -> None:
    save_config({"default_video_path": "/nonexistent/path/video.mp4"})
    assert get_default_video_path() is None


def test_get_default_blender_path_returns_path_when_exists(tmp_path: Path) -> None:
    blender_file = tmp_path / "blender.exe"
    blender_file.touch()
    save_config({"default_blender_path": str(blender_file)})
    result = get_default_blender_path()
    assert result == blender_file


def test_get_default_blender_path_returns_none_when_file_missing() -> None:
    save_config({"default_blender_path": "/nonexistent/path/blender.exe"})
    assert get_default_blender_path() is None


def test_get_default_backend_returns_none_when_not_set() -> None:
    save_config({})
    assert get_default_backend() is None


def test_get_default_backend_returns_value_when_set() -> None:
    save_config({"default_backend": "pose3d"})
    assert get_default_backend() == "pose3d"


def test_get_default_processors_returns_list_when_set() -> None:
    save_config({"default_processors": ["pose3d_projector"]})
    assert get_default_processors() == ["pose3d_projector"]


def test_get_default_exporters_returns_list_when_set() -> None:
    save_config({"default_exporters": ["tracks", "blender"]})
    assert get_default_exporters() == ["tracks", "blender"]
