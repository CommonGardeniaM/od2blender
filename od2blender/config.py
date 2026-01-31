from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).parent.parent / ".config"
CONFIG_PATH = CONFIG_DIR / "config.json"


def get_config() -> dict[str, Any]:
    """Load config from ~/.config/od2blender/config.json."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save config to ~/.config/od2blender/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_default_video_path() -> Path | None:
    """Get the default video path from config, if it exists."""
    config = get_config()
    path_str = config.get("default_video_path")
    if path_str:
        path = Path(path_str).expanduser()
        if path.exists():
            return path
    return None


def get_default_blender_path() -> Path | None:
    """Get the default blender path from config, if it exists."""
    config = get_config()
    path_str = config.get("default_blender_path")
    if path_str:
        path = Path(path_str).expanduser()
        if path.exists():
            return path
    return None


def get_default_pose3d_source() -> str | None:
    """Get the default pose3d source from config, if it exists."""
    config = get_config()
    source = config.get("default_pose3d_source")
    if isinstance(source, str):
        value = source.strip().lower()
        if value in {"projected", "raw"}:
            return value
    return None


def get_default_model_path() -> Path | None:
    """Get the default model path from config, if it exists."""
    config = get_config()
    path_str = config.get("default_model_path")
    if path_str:
        path = Path(path_str).expanduser()
        if path.exists():
            return path
    return None


def get_default_model_object() -> str | None:
    """Get the default model object name from config, if it exists."""
    config = get_config()
    name = config.get("default_model_object")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def save_default_paths(
    video_path: Path | None,
    blender_path: Path | None,
    backend: str | None = None,
    processors: list[str] | None = None,
    exporters: list[str] | None = None,
    pose3d_source: str | None = None,
    model_path: Path | None = None,
    model_object: str | None = None,
) -> None:
    """Save default settings to config."""
    config = get_config()
    if video_path:
        config["default_video_path"] = str(video_path.expanduser().resolve())
    if blender_path:
        config["default_blender_path"] = str(blender_path.expanduser().resolve())
    if backend:
        config["default_backend"] = backend
    if processors is not None:
        config["default_processors"] = list(processors)
    if exporters is not None:
        config["default_exporters"] = list(exporters)
    if pose3d_source:
        config["default_pose3d_source"] = str(pose3d_source).strip().lower()
    if model_path:
        config["default_model_path"] = str(model_path.expanduser().resolve())
    if model_object:
        config["default_model_object"] = str(model_object).strip()
    save_config(config)


def get_default_backend() -> str | None:
    """Get the default backend id from config, if it exists."""
    config = get_config()
    backend = config.get("default_backend")
    if isinstance(backend, str) and backend:
        return backend
    return None


def get_default_processors() -> list[str] | None:
    """Get the default processors list from config, if it exists."""
    config = get_config()
    processors = config.get("default_processors")
    if isinstance(processors, list) and all(isinstance(item, str) for item in processors):
        return processors
    return None


def get_default_exporters() -> list[str] | None:
    """Get the default exporters list from config, if it exists."""
    config = get_config()
    exporters = config.get("default_exporters")
    if isinstance(exporters, list) and all(isinstance(item, str) for item in exporters):
        return exporters
    return None
