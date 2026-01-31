from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from od2blender.pipeline import resolve_blender_path


def _ensure_blend_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".blend":
        return path
    return path.with_suffix(".blend")


def build_rigify_out_path(model_path: Path, out_path: Path | None) -> Path:
    if out_path is None:
        out_path = model_path.with_name(f"{model_path.stem}_rigify.blend")
    out_path = out_path.expanduser().resolve()
    return _ensure_blend_suffix(out_path)


def run_rigify_model(
    model_path: Path,
    out_path: Path | None,
    blender_path: Path | None,
    model_object: str | None = None,
    keep_helpers: bool = False,
    open_blender: bool = False,
    log_info: Callable[[str], None] | None = None,
    log_warn: Callable[[str], None] | None = None,
) -> Path:
    blender_resolved = resolve_blender_path(blender_path)
    if blender_resolved is None:
        raise FileNotFoundError("Blender executable not found")

    model_path = model_path.expanduser().resolve()
    out_path_resolved = build_rigify_out_path(model_path, out_path)

    script_path = Path(__file__).parent / "blender" / "rigify_model.py"
    cmd = [
        str(blender_resolved),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--model",
        str(model_path),
        "--out",
        str(out_path_resolved),
    ]
    if model_object:
        cmd.extend(["--model-object", str(model_object)])
    if keep_helpers:
        cmd.append("--keep-helpers")

    if log_info:
        log_info("Running Blender: {}".format(" ".join(cmd)))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout and log_info:
        for line in result.stdout.splitlines():
            log_info("[blender] {}".format(line))
    if result.stderr and log_warn:
        for line in result.stderr.splitlines():
            log_warn("[blender] {}".format(line))

    if result.returncode != 0:
        raise RuntimeError(f"Rigify generation failed with code {result.returncode}")

    if open_blender:
        try:
            subprocess.Popen([str(blender_resolved), str(out_path_resolved)])
            if log_info:
                log_info("Launched Blender UI")
        except Exception as exc:
            if log_warn:
                log_warn(f"Failed to launch Blender UI: {exc}")

    return out_path_resolved
