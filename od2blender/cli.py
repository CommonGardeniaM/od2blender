from __future__ import annotations

from pathlib import Path

import typer

from od2blender.config import (
    get_default_blender_path,
    get_default_backend,
    get_default_exporters,
    get_default_processors,
    get_default_video_path,
    save_default_paths,
)
from od2blender.pipeline import run_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def main(
    video_path: Path | None = typer.Argument(
        None,
        help="Path to the input video file (default: from config)",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory (default: <video_stem>_od2blender)",
    ),
    blender_path: Path | None = typer.Option(
        None,
        "--blender",
        help="Blender executable path (default: config, PATH, or BLENDER_BIN)",
    ),
    open_blender: bool = typer.Option(
        False,
        "--open",
        help="Open Blender UI after scene.blend is generated",
    ),
    no_blend: bool = typer.Option(
        False,
        "--no-blend",
        help="Skip Blender generation and only output tracks.json",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Inference backend id (default: config or pose3d)",
    ),
    processors: str | None = typer.Option(
        None,
        "--processors",
        help="Comma-separated processor ids (default: backend defaults)",
    ),
    exporters: str | None = typer.Option(
        None,
        "--exporters",
        help="Comma-separated exporter ids (default: tracks,blender)",
    ),
    pose_scale: float = typer.Option(
        1.0,
        "--pose-scale",
        help="Scale factor for pose3d coordinates",
    ),
    pose_model: Path | None = typer.Option(
        None,
        "--pose-model",
        help="Path to MediaPipe pose landmarker model (.task)",
    ),
) -> None:
    # Resolve defaults from config if not provided
    if video_path is None:
        video_path = get_default_video_path()
    if blender_path is None:
        blender_path = get_default_blender_path()
    if backend is None:
        backend = get_default_backend()

    if processors is None:
        processors_list = get_default_processors()
    else:
        processors_list = [item.strip() for item in processors.split(",") if item.strip()]

    if exporters is None:
        exporters_list = get_default_exporters()
    else:
        exporters_list = [item.strip() for item in exporters.split(",") if item.strip()]

    if video_path is None:
        typer.echo("Error: video_path is required (or set default in config)", err=True)
        raise typer.Exit(code=1)

    exit_code = run_pipeline(
        video_path=video_path,
        out_dir=out_dir,
        blender_path=blender_path,
        open_blender=open_blender,
        no_blend=no_blend,
        backend=backend,
        processors=processors_list,
        exporters=exporters_list,
        pose_scale=pose_scale,
        pose_model=pose_model,
    )

    # Save the used paths as defaults for next time
    save_default_paths(
        video_path,
        blender_path,
        backend=backend,
        processors=processors_list,
        exporters=exporters_list,
    )

    raise typer.Exit(code=int(exit_code))


if __name__ == "__main__":
    app()
