from __future__ import annotations

from pathlib import Path

import typer

from od2blender.config import (
    get_default_blender_path,
    get_default_backend,
    get_default_exporters,
    get_default_model_object,
    get_default_model_path,
    get_default_processors,
    get_default_pose3d_source,
    get_default_video_path,
    save_default_paths,
)
from od2blender.pipeline import run_pipeline
from od2blender.rigify import run_rigify_model

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def rigify_model(
    model_path: Path | None = typer.Argument(
        None,
        help="Path to the model file (default: from config)",
    ),
    out_path: Path | None = typer.Option(
        None,
        "--out",
        help="Output .blend path (default: <model>_rigify.blend)",
    ),
    blender_path: Path | None = typer.Option(
        None,
        "--blender",
        help="Blender executable path (default: config, PATH, or BLENDER_BIN)",
    ),
    model_object: str | None = typer.Option(
        None,
        "--model-object",
        help="Object name inside the model file (optional)",
    ),
    keep_helpers: bool = typer.Option(
        False,
        "--keep-helpers",
        help="Keep metarig and widget collections visible",
    ),
    open_blender: bool = typer.Option(
        False,
        "--open",
        help="Open Blender UI after rigify generation",
    ),
) -> None:
    if model_path is None:
        model_path = get_default_model_path()
    if blender_path is None:
        blender_path = get_default_blender_path()
    if model_object is None:
        model_object = get_default_model_object()

    if model_path is None:
        typer.echo("Error: model_path is required (or set default in config)", err=True)
        raise typer.Exit(code=1)

    try:
        output_path = run_rigify_model(
            model_path=model_path,
            out_path=out_path,
            blender_path=blender_path,
            model_object=model_object,
            keep_helpers=keep_helpers,
            open_blender=open_blender,
            log_info=typer.echo,
            log_warn=lambda msg: typer.echo(msg, err=True),
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    save_default_paths(
        video_path=None,
        blender_path=blender_path,
        model_path=model_path,
        model_object=model_object,
    )
    typer.echo(f"Rigify model saved: {output_path}")


@app.command(name="run")
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
    pose3d_source: str | None = typer.Option(
        None,
        "--pose3d-source",
        help="Pose source for Blender export: projected or raw",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model",
        help="Path to 3D model for Rigify retargeting",
    ),
    model_object: str | None = typer.Option(
        None,
        "--model-object",
        help="Object name inside the model file (optional)",
    ),
    test_motion: bool = typer.Option(
        False,
        "--test-motion",
        help="Generate a test crouch motion for the model",
    ),
) -> None:
    # Resolve defaults from config if not provided
    if video_path is None:
        video_path = get_default_video_path()
    if blender_path is None:
        blender_path = get_default_blender_path()
    if backend is None:
        backend = get_default_backend()
    if pose3d_source is None:
        pose3d_source = get_default_pose3d_source() or "projected"
    if model_path is None:
        model_path = get_default_model_path()
    if model_object is None:
        model_object = get_default_model_object()

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
        pose3d_source=pose3d_source,
        model_path=model_path,
        model_object=model_object,
        test_motion=test_motion,
    )

    # Save the used paths as defaults for next time
    save_default_paths(
        video_path,
        blender_path,
        backend=backend,
        processors=processors_list,
        exporters=exporters_list,
        pose3d_source=pose3d_source,
        model_path=model_path,
        model_object=model_object,
    )

    raise typer.Exit(code=int(exit_code))


if __name__ == "__main__":
    app()
