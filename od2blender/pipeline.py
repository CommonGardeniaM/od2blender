from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path

from loguru import logger

from od2blender.export import ExportContext, get_exporter
from od2blender.infer import get_backend
from od2blender.infer.base import InferenceError
from od2blender.ir.frames import normalize_frames
from od2blender.ir.schema import IR_VERSION, SCHEMA_NAME, SCHEMA_VERSION, build_ir_payload
from od2blender.json_utils import save_json
from od2blender.process import ProcessorContext, get_processor, run_processors
from od2blender.video_meta import VideoMetaError, read_video_meta

DEFAULT_PLANE_WIDTH = 2.0
DEFAULT_Z_DEPTH = 0.01
META_VERSION = "od2blender.meta.v1"


class ExitCode(IntEnum):
    SUCCESS = 0
    VIDEO_META_FAILURE = 10
    TRACK_FAILURE = 20
    BLENDER_NOT_FOUND = 30
    BLENDER_IMPORT_FAILURE = 31
    OUTPUT_WRITE_FAILURE = 40


def setup_logging(log_path: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def resolve_out_dir(video_path: Path, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    stem = video_path.stem
    suffix = "_od2blender"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return video_path.with_name(f"{stem}{suffix}")


def resolve_run_dir(base_dir: Path, started_at: datetime, mode: str) -> Path:
    stamp = started_at.astimezone().strftime("%Y%m%d_%H%M")
    base_name = f"{stamp}_{mode}"
    candidate = base_dir / base_name
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = base_dir / f"{base_name}_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_blender_path(blender_path: Path | None) -> Path | None:
    if blender_path is not None:
        candidate = Path(blender_path).expanduser()
        if candidate.is_file():
            return candidate
        return None
    env_path = os.environ.get("BLENDER_BIN")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
    which_path = shutil.which("blender")
    if which_path:
        return Path(which_path)
    return None


def get_blender_version(blender_path: Path | None) -> str | None:
    if blender_path is None:
        return None
    try:
        result = subprocess.run(
            [str(blender_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    output = (result.stdout or "").splitlines()
    if not output:
        return None
    return output[0].strip()


def build_meta(
    video_meta: dict,
    detector_meta: dict,
    blender_version: str | None,
    blender_path: Path | None,
    started_at: datetime,
    duration_sec: float,
    pipeline_meta: dict,
    tracks_schema: dict,
) -> dict:
    return {
        "version": META_VERSION,
        "run": {
            "started_at": started_at.isoformat(),
            "duration_sec": duration_sec,
        },
        "env": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "blender": blender_version,
        },
        "detector": detector_meta,
        "video": video_meta,
        "blender_path": str(blender_path) if blender_path else None,
        "pipeline": pipeline_meta,
        "tracks_schema": tracks_schema,
    }


def run_pipeline(
    video_path: Path,
    out_dir: Path | None,
    blender_path: Path | None,
    open_blender: bool,
    no_blend: bool,
    backend: str | None,
    processors: list[str] | None,
    exporters: list[str] | None,
    pose_scale: float,
    pose_model: Path | None,
) -> int:
    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc)

    video_path = video_path.expanduser().resolve()

    # Warn if input video is inside an existing output directory
    if "_od2blender" in str(video_path.parent):
        logger.warning(
            "Input video appears to be inside an existing output directory: {}. "
            "This may cause nested directory structures. Using original video is recommended.",
            video_path.parent,
        )

    backend_id = backend or "pose3d"
    backend_impl = get_backend(backend_id)
    mode_name = backend_impl.mode if backend_impl else backend_id

    base_dir = resolve_out_dir(video_path, out_dir).expanduser().resolve()
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Failed to create output directory: {exc}", file=sys.stderr)
        return int(ExitCode.OUTPUT_WRITE_FAILURE)

    run_dir = resolve_run_dir(base_dir, started_at, mode_name)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Failed to create run output directory: {exc}", file=sys.stderr)
        return int(ExitCode.OUTPUT_WRITE_FAILURE)

    log_path = run_dir / "run.log"
    setup_logging(log_path)

    logger.info("Starting od2blender")
    logger.info("Video path: {}", video_path)
    logger.info("Output base dir: {}", base_dir)
    logger.info("Output run dir: {}", run_dir)
    logger.info("Backend: {}", backend_id)

    if backend_impl is None:
        logger.error("Unknown backend: {}", backend_id)
        return int(ExitCode.TRACK_FAILURE)

    if not video_path.is_file():
        logger.error("Video path does not exist: {}", video_path)
        return int(ExitCode.VIDEO_META_FAILURE)

    processor_ids = list(processors) if processors is not None else list(
        backend_impl.default_processors
    )
    exporter_ids = list(exporters) if exporters is not None else ["tracks", "blender"]
    if no_blend and "blender" in exporter_ids:
        exporter_ids = [item for item in exporter_ids if item != "blender"]
    if "tracks" in exporter_ids:
        exporter_ids = [item for item in exporter_ids if item != "tracks"]
    exporter_ids = ["tracks", *exporter_ids]
    if open_blender and "blender" not in exporter_ids:
        logger.warning("--open ignored because blender export is disabled")

    logger.info(
        "Processors: {}",
        ", ".join(processor_ids) if processor_ids else "(none)",
    )
    logger.info("Exporters: {}", ", ".join(exporter_ids))

    meta_payload: dict | None = None
    meta_path = run_dir / "meta.json"

    try:
        media_dir = base_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        target_video = (media_dir / f"source{video_path.suffix}").resolve()
        if target_video.exists():
            try:
                if target_video.stat().st_size != video_path.stat().st_size:
                    logger.warning(
                        "Existing media differs in size, keeping existing file: {}",
                        target_video,
                    )
            except OSError:
                logger.warning("Existing media found, keeping file: {}", target_video)
        else:
            shutil.copy2(video_path, target_video)
            logger.info("Copied video to {}", target_video)

        video_meta = read_video_meta(video_path)
        logger.info(
            "Video meta: {}x{}, fps={}, frames={}",
            video_meta.width,
            video_meta.height,
            video_meta.fps,
            video_meta.frame_count,
        )

        backend_config = {"pose_scale": pose_scale, "pose_model": pose_model}
        inference = backend_impl.infer(
            video_path,
            video_meta,
            config=backend_config,
            log_warn=logger.warning,
        )
        logger.info("Inference complete")

        processor_instances = []
        for processor_id in processor_ids:
            processor = get_processor(processor_id)
            if processor is None:
                logger.error("Unknown processor: {}", processor_id)
                return int(ExitCode.TRACK_FAILURE)
            processor_instances.append(processor)

        processor_settings = {
            "plane_width": DEFAULT_PLANE_WIDTH,
            "z_depth": DEFAULT_Z_DEPTH,
            "pose_scale": pose_scale,
        }
        processor_context = ProcessorContext(
            video_meta=video_meta,
            settings=processor_settings,
            log_warn=logger.warning,
        )
        processed_frames, processing_meta = run_processors(
            inference.frames,
            processor_instances,
            processor_context,
        )

        tracks_payload = build_ir_payload(
            video_meta=video_meta.to_dict(),
            detector_meta=inference.detector_meta,
            frames=normalize_frames(processed_frames),
            mode=inference.mode,
            skeleton=inference.skeleton,
            units=inference.units,
            axis=inference.axis,
            processing=processing_meta,
            extra=inference.extra,
        )
        tracks_path = (run_dir / "tracks.json").resolve()

        blender_resolved = resolve_blender_path(blender_path)
        blender_version = get_blender_version(blender_resolved)
        pipeline_meta = {
            "backend": backend_id,
            "processors": processor_ids,
            "exporters": exporter_ids,
        }
        tracks_schema = {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "ir_version": IR_VERSION,
        }
        meta_payload = build_meta(
            video_meta=video_meta.to_dict(),
            detector_meta=inference.detector_meta,
            blender_version=blender_version,
            blender_path=blender_resolved,
            started_at=started_at,
            duration_sec=0.0,
            pipeline_meta=pipeline_meta,
            tracks_schema=tracks_schema,
        )

        exporter_context = ExportContext(
            run_dir=run_dir,
            video_path=target_video,
            tracks_path=tracks_path,
            scene_path=(run_dir / "scene.blend").resolve(),
            blender_path=blender_resolved,
            importer_path=(Path(__file__).parent / "blender" / "importer.py").resolve(),
            plane_width=DEFAULT_PLANE_WIDTH,
            z_depth=DEFAULT_Z_DEPTH,
            open_blender=open_blender,
            log_info=logger.info,
            log_warn=logger.warning,
        )

        for exporter_id in exporter_ids:
            exporter = get_exporter(exporter_id)
            if exporter is None:
                logger.error("Unknown exporter: {}", exporter_id)
                return int(ExitCode.OUTPUT_WRITE_FAILURE)
            try:
                exporter.export(tracks_payload, exporter_context)
                if exporter_id == "tracks":
                    logger.info("Wrote tracks.json")
                if exporter_id == "blender":
                    logger.info("scene.blend generated")
            except FileNotFoundError as exc:
                logger.exception("Executable not found: {}", exc)
                return int(ExitCode.BLENDER_NOT_FOUND)
            except RuntimeError as exc:
                if exporter_id == "blender":
                    logger.error("Blender importer failed: {}", exc)
                    return int(ExitCode.BLENDER_IMPORT_FAILURE)
                logger.exception("Export failure: {}", exc)
                return int(ExitCode.OUTPUT_WRITE_FAILURE)

        return int(ExitCode.SUCCESS)
    except VideoMetaError as exc:
        logger.exception("Video metadata error: {}", exc)
        return int(ExitCode.VIDEO_META_FAILURE)
    except InferenceError as exc:
        logger.exception("Tracking error: {}", exc)
        return int(ExitCode.TRACK_FAILURE)
    except FileNotFoundError as exc:
        logger.exception("Executable not found: {}", exc)
        return int(ExitCode.BLENDER_NOT_FOUND)
    except OSError as exc:
        logger.exception("Output write error: {}", exc)
        return int(ExitCode.OUTPUT_WRITE_FAILURE)
    except Exception as exc:
        logger.exception("Unexpected error: {}", exc)
        return int(ExitCode.OUTPUT_WRITE_FAILURE)
    finally:
        if meta_payload is not None:
            meta_payload["run"]["duration_sec"] = round(
                time.monotonic() - start_time, 3
            )
            try:
                save_json(meta_payload, meta_path)
                logger.info("Wrote meta.json")
            except Exception as exc:
                logger.error("Failed to write meta.json: {}", exc)
