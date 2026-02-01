"""CLI entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .io_pose import write_pose_json
from .pose_provider import MediaPipePoseProvider
from .pose_provider.mediapipe_provider import MediaPipeSettings
from .smoothing import smooth_pose_sequence
from .mmd_io import load_pmx_model, write_vmd_motion
from .retarget.retargeter import retarget_to_vmd


LOG = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmd-trace")
    sub = parser.add_subparsers(dest="command")

    trace = sub.add_parser("trace", help="Run full pipeline")
    trace.add_argument("--video", required=True, help="Input video path")
    trace.add_argument("--pmx", required=True, help="PMX model path")
    trace.add_argument("--out", required=True, help="Output directory")
    trace.add_argument("--config", required=False, help="Config YAML/JSON")
    trace.add_argument("--provider", default="mediapipe", help="Pose provider")
    smooth_group = trace.add_mutually_exclusive_group()
    smooth_group.add_argument("--smooth", dest="smooth", action="store_true", help="Enable smoothing")
    smooth_group.add_argument("--no-smooth", dest="smooth", action="store_false", help="Disable smoothing")
    trace.set_defaults(smooth=None)
    trace.add_argument("--pattern", default=None, help="Center/Groove pattern A/B")
    trace.add_argument("--fps_override", type=int, default=None)
    trace.add_argument("--debug_overlay", action="store_true")

    args = parser.parse_args()
    if args.command != "trace":
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(args.config)
    if args.pattern:
        config.center_groove.pattern = args.pattern

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.provider != "mediapipe":
        raise ValueError("Only mediapipe provider is available in MVP")

    provider_settings = MediaPipeSettings(
        min_confidence=config.provider.min_confidence,
        model_complexity=config.provider.model_complexity,
        model_path=config.provider.model_path or None,
    )
    provider = MediaPipePoseProvider(settings=provider_settings)
    debug_path = str(out_dir / "debug_overlay.mp4") if args.debug_overlay else None
    raw_seq = provider.infer(args.video, fps_override=args.fps_override, debug_overlay_path=debug_path)
    raw_seq.meta.axis_map = {"swap": config.axis_map.swap, "invert": config.axis_map.invert}
    write_pose_json(raw_seq, str(out_dir / "pose_raw.json"))

    use_smooth = config.smoothing.enabled if args.smooth is None else args.smooth
    if use_smooth:
        smooth_seq = smooth_pose_sequence(
            raw_seq,
            max_gap=config.smoothing.max_gap,
            ema_alpha=config.smoothing.ema_alpha,
            min_confidence=config.smoothing.min_confidence,
        )
    else:
        smooth_seq = raw_seq

    smooth_seq.meta.axis_map = raw_seq.meta.axis_map
    write_pose_json(smooth_seq, str(out_dir / "pose_smooth.json"))

    pmx = load_pmx_model(args.pmx)
    retarget = retarget_to_vmd(
        smooth_seq,
        pmx=pmx,
        bone_map=config.bone_map,
        axis_map={"swap": config.axis_map.swap, "invert": config.axis_map.invert},
        min_confidence=config.retarget.min_confidence,
        center_pattern=config.center_groove.pattern,
        center_lowpass=config.center_groove.lowpass_alpha,
        root_joint=config.center_groove.root_joint,
    )

    write_vmd_motion(str(out_dir / "motion.vmd"), model_name=pmx.name, bone_frames=retarget.bone_frames)
    LOG.info("Wrote VMD to %s", out_dir / "motion.vmd")
