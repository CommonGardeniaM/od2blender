"""CLI entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .io_pose import write_pose_json, read_pose_json
from .pose_provider import MediaPipePoseProvider
from .pose_provider.mediapipe_provider import MediaPipeSettings
from .mmd_io import load_pmx_model, write_vmd_motion
from .axis_auto import infer_axis_map
from .diagnose import collect_diagnostics, write_diagnostics
from .retarget.retargeter import retarget_to_vmd
from .visualize import VisualizeConfig, visualize_pose_on_image

LOG = logging.getLogger(__name__)


def _setup_trace_parser(sub: argparse._SubParsersAction) -> None:
    trace = sub.add_parser("trace", help="Run full pipeline")
    trace.add_argument("--image", required=True, help="Input image path")
    trace.add_argument("--pmx", required=True, help="PMX model path")
    trace.add_argument("--out", required=True, help="Output directory")
    trace.add_argument("--config", required=False, help="Config YAML/JSON")
    trace.add_argument("--provider", default="mediapipe", help="Pose provider")
    trace.add_argument("--pattern", default=None, help="Center/Groove pattern A/B")
    trace.add_argument("--debug_overlay", action="store_true")
    trace.add_argument("--diagnose", action="store_true", help="Write diagnostic report")


def _setup_visualize_parser(sub: argparse._SubParsersAction) -> None:
    viz = sub.add_parser("visualize", help="Visualize pose overlay on image")
    viz.add_argument("--image", required=True, help="Input image path")
    viz.add_argument("--pose", required=True, help="Pose JSON path")
    viz.add_argument("--out", required=True, help="Output image path")
    viz.add_argument("--no-skeleton", action="store_true", help="Hide skeleton lines")
    viz.add_argument("--no-joints", action="store_true", help="Hide joint markers")
    viz.add_argument("--labels", action="store_true", help="Show joint name labels")
    viz.add_argument("--no-info", action="store_true", help="Hide frame info")
    viz.add_argument("--line-thickness", type=int, default=2)
    viz.add_argument("--joint-radius", type=int, default=5)
    viz.add_argument("--low-conf-threshold", type=float, default=0.5)


def _run_trace(args: argparse.Namespace) -> None:
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
    debug_path = str(out_dir / "debug_overlay.jpg") if args.debug_overlay else None
    
    # Run inference on single image
    raw_seq = provider.infer(args.image, debug_overlay_path=debug_path)

    axis_map = {"swap": config.axis_map.swap, "invert": config.axis_map.invert}
    pmx = load_pmx_model(args.pmx)
    if config.axis_auto.enabled:
        from .retarget.retargeter import _compute_model_basis
        rest_basis = _compute_model_basis(pmx, config.bone_map)
        inferred = infer_axis_map(
            raw_seq,
            min_confidence=config.retarget.min_confidence,
            joint_min_confidence=config.retarget.joint_min_confidence,
            source=config.axis_auto.source,
            max_frames=config.axis_auto.max_frames,
            rest_basis=rest_basis,
        )
        if inferred is not None:
            axis_map = {"swap": inferred.swap, "invert": inferred.invert}
            LOG.info("Axis auto-infer swap=%s invert=%s score=%.3f", inferred.swap, inferred.invert, inferred.score)

    raw_seq.meta.axis_map = axis_map
    write_pose_json(raw_seq, str(out_dir / "pose.json"))

    # No smoothing for single image
    pose_seq = raw_seq

    retarget = retarget_to_vmd(
        pose_seq,
        pmx=pmx,
        bone_map=config.bone_map,
        axis_map=axis_map,
        min_confidence=config.retarget.min_confidence,
        joint_min_confidence=config.retarget.joint_min_confidence,
        center_pattern=config.center_groove.pattern,
        center_lowpass=config.center_groove.lowpass_alpha,
        root_joint=config.center_groove.root_joint,
        facing_mode=config.facing.mode,
        facing_yaw_offset_deg=config.facing.yaw_offset_deg,
        facing_source=config.facing.source,
    )

    write_vmd_motion(str(out_dir / "motion.vmd"), model_name=pmx.name, bone_frames=retarget.bone_frames)
    LOG.info("Wrote VMD to %s", out_dir / "motion.vmd")

    if args.diagnose:
        diag = collect_diagnostics(
            pose_seq=pose_seq,
            pmx=pmx,
            bone_frames=retarget.bone_frames,
            bone_map=config.bone_map,
            axis_map=axis_map,
            min_confidence=config.retarget.min_confidence,
            joint_min_confidence=config.retarget.joint_min_confidence,
        )
        write_diagnostics(diag, out_dir / "diagnostic_report.md")
        LOG.info("Wrote diagnostics to %s", out_dir / "diagnostic_report.md")


def _run_visualize(args: argparse.Namespace) -> None:
    LOG.info("Loading pose data from %s", args.pose)
    pose_sequence = read_pose_json(args.pose)

    viz_config = VisualizeConfig(
        show_skeleton=not args.no_skeleton,
        show_joints=not args.no_joints,
        show_labels=args.labels,
        show_frame_info=not args.no_info,
        line_thickness=args.line_thickness,
        joint_radius=args.joint_radius,
        low_confidence_threshold=args.low_conf_threshold,
    )

    visualize_pose_on_image(
        args.image,
        pose_sequence,
        args.out,
        viz_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmd-trace")
    sub = parser.add_subparsers(dest="command")

    _setup_trace_parser(sub)
    _setup_visualize_parser(sub)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.command == "trace":
        _run_trace(args)
    elif args.command == "visualize":
        _run_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
