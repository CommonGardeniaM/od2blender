"""Command-line interface for mmd-trace."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mmd_trace.app.single_image_pipeline import generate_vpd_from_image
from mmd_trace.app.spec import AxisTransformSpec, DebugSpec, RunSpec, ValidationError
from mmd_trace.io import load_pmx
from mmd_trace.retarget.pipeline import SolveMode
from mmd_trace.viz.solver_debug import generate_debug_visualization

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _parse_axis_spec(args: argparse.Namespace) -> AxisTransformSpec:
    return AxisTransformSpec(
        axis_x=args.axis_x,
        axis_y=args.axis_y,
        axis_z=args.axis_z,
        axis_matrix=args.axis_matrix,
        flip_y=True,
    )


def cmd_list_bones(args: argparse.Namespace) -> int:
    try:
        model = load_pmx(args.pmx)

        if args.format == "json":
            data = {
                "pmx_file": str(Path(args.pmx)),
                "total_bones": len(model.bones),
                "bones": [
                    {
                        "index": bone.index,
                        "name": bone.name,
                        "name_en": bone.name_en,
                        "position": bone.position.tolist(),
                        "parent": bone.parent_index,
                        "flags": bone.flags,
                        "tail_index": bone.tail_index,
                        "tail_offset": bone.tail_offset.tolist() if bone.tail_offset is not None else None,
                    }
                    for bone in model.bones
                ],
            }
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(json_str, encoding="utf-8")
                LOG.info("Exported bone list to: %s", args.out)
            else:
                print(json_str)
        else:
            lines = []
            lines.append(f"\nPMX File: {args.pmx}")
            lines.append(f"Total Bones: {len(model.bones)}")
            lines.append("-" * 70)
            lines.append(f"{'Index':<6} {'Name':<30} {'Name(EN)':<20} {'Parent'}")
            lines.append("-" * 70)
            for bone in model.bones:
                parent_str = str(bone.parent_index) if bone.parent_index >= 0 else "-"
                lines.append(f"{bone.index:<6} {bone.name:<30} {bone.name_en:<20} {parent_str}")
            lines.append("-" * 70)
            output = "\n".join(lines) + "\n"
            if args.out:
                Path(args.out).write_text(output, encoding="utf-8")
                LOG.info("Exported bone list to: %s", args.out)
            else:
                print(output, end="")

        return 0
    except FileNotFoundError as exc:
        LOG.error("File not found: %s", exc)
        return 1


def cmd_pose_vpd(args: argparse.Namespace) -> int:
    try:
        axis_spec = _parse_axis_spec(args)
        run_spec = RunSpec(
            pmx=args.pmx,
            image=args.image,
            out=args.out,
            vis_th=args.vis_th,
            det_conf=args.det_conf,
            solver=args.solver,
        )
    except ValidationError as exc:
        LOG.error("Invalid arguments: %s", exc)
        return 1

    axis = axis_spec.to_internal()

    try:
        generate_vpd_from_image(
            image_path=run_spec.image,
            pmx_path=run_spec.pmx,
            out_path=run_spec.out,
            axis=axis,
            vis_th=run_spec.vis_th,
            det_conf=run_spec.det_conf,
            solver=run_spec.solver_mode,
            print_vpd=args.print_vpd,
            print_debug=args.print_debug,
        )
        return 0
    except Exception as exc:
        LOG.error("Error generating VPD: %s", exc)
        return 1


def cmd_debug_visualize(args: argparse.Namespace) -> int:
    try:
        axis_spec = _parse_axis_spec(args)
        debug_spec = DebugSpec(
            pmx=args.pmx,
            image=args.image,
            out=args.out,
            vis_th=args.vis_th,
            det_conf=args.det_conf,
            solver=args.solver,
            mode=args.mode,
            project=args.project,
            axis_scale=args.axis_scale,
            dpi=args.dpi,
        )
    except ValidationError as exc:
        LOG.error("Invalid arguments: %s", exc)
        return 1

    axis = axis_spec.to_internal()

    try:
        generate_debug_visualization(
            image_path=debug_spec.image,
            pmx_path=debug_spec.pmx,
            out_dir=debug_spec.out,
            axis=axis,
            vis_th=debug_spec.vis_th,
            det_conf=debug_spec.det_conf,
            solver=SolveMode(debug_spec.solver),
            mode=debug_spec.mode,
            project=debug_spec.project,
            axis_scale=debug_spec.axis_scale,
            dpi=debug_spec.dpi,
        )
        return 0
    except Exception as exc:
        LOG.error("Error generating debug visualization: %s", exc)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mmd-trace",
        description="Single-image pose to MMD VPD converter",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    bones_parser = subparsers.add_parser("list-bones", help="List bones from PMX file")
    bones_parser.add_argument("--pmx", required=True, help="Path to PMX file")
    bones_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    bones_parser.add_argument("--out", help="Output file path (default: stdout)")

    pose_parser = subparsers.add_parser("pose-vpd", help="Generate VPD from a single image")
    pose_parser.add_argument("--pmx", required=True, help="Path to PMX file")
    pose_parser.add_argument("--image", required=True, help="Input image path")
    pose_parser.add_argument("--out", default="pose.vpd", help="Output VPD path")
    pose_parser.add_argument("--axis_x", type=float, default=1.0, help="Axis mapping for X")
    pose_parser.add_argument("--axis_y", type=float, default=1.0, help="Axis mapping for Y")
    pose_parser.add_argument("--axis_z", type=float, default=-1.0, help="Axis mapping for Z")
    pose_parser.add_argument(
        "--axis_matrix",
        default=None,
        help="3x3 axis transform as 9 comma-separated floats (row-major), overrides axis_x/y/z",
    )
    pose_parser.add_argument("--vis_th", type=float, default=0.2, help="Landmark visibility threshold")
    pose_parser.add_argument("--det_conf", type=float, default=0.5, help="Detection confidence")
    pose_parser.add_argument(
        "--solver",
        choices=[SolveMode.ROLL_2D.value, SolveMode.WORLD_3D.value],
        default=SolveMode.ROLL_2D.value,
        help="Solver mode (default: 2d_roll)",
    )
    pose_parser.add_argument("--print_vpd", action="store_true", help="Print the generated VPD text")
    pose_parser.add_argument("--print_debug", action="store_true", help="Print debug info")

    debug_parser = subparsers.add_parser("debug-visualize", help="Generate debug visualization")
    debug_parser.add_argument("--pmx", required=True, help="Path to PMX file")
    debug_parser.add_argument("--image", required=True, help="Input image path")
    debug_parser.add_argument("--out", default="debug_output", help="Output directory")
    debug_parser.add_argument("--axis_x", type=float, default=1.0, help="Axis mapping for X")
    debug_parser.add_argument("--axis_y", type=float, default=1.0, help="Axis mapping for Y")
    debug_parser.add_argument("--axis_z", type=float, default=-1.0, help="Axis mapping for Z")
    debug_parser.add_argument(
        "--axis_matrix",
        default=None,
        help="3x3 axis transform as 9 comma-separated floats (row-major), overrides axis_x/y/z",
    )
    debug_parser.add_argument("--vis_th", type=float, default=0.2, help="Landmark visibility threshold")
    debug_parser.add_argument("--det_conf", type=float, default=0.5, help="Detection confidence")
    debug_parser.add_argument(
        "--solver",
        choices=[SolveMode.ROLL_2D.value, SolveMode.WORLD_3D.value],
        default=SolveMode.ROLL_2D.value,
        help="Solver mode (default: 2d_roll)",
    )
    debug_parser.add_argument(
        "--mode",
        choices=["2d", "3d", "both"],
        default="both",
        help="Visualization mode (default: both)",
    )
    debug_parser.add_argument(
        "--project",
        choices=["2d", "world"],
        default="2d",
        help="2D overlay projection mode (default: 2d)",
    )
    debug_parser.add_argument(
        "--axis_scale",
        type=float,
        default=1.0,
        help="Scale factor for axis length (default: 1.0)",
    )
    debug_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output images (default: 150)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "list-bones":
        return cmd_list_bones(args)
    if args.command == "pose-vpd":
        return cmd_pose_vpd(args)
    if args.command == "debug-visualize":
        return cmd_debug_visualize(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
