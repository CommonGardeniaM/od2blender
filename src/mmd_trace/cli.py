"""Command-line interface for mmd-trace."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mmd_trace.io import load_pmx

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_list_bones(args: argparse.Namespace) -> int:
    """Handle list-bones command."""
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
                        "ik_target": bone.ik.target_index if bone.ik is not None else None,
                        "ik_links": [link.bone_index for link in bone.ik.links] if bone.ik is not None else [],
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
        else:  # table
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
        
    except FileNotFoundError as e:
        LOG.error("File not found: %s", e)
        return 1


def cmd_pose_vpd(args: argparse.Namespace) -> int:
    """Handle pose-vpd command."""
    try:
        from mmd_trace.io.pose import generate_vpd_from_image

        generate_vpd_from_image(
            image_path=args.image,
            pmx_path=args.pmx,
            out_path=args.out,
            task_model=args.task_model,
            model_type=args.model_type,
            axis_x=args.axis_x,
            axis_y=args.axis_y,
            axis_z=args.axis_z,
            vis_th=args.vis_th,
            det_conf=args.det_conf,
            print_vpd=args.print_vpd,
            print_debug=args.print_debug,
        )
        return 0
    except FileNotFoundError as e:
        LOG.error("File not found: %s", e)
        return 1
    except Exception as e:
        LOG.error("Error generating VPD: %s", e)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="mmd-trace",
        description="Monocular video to MMD VMD motion converter",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-bones command
    bones_parser = subparsers.add_parser(
        "list-bones",
        help="List bones from PMX file",
    )
    bones_parser.add_argument(
        "--pmx",
        required=True,
        help="Path to PMX file",
    )
    bones_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    bones_parser.add_argument(
        "--out",
        help="Output file path (default: stdout)",
    )

    pose_parser = subparsers.add_parser(
        "pose-vpd",
        help="Generate VPD from a single image using MediaPipe Tasks API",
    )
    pose_parser.add_argument("--pmx", required=True, help="Path to PMX file")
    pose_parser.add_argument("--image", required=True, help="Input image path")
    pose_parser.add_argument("--out", default="pose.vpd", help="Output VPD path")
    pose_parser.add_argument(
        "--task_model",
        default=None,
        help="MediaPipe Tasks API model file (.task). If omitted, downloads by --model_type",
    )
    pose_parser.add_argument(
        "--model_type",
        choices=["lite", "full", "heavy"],
        default="full",
        help="Model type to download when --task_model is not provided",
    )
    pose_parser.add_argument("--axis_x", type=float, default=1.0, help="Axis mapping for X")
    pose_parser.add_argument("--axis_y", type=float, default=1.0, help="Axis mapping for Y")
    pose_parser.add_argument("--axis_z", type=float, default=-1.0, help="Axis mapping for Z")
    pose_parser.add_argument("--vis_th", type=float, default=0.2, help="Landmark visibility threshold")
    pose_parser.add_argument("--det_conf", type=float, default=0.5, help="Detection confidence")
    pose_parser.add_argument(
        "--print_vpd",
        action="store_true",
        help="Print the generated VPD text to stdout",
    )
    pose_parser.add_argument(
        "--print_debug",
        action="store_true",
        help="Print debug stats (visibility and scale)",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "list-bones":
        return cmd_list_bones(args)
    if args.command == "pose-vpd":
        return cmd_pose_vpd(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
