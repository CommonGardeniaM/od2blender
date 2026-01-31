from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Rigify rig for a model")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--out", required=True, help="Output .blend path")
    parser.add_argument("--model-object", help="Object name inside the model file")
    parser.add_argument(
        "--keep-helpers",
        action="store_true",
        help="Keep metarig and widget collections visible",
    )
    return parser.parse_args(argv)


def get_script_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parse_args(argv)


def ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)


def main() -> int:
    args = get_script_args()
    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not model_path.is_file():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 1

    ensure_repo_on_path()
    from od2blender.blender import importer

    importer.clear_scene()
    # メッシュのみを抽出（既存リグを無視）
    mesh_obj = importer.import_model(model_path, args.model_object, extract_mesh_only=True)
    # 既存のウェイトをクリア
    importer.strip_mesh_rigging(mesh_obj)
    importer.normalize_mesh_object(mesh_obj)

    importer.ensure_rigify_enabled()
    metarig = importer.add_human_metarig()
    importer.fit_metarig_to_mesh(metarig, mesh_obj)
    rig_obj = importer.generate_rigify_rig(metarig)
    importer.parent_mesh_to_rig(mesh_obj, rig_obj)
    if not args.keep_helpers:
        importer.hide_rigify_helpers(metarig)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"Rigify model saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
