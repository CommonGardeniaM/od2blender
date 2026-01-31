from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rig diagnostics for od2blender")
    parser.add_argument("--blend", required=True, help="Path to the .blend file")
    parser.add_argument("--mesh", help="Optional mesh object name to focus on")
    parser.add_argument("--out", help="Output JSON report path")
    return parser.parse_args(argv)


def get_script_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parse_args(argv)


def get_object_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector] | None:
    if not obj.bound_box:
        return None
    coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(vec.x for vec in coords)
    min_y = min(vec.y for vec in coords)
    min_z = min(vec.z for vec in coords)
    max_x = max(vec.x for vec in coords)
    max_y = max(vec.y for vec in coords)
    max_z = max(vec.z for vec in coords)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))


def bounds_to_dict(bounds: tuple[Vector, Vector] | None) -> dict | None:
    if not bounds:
        return None
    min_v, max_v = bounds
    size = max_v - min_v
    center = (min_v + max_v) * 0.5
    return {
        "min": [min_v.x, min_v.y, min_v.z],
        "max": [max_v.x, max_v.y, max_v.z],
        "size": [size.x, size.y, size.z],
        "center": [center.x, center.y, center.z],
    }


def count_prefix(names: list[str], prefix: str) -> int:
    return sum(1 for name in names if name.startswith(prefix))


def count_contains(names: list[str], token: str) -> int:
    token_lower = token.lower()
    return sum(1 for name in names if token_lower in name.lower())


def count_side(names: list[str], side: str) -> int:
    suffix = f".{side}"
    alt = f"_{side}"
    return sum(1 for name in names if suffix in name or alt in name)


def find_pose_bone(
    rig_obj: bpy.types.Object,
    candidates: list[str],
) -> bpy.types.PoseBone | None:
    pose_bones = rig_obj.pose.bones
    for name in candidates:
        bone = pose_bones.get(name)
        if bone:
            return bone
    lowered = {bone.name.lower(): bone for bone in pose_bones}
    for name in candidates:
        match = lowered.get(name.lower())
        if match:
            return match
    for name in candidates:
        token = name.lower()
        for bone in pose_bones:
            if token in bone.name.lower():
                return bone
    return None


def resolve_rigify_controls(rig_obj: bpy.types.Object) -> dict[str, bpy.types.PoseBone]:
    mapping: dict[str, bpy.types.PoseBone] = {}
    candidates = {
        "root": ["CTRL-root", "root", "CTRL-hips", "hips", "pelvis", "Torso", "torso"],
        "spine": ["CTRL-spine", "CTRL-spine.001", "spine"],
        "chest": ["CTRL-chest", "CTRL-spine.002", "chest"],
        "head": ["CTRL-head", "head"],
        "hand_ik.L": ["CTRL-hand_ik.L", "hand_ik.L", "CTRL-hand.L", "hand.L"],
        "hand_ik.R": ["CTRL-hand_ik.R", "hand_ik.R", "CTRL-hand.R", "hand.R"],
        "foot_ik.L": ["CTRL-foot_ik.L", "foot_ik.L", "IK.L", "ik.L", "CTRL-foot.L", "foot.L"],
        "foot_ik.R": ["CTRL-foot_ik.R", "foot_ik.R", "IK.R", "ik.R", "CTRL-foot.R", "foot.R"],
        "pole.L": ["knee_target.L", "PullTarget.L", "pole.L"],
        "pole.R": ["knee_target.R", "PullTarget.R", "pole.R"],
    }
    for key, names in candidates.items():
        bone = find_pose_bone(rig_obj, names)
        if bone:
            mapping[key] = bone
    return mapping


def mesh_info(obj: bpy.types.Object) -> dict:
    mesh = obj.data
    vertex_count = len(mesh.vertices)
    vertex_group_count = len(obj.vertex_groups)
    weighted_vertex_count = 0
    weight_assignment_count = 0
    group_stats: dict[int, dict[str, float | int]] = {}
    if vertex_group_count:
        for vertex in mesh.vertices:
            if vertex.groups:
                weighted_vertex_count += 1
                weight_assignment_count += len(vertex.groups)
                for group_item in vertex.groups:
                    group_index = group_item.group
                    weight_value = float(group_item.weight)
                    stats = group_stats.get(group_index)
                    if stats is None:
                        stats = {"vertex_count": 0, "weight_total": 0.0, "weight_max": 0.0}
                        group_stats[group_index] = stats
                    stats["vertex_count"] = int(stats["vertex_count"]) + 1
                    stats["weight_total"] = float(stats["weight_total"]) + weight_value
                    stats["weight_max"] = max(float(stats["weight_max"]), weight_value)

    group_entries = []
    for index, group in enumerate(obj.vertex_groups):
        stats = group_stats.get(index, {"vertex_count": 0, "weight_total": 0.0, "weight_max": 0.0})
        group_entries.append(
            {
                "name": group.name,
                "index": index,
                "vertex_count": int(stats["vertex_count"]),
                "weight_total": float(stats["weight_total"]),
                "weight_max": float(stats["weight_max"]),
            }
        )

    group_entries.sort(key=lambda item: item["vertex_count"], reverse=True)

    armature_modifiers = []
    for mod in obj.modifiers:
        if mod.type == "ARMATURE":
            armature_modifiers.append(
                {
                    "name": mod.name,
                    "object": mod.object.name if mod.object else None,
                    "use_vertex_groups": getattr(mod, "use_vertex_groups", None),
                    "use_bone_envelopes": getattr(mod, "use_bone_envelopes", None),
                    "use_deform_preserve_volume": getattr(
                        mod, "use_deform_preserve_volume", None
                    ),
                }
            )

    modifier_stack = [
        {"name": mod.name, "type": mod.type} for mod in obj.modifiers
    ]

    parent_armature = None
    if obj.parent and obj.parent.type == "ARMATURE":
        parent_armature = obj.parent.name

    shape_key_names: list[str] = []
    if mesh.shape_keys and mesh.shape_keys.key_blocks:
        shape_key_names = [key.name for key in mesh.shape_keys.key_blocks]

    material_names = [slot.material.name for slot in obj.material_slots if slot.material]

    transform = {
        "location": [obj.location.x, obj.location.y, obj.location.z],
        "rotation_euler": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
        "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
    }

    bounds = get_object_bounds(obj)
    return {
        "name": obj.name,
        "data_name": mesh.name,
        "vertex_count": vertex_count,
        "vertex_group_count": vertex_group_count,
        "vertex_group_names": [group.name for group in obj.vertex_groups],
        "vertex_group_stats": group_entries,
        "weighted_vertex_count": weighted_vertex_count,
        "weight_assignment_count": weight_assignment_count,
        "has_armature_modifier": bool(armature_modifiers),
        "armature_modifiers": armature_modifiers,
        "modifier_stack": modifier_stack,
        "parent_armature": parent_armature,
        "parent_type": obj.parent.type if obj.parent else None,
        "parent_bone": obj.parent_bone if obj.parent else None,
        "shape_keys": shape_key_names,
        "materials": material_names,
        "transform": transform,
        "bounds": bounds_to_dict(bounds),
    }


def armature_info(obj: bpy.types.Object) -> dict:
    bone_names = [bone.name for bone in obj.data.bones]
    bone_hierarchy = []
    for bone in obj.data.bones:
        bone_hierarchy.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "children": [child.name for child in bone.children],
                "head": [bone.head_local.x, bone.head_local.y, bone.head_local.z],
                "tail": [bone.tail_local.x, bone.tail_local.y, bone.tail_local.z],
                "length": float(bone.length),
                "use_deform": bool(getattr(bone, "use_deform", False)),
            }
        )

    constraint_entries = []
    for pose_bone in obj.pose.bones:
        if not pose_bone.constraints:
            continue
        constraints = []
        for constraint in pose_bone.constraints:
            target = getattr(constraint, "target", None)
            constraints.append(
                {
                    "name": constraint.name,
                    "type": constraint.type,
                    "target": target.name if target else None,
                    "subtarget": getattr(constraint, "subtarget", None),
                    "influence": getattr(constraint, "influence", None),
                    "owner_space": getattr(constraint, "owner_space", None),
                    "target_space": getattr(constraint, "target_space", None),
                }
            )
        constraint_entries.append({"bone": pose_bone.name, "constraints": constraints})
    prefix_counts = {
        "CTRL": count_prefix(bone_names, "CTRL-"),
        "DEF": count_prefix(bone_names, "DEF-"),
        "ORG": count_prefix(bone_names, "ORG-"),
        "MCH": count_prefix(bone_names, "MCH-"),
    }
    side_counts = {
        "L": count_side(bone_names, "L"),
        "R": count_side(bone_names, "R"),
    }
    ik_count = count_contains(bone_names, "_ik") + count_contains(bone_names, "ik.")
    fk_count = count_contains(bone_names, "_fk") + count_contains(bone_names, "fk.")

    rigify_score = 0
    rigify_score += 2 if prefix_counts["DEF"] else 0
    rigify_score += 2 if prefix_counts["ORG"] else 0
    rigify_score += 1 if prefix_counts["MCH"] else 0
    rigify_score += 1 if prefix_counts["CTRL"] else 0
    if "rig" in obj.name.lower():
        rigify_score += 1

    rig_id = None
    try:
        rig_id = obj.data.get("rig_id")
    except Exception:
        rig_id = None

    controls = resolve_rigify_controls(obj)
    matched = {key: bone.name for key, bone in controls.items()}
    missing = [
        key
        for key in [
            "root",
            "spine",
            "chest",
            "head",
            "hand_ik.L",
            "hand_ik.R",
            "foot_ik.L",
            "foot_ik.R",
            "pole.L",
            "pole.R",
        ]
        if key not in controls
    ]

    bounds = get_object_bounds(obj)
    transform = {
        "location": [obj.location.x, obj.location.y, obj.location.z],
        "rotation_euler": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
        "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
    }

    bone_collections = []
    armature_data = obj.data
    if hasattr(armature_data, "collections"):
        for collection in armature_data.collections:
            bone_collections.append(
                {
                    "name": collection.name,
                    "bone_count": len(collection.bones),
                    "bones": [bone.name for bone in collection.bones],
                }
            )
    return {
        "name": obj.name,
        "bone_count": len(bone_names),
        "bone_names": bone_names,
        "bone_hierarchy": bone_hierarchy,
        "pose_constraints": constraint_entries,
        "prefix_counts": prefix_counts,
        "side_counts": side_counts,
        "ik_count": ik_count,
        "fk_count": fk_count,
        "rigify_score": rigify_score,
        "rigify_like": rigify_score >= 3 or bool(rig_id),
        "rig_id": rig_id,
        "controls_matched": matched,
        "controls_missing": missing,
        "bone_collections": bone_collections,
        "transform": transform,
        "bounds": bounds_to_dict(bounds),
    }


def score_mesh(entry: dict) -> int:
    score = 0
    if entry.get("parent_armature"):
        score += 100
    if entry.get("has_armature_modifier"):
        score += 80
    score += min(entry.get("vertex_group_count", 0), 50)
    score += min(entry.get("weighted_vertex_count", 0) // 100, 50)
    score += min(entry.get("vertex_count", 0) // 1000, 30)
    return score


def pick_recommended_mesh(mesh_entries: list[dict], mesh_name: str | None) -> dict | None:
    if not mesh_entries:
        return None
    if mesh_name:
        for entry in mesh_entries:
            if entry["name"] == mesh_name:
                return entry
    return max(mesh_entries, key=score_mesh)


def pick_linked_armature(mesh_entry: dict, armature_entries: dict[str, dict]) -> dict | None:
    if not mesh_entry:
        return None
    parent_armature = mesh_entry.get("parent_armature")
    if parent_armature and parent_armature in armature_entries:
        return armature_entries[parent_armature]

    for mod in mesh_entry.get("armature_modifiers") or []:
        target = mod.get("object")
        if target and target in armature_entries:
            return armature_entries[target]
    return None


def summarize(report: dict) -> None:
    print("Rig Diagnose Summary")
    print("=" * 72)
    print(f"Blend: {report.get('blend_path')}")
    print(f"Meshes: {len(report.get('meshes', []))}")
    print(f"Armatures: {len(report.get('armatures', []))}")
    recommended = report.get("recommended") or {}
    if recommended.get("mesh"):
        print(f"Recommended mesh: {recommended['mesh']}")
    if recommended.get("armature"):
        print(f"Linked armature: {recommended['armature']}")
    if recommended.get("controls_missing"):
        missing = ", ".join(recommended["controls_missing"])
        print(f"Missing controls: {missing}")
    if report.get("notes"):
        for note in report["notes"]:
            print(f"Note: {note}")
    print("=" * 72)


def analyze(blend_path: Path, mesh_name: str | None) -> dict:
    mesh_objs = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    armature_objs = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]

    mesh_entries = [mesh_info(obj) for obj in mesh_objs]
    armature_entries = {obj.name: armature_info(obj) for obj in armature_objs}

    recommended_mesh = pick_recommended_mesh(mesh_entries, mesh_name)
    linked_armature = None
    if recommended_mesh:
        linked_armature = pick_linked_armature(recommended_mesh, armature_entries)

    if linked_armature is None and armature_entries:
        linked_armature = max(
            armature_entries.values(),
            key=lambda item: item.get("rigify_score", 0),
        )

    notes: list[str] = []
    if recommended_mesh:
        if recommended_mesh.get("vertex_group_count", 0) > 0:
            notes.append("Mesh has vertex groups; existing weights may be present.")
        if recommended_mesh.get("has_armature_modifier") or recommended_mesh.get(
            "parent_armature"
        ):
            notes.append("Mesh is already linked to an armature.")
        if not recommended_mesh.get("has_armature_modifier") and not recommended_mesh.get(
            "parent_armature"
        ):
            notes.append("Mesh is not linked to an armature.")

    if linked_armature:
        missing = linked_armature.get("controls_missing") or []
        if missing:
            notes.append("Rigify controls are missing or named differently.")
    else:
        notes.append("No armature detected in the file.")

    recommended = {
        "mesh": recommended_mesh.get("name") if recommended_mesh else None,
        "armature": linked_armature.get("name") if linked_armature else None,
        "controls_missing": linked_armature.get("controls_missing") if linked_armature else None,
        "model_object_hint": recommended_mesh.get("name") if recommended_mesh else None,
    }

    if recommended_mesh and linked_armature:
        mesh_bounds = recommended_mesh.get("bounds")
        arm_bounds = linked_armature.get("bounds")
        if mesh_bounds and arm_bounds:
            mesh_height = mesh_bounds["size"][2]
            rig_height = arm_bounds["size"][2]
            if rig_height and rig_height > 0:
                recommended["mesh_to_rig_height_ratio"] = mesh_height / rig_height

    return {
        "blend_path": str(blend_path),
        "meshes": mesh_entries,
        "armatures": list(armature_entries.values()),
        "recommended": recommended,
        "notes": notes,
    }


def main() -> int:
    args = get_script_args()
    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        print(f"ERROR: blend file not found: {blend_path}", file=sys.stderr)
        return 1

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    report = analyze(blend_path, args.mesh)
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else blend_path.with_name(f"{blend_path.stem}_rig_report.json")
    )
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    summarize(report)
    print(f"Report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
