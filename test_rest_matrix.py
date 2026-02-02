"""Debug script to inspect PMX rest matrices and initial pose data."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mmd_trace.io.pmx import load_pmx
import numpy as np


def print_bone_hierarchy(model, bone_index: int, indent: int = 0) -> None:
    """Print bone hierarchy tree."""
    bone = model.get_bone_by_index(bone_index)
    if bone is None:
        return
    
    prefix = "  " * indent
    parent_str = f" (parent: {bone.parent_index})" if bone.parent_index >= 0 else " (root)"
    print(f"{prefix}[{bone.index}] {bone.name}{parent_str}")
    
    # Print rest direction
    rest_dir = model.get_bone_rest_direction(bone_index)
    print(f"{prefix}  Rest Direction: [{rest_dir[0]:.4f}, {rest_dir[1]:.4f}, {rest_dir[2]:.4f}]")
    
    # Print position
    print(f"{prefix}  Position: [{bone.position[0]:.4f}, {bone.position[1]:.4f}, {bone.position[2]:.4f}]")
    
    # Print flags info
    flags_info = []
    if bone.is_rotatable:
        flags_info.append("rot")
    if bone.is_translatable:
        flags_info.append("trans")
    if bone.has_local_coord:
        flags_info.append(f"local({bone.local_x_vector is not None})")
    if bone.has_fixed_axis:
        flags_info.append(f"fixed({bone.fixed_axis is not None})")
    if bone.has_ik:
        flags_info.append(f"ik(target={bone.ik.target_index if bone.ik else 'none'})")
    if flags_info:
        print(f"{prefix}  Flags: {', '.join(flags_info)}")
    
    # Print rest matrix (first few columns)
    rest_matrix = model.get_bone_rest_matrix(bone_index)
    print(f"{prefix}  Rest Matrix:")
    for i in range(3):
        print(f"{prefix}    [{rest_matrix[i, 0]:+.4f}, {rest_matrix[i, 1]:+.4f}, {rest_matrix[i, 2]:+.4f}]")
    
    # If has parent, show local basis
    if bone.parent_index >= 0:
        local_basis = model.compute_bone_local_basis(bone_index)
        print(f"{prefix}  Local Basis (relative to parent):")
        for i in range(3):
            print(f"{prefix}    [{local_basis[i, 0]:+.4f}, {local_basis[i, 1]:+.4f}, {local_basis[i, 2]:+.4f}]")
    
    print()
    
    # Recursively print children
    children = model.get_bone_children(bone_index)
    for child in children:
        print_bone_hierarchy(model, child.index, indent + 1)


def main() -> int:
    """Main entry point."""
    pmx_path = Path("Millial_forMMD_v1.0.0.pmx")
    if not pmx_path.exists():
        print(f"Error: PMX file not found: {pmx_path}")
        return 1
    
    print(f"Loading PMX: {pmx_path}")
    print("=" * 70)
    
    model = load_pmx(str(pmx_path))
    
    print(f"Model: {model.model_name}")
    print(f"Bones: {model.bone_count}")
    print(f"Vertices: {model.vertex_count}")
    print()
    
    # Print all root bones and their hierarchies
    print("BONE HIERARCHY (Root bones and their children):")
    print("=" * 70)
    root_bones = [bone for bone in model.bones if bone.parent_index < 0]
    for root in root_bones:
        print_bone_hierarchy(model, root.index)
    
    # Print specific bones of interest
    print()
    print("SPECIFIC BONES OF INTEREST:")
    print("=" * 70)
    
    interesting_bones = [
        "左腕", "左ひじ", "左手首",
        "右腕", "右ひじ", "右手首",
        "左足", "左ひざ", "左足首",
        "右足", "右ひざ", "右足首",
        "下半身", "上半身", "上半身2",
    ]
    
    for bone_name in interesting_bones:
        bone = model.get_bone(bone_name)
        if bone:
            print(f"\n{bone_name}:")
            print(f"  Index: {bone.index}")
            print(f"  Parent: {bone.parent_index}")
            
            # Get chain
            chain = model.get_parent_chain(bone.index)
            chain_names = [b.name for b in chain]
            print(f"  Parent Chain: {' -> '.join(chain_names) if chain_names else 'root'}")
            
            # Rest direction
            rest_dir = model.get_bone_rest_direction(bone.index)
            print(f"  Rest Direction: [{rest_dir[0]:.4f}, {rest_dir[1]:.4f}, {rest_dir[2]:.4f}]")
            
            # Position
            print(f"  Position: [{bone.position[0]:.4f}, {bone.position[1]:.4f}, {bone.position[2]:.4f}]")
            
            # Rest matrix
            rest_matrix = model.get_bone_rest_matrix(bone.index)
            print(f"  Rest Matrix:")
            for i in range(3):
                print(f"    [{rest_matrix[i, 0]:+.4f}, {rest_matrix[i, 1]:+.4f}, {rest_matrix[i, 2]:+.4f}]")
            
            # Local basis
            local_basis = model.compute_bone_local_basis(bone.index)
            print(f"  Local Basis:")
            for i in range(3):
                print(f"    [{local_basis[i, 0]:+.4f}, {local_basis[i, 1]:+.4f}, {local_basis[i, 2]:+.4f}]")
            
            # Additional info
            if bone.has_local_coord:
                print(f"  Has Local Coord: X={bone.local_x_vector is not None}, Z={bone.local_z_vector is not None}")
            if bone.has_fixed_axis:
                print(f"  Has Fixed Axis: {bone.fixed_axis}")
            if bone.has_ik and bone.ik:
                print(f"  IK: target={bone.ik.target_index}, links={len(bone.ik.links)}")
    
    # Validation
    print()
    print("VALIDATION:")
    print("=" * 70)
    issues = model.validate()
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("Validation: PASSED - No issues found!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
