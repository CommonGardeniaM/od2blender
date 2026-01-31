"""Extract bone names from a Blender .blend file without Blender installed.

This script searches for bone names in .blend files using multiple methods.
"""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path


def is_likely_bone(name: str) -> bool:
    """Check if a name is likely to be a bone name based on heuristics."""
    if len(name) < 2 or len(name) > 40:
        return False
    
    # Skip common false positives
    false_positives = [
        'RuntimeHandle', 'Runtime', 'FileSpaceData', 'FileHandler',
        'UserDef_', 'GpencilModifierData', 'ModifierData', 'Constraint',
        'NodeTree', 'NodeSocket', 'Geometry', 'Parameter', 'Instance',
        'Membership', 'ActionSlot', 'CacheArchive', 'Viewport'
    ]
    
    for fp in false_positives:
        if fp in name:
            return False
    
    # Bone name indicators
    bone_keywords = [
        'foot', 'thigh', 'shin', 'leg', 'arm', 'hand', 'head', 'neck',
        'spine', 'pelvis', 'hip', 'knee', 'elbow', 'wrist', 'shoulder',
        'root', 'torso', 'toe', 'heel', 'ik', 'fk', 'target', 'pole',
        'ctrl', 'mch', 'org', 'def', 'upper', 'lower', 'fore', 'pull'
    ]
    
    name_lower = name.lower()
    
    # Must contain at least one bone keyword OR have .L/.R suffix
    has_keyword = any(kw in name_lower for kw in bone_keywords)
    has_side_suffix = '.L' in name or '.R' in name or '_L' in name or '_R' in name
    
    return has_keyword or has_side_suffix


def extract_all_names(blend_path: Path) -> set[str]:
    """Extract all potential names from the blend file."""
    names: set[str] = set()
    
    # Try to open as gzip first (newer .blend files), then raw
    try:
        with gzip.open(blend_path, 'rb') as f:
            data = f.read()
    except (gzip.BadGzipFile, OSError):
        with open(blend_path, 'rb') as f:
            data = f.read()
    
    # Pattern 1: BO + bone name (standard Blender bone naming)
    bo_pattern = rb'BO([a-zA-Z_][a-zA-Z0-9_.\-]{1,61})\x00'
    for match in re.findall(bo_pattern, data):
        try:
            name = match.decode('utf-8', errors='ignore')
            if name:
                names.add(name)
        except UnicodeDecodeError:
            continue
    
    # Pattern 2: PB + pose bone name
    pb_pattern = rb'PB([a-zA-Z_][a-zA-Z0-9_.\-]{1,61})\x00'
    for match in re.findall(pb_pattern, data):
        try:
            name = match.decode('utf-8', errors='ignore')
            if name:
                names.add(name)
        except UnicodeDecodeError:
            continue
    
    # Pattern 3: Names that look like Rigify bones (with .L/.R)
    lr_pattern = rb'(?<![a-zA-Z0-9_.\-])([a-zA-Z_][a-zA-Z0-9_]*\.[LR])\x00'
    for match in re.findall(lr_pattern, data):
        try:
            name = match.decode('utf-8', errors='ignore')
            if name and is_likely_bone(name):
                names.add(name)
        except UnicodeDecodeError:
            continue
    
    # Pattern 4: MCH-, ORG-, DEF- prefixed names
    prefix_pattern = rb'(?<![a-zA-Z0-9_.\-])((?:MCH|ORG|DEF|CTRL)-[a-zA-Z0-9_.\-]+)\x00'
    for match in re.findall(prefix_pattern, data):
        try:
            name = match.decode('utf-8', errors='ignore')
            if name:
                names.add(name)
        except UnicodeDecodeError:
            continue
    
    # Pattern 5: Look for names followed by null bytes (DNA strings)
    # This catches most structure names but we'll filter them
    general_pattern = rb'(?<![a-zA-Z0-9_.\-])([a-zA-Z][a-zA-Z0-9_]{2,30})\x00'
    for match in re.findall(general_pattern, data):
        try:
            name = match.decode('utf-8', errors='ignore')
            if name and is_likely_bone(name):
                names.add(name)
        except UnicodeDecodeError:
            continue
    
    return names


def categorize_bones(bone_names: list[str]) -> dict[str, list[str]]:
    """Categorize bones by type."""
    categories: dict[str, list[str]] = {
        'Left Side (.L)': [],
        'Right Side (.R)': [],
        'IK Bones': [],
        'FK Bones': [],
        'MCH (Mechanism)': [],
        'ORG (Original)': [],
        'DEF (Deform)': [],
        'CTRL (Controls)': [],
        'Target Bones': [],
        'Root/Torso/Main': [],
        'Leg Bones': [],
        'Arm Bones': [],
        'Spine/Head': [],
        'Other': [],
    }
    
    for name in bone_names:
        name_lower = name.lower()
        assigned = False
        
        # Check prefixes first
        if name.startswith('MCH-'):
            categories['MCH (Mechanism)'].append(name)
            assigned = True
        elif name.startswith('ORG-'):
            categories['ORG (Original)'].append(name)
            assigned = True
        elif name.startswith('DEF-'):
            categories['DEF (Deform)'].append(name)
            assigned = True
        elif name.startswith('CTRL-') or name.startswith('Ctrl'):
            categories['CTRL (Controls)'].append(name)
            assigned = True
        
        # Check suffixes
        if '.L' in name or '_L' in name:
            categories['Left Side (.L)'].append(name)
        elif '.R' in name or '_R' in name:
            categories['Right Side (.R)'].append(name)
        
        # Check IK/FK
        if 'IK' in name or '_ik' in name_lower:
            categories['IK Bones'].append(name)
        elif 'FK' in name or '_fk' in name_lower:
            categories['FK Bones'].append(name)
        
        # Check targets
        if 'target' in name_lower:
            categories['Target Bones'].append(name)
        
        # Check body parts
        if any(x in name_lower for x in ['foot', 'thigh', 'shin', 'leg', 'toe', 'heel', 'knee']):
            categories['Leg Bones'].append(name)
        elif any(x in name_lower for x in ['arm', 'hand', 'shoulder', 'wrist', 'elbow', 'forearm']):
            categories['Arm Bones'].append(name)
        elif any(x in name_lower for x in ['spine', 'neck', 'head', 'pelvis', 'torso']):
            categories['Spine/Head'].append(name)
        
        # Check root
        if 'root' in name_lower:
            categories['Root/Torso/Main'].append(name)
        
        # If not assigned to a prefix category, add to Other
        if not assigned:
            if name not in (categories['Left Side (.L)'] + categories['Right Side (.R)'] +
                          categories['IK Bones'] + categories['FK Bones'] +
                          categories['Target Bones'] + categories['Leg Bones'] +
                          categories['Arm Bones'] + categories['Spine/Head'] +
                          categories['Root/Torso/Main']):
                categories['Other'].append(name)
    
    # Remove duplicates and sort
    for key in categories:
        categories[key] = sorted(set(categories[key]))
    
    return categories


def print_results(bone_names: list[str]) -> None:
    """Print the bone names in a nice format."""
    print(f"\nFound {len(bone_names)} bone(s):\n")
    
    # Print all bones
    for name in sorted(bone_names):
        print(f"  {name}")
    
    # Print categories
    categories = categorize_bones(bone_names)
    
    print("\n" + "=" * 70)
    print("BONES BY CATEGORY")
    print("=" * 70)
    
    for category, bones in categories.items():
        if bones:
            print(f"\n{category} ({len(bones)}):")
            for name in bones:
                print(f"  {name}")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        blend_path = Path(r"C:\Users\commo\Desktop\dev\python\od2blender\Male Lowpoly Mesh.blend")
    else:
        blend_path = Path(sys.argv[1])
    
    if not blend_path.exists():
        print(f"Error: File not found: {blend_path}", file=sys.stderr)
        return 1
    
    print(f"Analyzing: {blend_path}")
    print("=" * 70)
    
    try:
        names = extract_all_names(blend_path)
        bone_names = [n for n in names if is_likely_bone(n)]
        
        if bone_names:
            print_results(bone_names)
        else:
            print("  No bones found")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results to file
    output_path = blend_path.parent / f"{blend_path.stem}_bones.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Bone names from: {blend_path}\n")
        f.write(f"Total bones found: {len(bone_names)}\n")
        f.write("=" * 70 + "\n\n")
        
        categories = categorize_bones(bone_names)
        for category, bones in categories.items():
            if bones:
                f.write(f"\n{category} ({len(bones)}):\n")
                for name in sorted(bones):
                    f.write(f"  {name}\n")
        
        f.write("\n\nAll bones (alphabetical):\n")
        for name in sorted(bone_names):
            f.write(f"{name}\n")
    
    print(f"\n\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
