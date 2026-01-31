from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="od2blender importer")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=True, help="Output .blend path")
    parser.add_argument("--plane-width", type=float, default=2.0)
    parser.add_argument("--z-depth", type=float, default=0.01)
    return parser.parse_args(argv)


def blender_relpath(scene_path: Path, media_path: Path) -> str:
    try:
        rel = bpy.path.relpath(str(media_path), start=str(scene_path.parent))
    except Exception:
        return str(media_path)
    return rel


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def set_viewport_shading() -> None:
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"


def to_z_up_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Rotate +90deg around X so Y-up becomes Z-up."""
    return (x, -z, y)


def get_or_create_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection:
        return collection
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def configure_scene(width: int, height: int, fps: float, frame_count: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    fps_int = int(round(fps)) if fps else 24
    fps_int = max(fps_int, 1)
    scene.render.fps = fps_int
    scene.render.fps_base = fps_int / fps if fps else 1.0

    scene.frame_start = 1
    scene.frame_end = max(frame_count, 1)
    scene.frame_set(1)


def compute_pose_bounds(
    frames: list[dict],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")
    found = False
    for frame in frames:
        poses = frame.get("poses3d") or []
        if not poses:
            continue
        pose = poses[0] if isinstance(poses[0], dict) else None
        if not pose:
            continue
        landmarks = pose.get("landmarks") or []
        for coords in landmarks:
            if not isinstance(coords, (list, tuple)) or len(coords) < 3:
                continue
            try:
                x = float(coords[0])
                y = float(coords[1])
                z = float(coords[2])
            except (TypeError, ValueError):
                continue
            bx, by, bz = to_z_up_coords(x, y, z)
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            min_z = min(min_z, bz)
            max_x = max(max_x, bx)
            max_y = max(max_y, by)
            max_z = max(max_z, bz)
            found = True
    if not found:
        return None
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def create_camera(name: str, collection: bpy.types.Collection) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(name)
    camera_obj = bpy.data.objects.new(name, camera_data)
    collection.objects.link(camera_obj)
    return camera_obj


def configure_camera_for_bounds(
    camera_obj: bpy.types.Object,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    scene: bpy.types.Scene,
    margin_ratio: float = 0.1,
    min_ortho_scale: float = 0.1,
) -> None:
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = max_z - min_z
    aspect = (
        scene.render.resolution_y / scene.render.resolution_x
        if scene.render.resolution_x
        else 1.0
    )
    ortho_scale = max(span_x, span_z / aspect) if aspect else max(span_x, span_z)
    ortho_scale = max(ortho_scale, min_ortho_scale) * (1.0 + margin_ratio)

    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    depth_margin = max(span_y * 0.5, ortho_scale * 0.1, 0.1)
    camera_y = min_y - depth_margin

    camera = camera_obj.data
    camera.type = "ORTHO"
    camera.ortho_scale = ortho_scale
    camera.clip_start = 0.001
    camera.clip_end = max((max_y - camera_y) + depth_margin, camera.clip_start * 10.0)

    camera_obj.location = (center_x, camera_y, center_z)
    camera_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def configure_camera_for_video_plane(
    camera_obj: bpy.types.Object,
    plane_width: float,
    plane_height: float,
) -> None:
    camera = camera_obj.data
    camera.type = "ORTHO"
    camera.ortho_scale = max(plane_width, 0.1) * 1.05
    camera.clip_start = 0.001
    camera.clip_end = max(plane_width, plane_height, 1.0) * 10.0
    camera_obj.location = (0.0, -max(plane_width, plane_height, 1.0), 0.0)
    camera_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def create_video_plane(
    name: str,
    plane_width: float,
    plane_height: float,
    video_path: Path,
    scene_path: Path,
    frame_count: int,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    half_w = plane_width / 2.0
    half_h = plane_height / 2.0
    verts = [
        (-half_w, -half_h, 0.0),
        (half_w, -half_h, 0.0),
        (half_w, half_h, 0.0),
        (-half_w, half_h, 0.0),
    ]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    if uv_layer:
        uv_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for poly in mesh.polygons:
            for loop_index, uv in zip(poly.loop_indices, uv_coords):
                uv_layer.data[loop_index].uv = uv

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    material = bpy.data.materials.new("OD_Video_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    tex = nodes.new("ShaderNodeTexImage")
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")

    image = bpy.data.images.load(str(video_path))
    image.source = "MOVIE"
    tex.image = image
    tex.image_user.use_auto_refresh = True
    tex.image_user.frame_start = 1
    tex.image_user.frame_duration = frame_count
    tex.image_user.use_cyclic = False

    rel_path = blender_relpath(scene_path, video_path)
    image.filepath = rel_path
    image.filepath_raw = rel_path

    emission.inputs["Strength"].default_value = 1.0
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    return obj


def orient_video_plane(obj: bpy.types.Object) -> None:
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def add_video_strip_to_vse(video_path: Path) -> None:
    scene = bpy.context.scene
    scene.sequence_editor_create()
    scene.sequence_editor.sequences.new_movie(
        "OD_VideoStrip",
        str(video_path),
        channel=1,
        frame_start=1,
    )


def get_or_create_empty(
    track_id: int,
    collection: bpy.types.Collection,
    display_size: float,
) -> bpy.types.Object:
    name = f"track_{track_id:04d}"
    existing = bpy.data.objects.get(name)
    if existing:
        return existing
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = display_size
    collection.objects.link(obj)
    return obj


def get_pose_skeleton(tracks: dict) -> tuple[list[str], list[tuple[int, int]]]:
    skeleton = tracks.get("skeleton") or {}
    names = [str(name) for name in skeleton.get("landmarks") or []]
    bones = []
    for item in skeleton.get("bones") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = item
        bones.append((int(start), int(end)))
    return names, bones


def create_pose_empties(
    names: list[str],
    collection: bpy.types.Collection,
    display_size: float,
    prefix: str = "pose",
) -> list[bpy.types.Object]:
    empties: list[bpy.types.Object] = []
    for name in names:
        obj = bpy.data.objects.new(f"{prefix}_{name}", None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = display_size
        collection.objects.link(obj)
        empties.append(obj)
    return empties


def set_pose_empties(empties: list[bpy.types.Object], landmarks: list) -> None:
    for idx, coords in enumerate(landmarks):
        if idx >= len(empties):
            break
        x, y, z = coords[0], coords[1], coords[2]
        empties[idx].location = to_z_up_coords(x, y, z)


def find_first_pose_landmarks(frames: list[dict]) -> list | None:
    for frame in frames:
        poses = frame.get("poses3d") or []
        if not poses:
            continue
        landmarks = poses[0].get("landmarks") if isinstance(poses[0], dict) else None
        if landmarks:
            return landmarks
    return None


def create_pose_armature(
    names: list[str],
    bones: list[tuple[int, int]],
    landmarks: list,
    empties: list[bpy.types.Object],
    collection: bpy.types.Collection,
) -> bpy.types.Object | None:
    if not bones or not landmarks:
        return None

    armature = bpy.data.armatures.new("OD_PoseArmature")
    armature_obj = bpy.data.objects.new("OD_PoseArmature", armature)
    armature.display_type = "STICK"
    armature_obj.show_in_front = True
    collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    bone_map: dict[str, tuple[int, int]] = {}
    for start, end in bones:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        head = to_z_up_coords(*landmarks[start][:3])
        tail = to_z_up_coords(*landmarks[end][:3])
        if start < len(names) and end < len(names):
            name = f"{names[start]}_{names[end]}"
        else:
            name = f"bone_{start}_{end}"
        edit_bone = armature.edit_bones.new(name)
        edit_bone.head = head
        edit_bone.tail = tail
        bone_map[name] = (start, end)

    bpy.ops.object.mode_set(mode="POSE")
    for bone_name, (start, end) in bone_map.items():
        pose_bone = armature_obj.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        copy_loc = pose_bone.constraints.new("COPY_LOCATION")
        copy_loc.target = empties[start]
        copy_loc.owner_space = "WORLD"
        copy_loc.target_space = "WORLD"

        stretch = pose_bone.constraints.new("STRETCH_TO")
        stretch.target = empties[end]
        stretch.owner_space = "WORLD"
        stretch.target_space = "WORLD"

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_obj


def animate_pose_empties(frames: list[dict], empties: list[bpy.types.Object]) -> None:
    for frame in frames:
        poses = frame.get("poses3d") or []
        if not poses:
            continue
        pose = poses[0] if isinstance(poses[0], dict) else None
        if not pose:
            continue
        landmarks = pose.get("landmarks") or []
        blender_frame = int(frame.get("i", 0)) + 1
        for idx, coords in enumerate(landmarks):
            if idx >= len(empties):
                break
            x, y, z = coords[0], coords[1], coords[2]
            empties[idx].location = to_z_up_coords(x, y, z)
            empties[idx].keyframe_insert(data_path="location", frame=blender_frame)


def main() -> int:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = parse_args(argv)
    tracks_path = Path(args.tracks)
    video_path = Path(args.video)
    scene_path = Path(args.out)

    with tracks_path.open("r", encoding="utf-8") as handle:
        tracks = json.load(handle)

    video = tracks.get("video") or {}
    frames = tracks.get("frames") or []

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    fps = float(video.get("fps", 0.0))
    frame_count = int(video.get("frame_count") or len(frames))

    plane_width = float(args.plane_width)
    plane_height = plane_width * (height / width) if width else plane_width
    z_depth = float(args.z_depth)

    clear_scene()
    configure_scene(width, height, fps, frame_count)

    video_collection = get_or_create_collection("OD_Video")
    tracks_collection = get_or_create_collection("OD_Tracks")

    video_plane = create_video_plane(
        name="OD_VideoPlane",
        plane_width=plane_width,
        plane_height=plane_height,
        video_path=video_path,
        scene_path=scene_path,
        frame_count=frame_count,
        collection=video_collection,
    )
    orient_video_plane(video_plane)

    camera_collection = get_or_create_collection("OD_Camera")
    camera_obj = create_camera("OD_Camera", camera_collection)
    pose_bounds = compute_pose_bounds(frames)
    if pose_bounds:
        configure_camera_for_bounds(camera_obj, pose_bounds, bpy.context.scene)
    else:
        configure_camera_for_video_plane(camera_obj, plane_width, plane_height)
    bpy.context.scene.camera = camera_obj

    set_viewport_shading()

    display_size = plane_width * 0.02
    for frame in frames:
        frame_index = int(frame.get("i", 0))
        blender_frame = frame_index + 1
        for det in frame.get("detections") or []:
            track_id = int(det["id"])
            u, v = det["center_norm"]
            x = (u - 0.5) * plane_width
            y = (0.5 - v) * plane_height
            obj = get_or_create_empty(track_id, tracks_collection, display_size)
            obj.location = to_z_up_coords(x, y, z_depth)
            obj.keyframe_insert(data_path="location", frame=blender_frame)

    pose_names, pose_bones = get_pose_skeleton(tracks)
    if pose_names:
        pose_collection = get_or_create_collection("OD_Pose3D")
        pose_display_size = plane_width * 0.01
        pose_empties = create_pose_empties(
            pose_names,
            pose_collection,
            pose_display_size,
            prefix="pose",
        )
        initial_landmarks = find_first_pose_landmarks(frames)
        if initial_landmarks:
            set_pose_empties(pose_empties, initial_landmarks)
            create_pose_armature(
                pose_names,
                pose_bones,
                initial_landmarks,
                pose_empties,
                pose_collection,
            )
        animate_pose_empties(frames, pose_empties)

    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
