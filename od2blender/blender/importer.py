from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="od2blender importer")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out", required=True, help="Output .blend path")
    parser.add_argument("--plane-width", type=float, default=2.0)
    parser.add_argument("--z-depth", type=float, default=0.01)
    parser.add_argument("--pose-scale", type=float, default=1.0)
    parser.add_argument(
        "--pose3d-source",
        choices=["projected", "raw"],
        default="projected",
        help="Pose source to drive the rig",
    )
    parser.add_argument("--model", help="Path to a 3D model for Rigify")
    parser.add_argument("--model-object", help="Object name inside the model file")
    parser.add_argument(
        "--test-motion",
        action="store_true",
        help="Generate a test crouch motion for the model",
    )
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


def ensure_object_mode() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def activate_object(obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


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


def normalize_mesh_object(obj: bpy.types.Object) -> None:
    activate_object(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bounds = get_object_bounds(obj)
    if not bounds:
        return
    min_v, max_v = bounds
    center_x = (min_v.x + max_v.x) * 0.5
    center_y = (min_v.y + max_v.y) * 0.5
    obj.location = obj.location - Vector((center_x, center_y, min_v.z))


def hide_object(obj: bpy.types.Object) -> None:
    obj.hide_viewport = True
    obj.hide_render = True


def find_layer_collection(
    layer_collection: bpy.types.LayerCollection,
    name: str,
) -> bpy.types.LayerCollection | None:
    if layer_collection.collection.name == name:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, name)
        if found:
            return found
    return None


def hide_collection(collection: bpy.types.Collection) -> None:
    collection.hide_viewport = True
    collection.hide_render = True
    for view_layer in bpy.context.scene.view_layers:
        layer_collection = find_layer_collection(view_layer.layer_collection, collection.name)
        if layer_collection:
            layer_collection.hide_viewport = True
            layer_collection.exclude = True


def strip_mesh_rigging(mesh_obj: bpy.types.Object) -> None:
    for mod in list(mesh_obj.modifiers):
        if mod.type == "ARMATURE":
            mesh_obj.modifiers.remove(mod)
    if mesh_obj.parent and mesh_obj.parent.type == "ARMATURE":
        mesh_obj.parent = None
        mesh_obj.matrix_parent_inverse = Matrix.Identity(4)


def import_model(model_path: Path, object_name: str | None) -> bpy.types.Object:
    if not model_path.is_file():
        raise RuntimeError(f"Model path not found: {model_path}")
    ext = model_path.suffix.lower()
    loaded_objects: list[bpy.types.Object] = []
    if ext == ".blend":
        with bpy.data.libraries.load(str(model_path), link=False) as (data_from, data_to):
            if object_name:
                if object_name not in data_from.objects:
                    raise RuntimeError(f"Object '{object_name}' not found in {model_path}")
                data_to.objects = [object_name]
            else:
                data_to.objects = list(data_from.objects)
        loaded_objects = [obj for obj in data_to.objects if obj is not None]
        for obj in loaded_objects:
            if obj.name not in bpy.context.scene.objects:
                bpy.context.scene.collection.objects.link(obj)
    else:
        before = {obj.name for obj in bpy.data.objects}
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(model_path))
        elif ext in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(model_path))
        elif ext == ".obj":
            bpy.ops.import_scene.obj(filepath=str(model_path))
        else:
            raise RuntimeError(f"Unsupported model format: {model_path}")
        loaded_objects = [obj for obj in bpy.data.objects if obj.name not in before]

    target: bpy.types.Object | None = None
    if object_name:
        target = next((obj for obj in loaded_objects if obj.name == object_name), None)
        if target is None:
            raise RuntimeError(f"Object '{object_name}' was not imported")
    else:
        target = next((obj for obj in loaded_objects if obj.type == "MESH"), None)
    if target is None:
        raise RuntimeError("No mesh object found in model file")
    for obj in list(loaded_objects):
        if obj is target:
            continue
        if obj.type == "ARMATURE":
            bpy.data.objects.remove(obj, do_unlink=True)
    return target


def get_or_create_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection:
        return collection
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def ensure_rigify_enabled() -> None:
    if "rigify" in bpy.context.preferences.addons:
        return
    try:
        bpy.ops.preferences.addon_enable(module="rigify")
    except Exception:
        pass
    if "rigify" not in bpy.context.preferences.addons:
        raise RuntimeError(
            "Rigify addon is not enabled. Enable it in Edit > Preferences > Add-ons > Rigify"
        )


def add_human_metarig() -> bpy.types.Object:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    try:
        bpy.ops.object.armature_human_metarig_add()
    except Exception:
        bpy.ops.object.armature_basic_human_metarig_add()
    metarig = bpy.context.view_layer.objects.active
    if metarig is None:
        raise RuntimeError("Failed to create Rigify metarig")
    metarig.name = "metarig"
    return metarig


def fit_metarig_to_mesh(metarig: bpy.types.Object, mesh_obj: bpy.types.Object) -> None:
    mesh_bounds = get_object_bounds(mesh_obj)
    rig_bounds = get_object_bounds(metarig)
    if not mesh_bounds or not rig_bounds:
        return
    mesh_min, mesh_max = mesh_bounds
    rig_min, rig_max = rig_bounds
    mesh_height = mesh_max.z - mesh_min.z
    rig_height = rig_max.z - rig_min.z
    if rig_height <= 0.0:
        return
    scale = mesh_height / rig_height
    metarig.scale = (scale, scale, scale)
    activate_object(metarig)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rig_bounds = get_object_bounds(metarig)
    if not rig_bounds:
        return
    rig_min, rig_max = rig_bounds
    rig_center = (rig_min + rig_max) * 0.5
    mesh_center = (mesh_min + mesh_max) * 0.5
    offset = Vector((mesh_center.x - rig_center.x, mesh_center.y - rig_center.y, mesh_min.z - rig_min.z))
    metarig.location = metarig.location + offset


def generate_rigify_rig(metarig: bpy.types.Object) -> bpy.types.Object:
    ensure_object_mode()
    before_names = {obj.name for obj in bpy.data.objects}
    activate_object(metarig)
    bpy.ops.pose.rigify_generate()
    new_armatures = [
        obj
        for obj in bpy.data.objects
        if obj.name not in before_names and obj.type == "ARMATURE"
    ]
    if not new_armatures:
        rig = bpy.data.objects.get("rig")
        if rig and rig.type == "ARMATURE":
            return rig
        raise RuntimeError("Rigify generation did not create a rig")
    return new_armatures[0]


def parent_mesh_to_rig(mesh_obj: bpy.types.Object, rig_obj: bpy.types.Object) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    rig_obj.select_set(True)
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def hide_rigify_helpers(metarig: bpy.types.Object) -> None:
    hide_object(metarig)
    for collection in bpy.data.collections:
        name = collection.name
        if name.startswith("WGT") or name.startswith("WGTS"):
            hide_collection(collection)


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
        landmarks: list | None = None
        if poses:
            pose = poses[0] if isinstance(poses[0], dict) else None
            if pose:
                landmarks = pose.get("landmarks") or None
        if not landmarks:
            poses_raw = frame.get("poses3d_raw") or []
            pose_raw = poses_raw[0] if poses_raw and isinstance(poses_raw[0], dict) else None
            if pose_raw:
                landmarks = pose_raw.get("world") or None
        if not landmarks:
            continue
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


def get_pose_landmarks(frame: dict, source: str) -> list | None:
    if source == "raw":
        poses = frame.get("poses3d_raw") or []
        if not poses:
            return None
        pose = poses[0] if isinstance(poses[0], dict) else None
        if not pose:
            return None
        return pose.get("world") or None
    poses = frame.get("poses3d") or []
    if not poses:
        return None
    pose = poses[0] if isinstance(poses[0], dict) else None
    if not pose:
        return None
    return pose.get("landmarks") or None


def set_pose_empties(
    empties: list[bpy.types.Object],
    landmarks: list,
    *,
    scale: float = 1.0,
    origin: Vector | None = None,
) -> None:
    for idx, coords in enumerate(landmarks):
        if idx >= len(empties):
            break
        if not isinstance(coords, (list, tuple)) or len(coords) < 3:
            continue
        x, y, z = coords[0], coords[1], coords[2]
        bx, by, bz = to_z_up_coords(x, y, z)
        pos = Vector((bx, by, bz))
        if origin is not None:
            pos -= origin
        pos *= scale
        empties[idx].location = pos


def find_first_pose_landmarks(frames: list[dict], source: str) -> list | None:
    for frame in frames:
        landmarks = get_pose_landmarks(frame, source)
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


def animate_pose_empties(
    frames: list[dict],
    empties: list[bpy.types.Object],
    *,
    source: str,
    scale: float = 1.0,
    origin: Vector | None = None,
) -> None:
    for frame in frames:
        landmarks = get_pose_landmarks(frame, source)
        if not landmarks:
            continue
        blender_frame = int(frame.get("i", 0)) + 1
        for idx, coords in enumerate(landmarks):
            if idx >= len(empties):
                break
            if not isinstance(coords, (list, tuple)) or len(coords) < 3:
                continue
            x, y, z = coords[0], coords[1], coords[2]
            bx, by, bz = to_z_up_coords(x, y, z)
            pos = Vector((bx, by, bz))
            if origin is not None:
                pos -= origin
            pos *= scale
            empties[idx].location = pos
            empties[idx].keyframe_insert(data_path="location", frame=blender_frame)


def build_landmark_index(names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def landmark_vector(
    landmarks: list,
    index: int | None,
    *,
    scale: float = 1.0,
    origin: Vector | None = None,
) -> Vector | None:
    if index is None or index < 0 or index >= len(landmarks):
        return None
    coords = landmarks[index]
    if not isinstance(coords, (list, tuple)) or len(coords) < 3:
        return None
    bx, by, bz = to_z_up_coords(coords[0], coords[1], coords[2])
    pos = Vector((bx, by, bz))
    if origin is not None:
        pos -= origin
    return pos * scale


def midpoint(a: Vector | None, b: Vector | None) -> Vector | None:
    if a is None or b is None:
        return None
    return (a + b) * 0.5


def orientation_from_torso(
    hip_left: Vector | None,
    hip_right: Vector | None,
    shoulder_left: Vector | None,
    shoulder_right: Vector | None,
) -> Matrix:
    if hip_left is None or hip_right is None or shoulder_left is None or shoulder_right is None:
        return Matrix.Identity(3)
    x_axis = (hip_right - hip_left)
    if x_axis.length <= 1e-6:
        return Matrix.Identity(3)
    x_axis.normalize()
    hip_mid = (hip_left + hip_right) * 0.5
    shoulder_mid = (shoulder_left + shoulder_right) * 0.5
    z_axis = shoulder_mid - hip_mid
    if z_axis.length <= 1e-6:
        return Matrix.Identity(3)
    z_axis.normalize()
    y_axis = z_axis.cross(x_axis)
    if y_axis.length <= 1e-6:
        return Matrix.Identity(3)
    y_axis.normalize()
    x_axis = y_axis.cross(z_axis)
    if x_axis.length <= 1e-6:
        return Matrix.Identity(3)
    x_axis.normalize()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def find_pose_bone(rig_obj: bpy.types.Object, candidates: list[str]) -> bpy.types.PoseBone | None:
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


def set_pose_bone_matrix(pose_bone: bpy.types.PoseBone, matrix: Matrix) -> None:
    pose_bone.matrix = matrix


def set_pose_bone_translation(pose_bone: bpy.types.PoseBone, target_pos: Vector) -> None:
    current_rot = pose_bone.matrix.to_3x3()
    pose_bone.matrix = Matrix.Translation(target_pos) @ current_rot.to_4x4()


def compute_pose_origin_and_scale(
    landmarks: list,
    indices: dict[str, int],
    mesh_obj: bpy.types.Object | None,
    base_scale: float,
) -> tuple[Vector | None, float]:
    if not landmarks:
        return None, base_scale
    hip_left = landmark_vector(landmarks, indices.get("left_hip"), scale=1.0)
    hip_right = landmark_vector(landmarks, indices.get("right_hip"), scale=1.0)
    origin = midpoint(hip_left, hip_right)

    scale = base_scale
    if mesh_obj:
        mesh_bounds = get_object_bounds(mesh_obj)
        if mesh_bounds:
            mesh_min, mesh_max = mesh_bounds
            mesh_height = mesh_max.z - mesh_min.z
            pose_coords = [
                landmark_vector(landmarks, idx, scale=1.0, origin=origin)
                for idx in range(len(landmarks))
            ]
            pose_coords = [vec for vec in pose_coords if vec is not None]
            if pose_coords:
                pose_min_z = min(vec.z for vec in pose_coords)
                pose_max_z = max(vec.z for vec in pose_coords)
                pose_height = pose_max_z - pose_min_z
                if pose_height > 1e-6:
                    scale *= mesh_height / pose_height
    return origin, scale


def compute_crouch_depth(mesh_obj: bpy.types.Object | None) -> float:
    depth = 0.2
    if mesh_obj:
        bounds = get_object_bounds(mesh_obj)
        if bounds:
            mesh_min, mesh_max = bounds
            mesh_height = mesh_max.z - mesh_min.z
            if mesh_height > 1e-6:
                depth = mesh_height * 0.2
    return max(depth, 0.05)


def keyframe_pose_bone_translation(
    pose_bone: bpy.types.PoseBone,
    target_pos: Vector,
    frame: int,
) -> None:
    set_pose_bone_translation(pose_bone, target_pos)
    pose_bone.keyframe_insert(data_path="location", frame=frame)


def animate_test_crouch_motion(
    rig_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object | None,
) -> None:
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    if frame_end <= frame_start:
        return
    duration = frame_end - frame_start
    fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else scene.render.fps
    span = int(round(fps)) if fps else duration
    span = max(span, 2)
    span = min(span, duration)
    mid_frame = frame_start + span // 2
    end_frame = frame_start + span

    controls = resolve_rigify_controls(rig_obj)
    root = controls.get("root")
    if root is None:
        raise RuntimeError("Failed to resolve root control")
    foot_left = controls.get("foot_ik.L")
    foot_right = controls.get("foot_ik.R")
    pole_left = controls.get("pole.L")
    pole_right = controls.get("pole.R")

    ensure_object_mode()
    activate_object(rig_obj)
    bpy.ops.object.mode_set(mode="POSE")

    root_base = root.matrix.to_translation()
    foot_left_base = foot_left.matrix.to_translation() if foot_left else None
    foot_right_base = foot_right.matrix.to_translation() if foot_right else None
    pole_left_base = pole_left.matrix.to_translation() if pole_left else None
    pole_right_base = pole_right.matrix.to_translation() if pole_right else None
    
    depth = compute_crouch_depth(mesh_obj)
    root_down = root_base - Vector((0.0, 0.0, depth * 0.3))

    # 屈む動作：IKを下げて膝を曲げる
    if foot_left_base:
        foot_left_down = foot_left_base - Vector((0.0, 0.0, depth * 0.7))
    else:
        foot_left_down = None
    
    if foot_right_base:
        foot_right_down = foot_right_base - Vector((0.0, 0.0, depth * 0.7))
    else:
        foot_right_down = None
    
    # ポールターゲットを前に出す（膝を前に突き出す）
    if pole_left_base:
        pole_left_forward = pole_left_base + Vector((0.0, depth * 0.5, 0.0))
    else:
        pole_left_forward = None
    
    if pole_right_base:
        pole_right_forward = pole_right_base + Vector((0.0, depth * 0.5, 0.0))
    else:
        pole_right_forward = None

    # キーフレーム設定
    poses = [
        (frame_start, root_base, foot_left_base, foot_right_base, pole_left_base, pole_right_base),
        (mid_frame, root_down, foot_left_down, foot_right_down, pole_left_forward, pole_right_forward),
        (end_frame, root_base, foot_left_base, foot_right_base, pole_left_base, pole_right_base),
    ]
    
    for frame, root_pos, foot_l, foot_r, pole_l, pole_r in poses:
        keyframe_pose_bone_translation(root, root_pos, frame)
        
        if foot_left and foot_l:
            keyframe_pose_bone_translation(foot_left, foot_l, frame)
        if foot_right and foot_r:
            keyframe_pose_bone_translation(foot_right, foot_r, frame)
        if pole_left and pole_l:
            keyframe_pose_bone_translation(pole_left, pole_l, frame)
        if pole_right and pole_r:
            keyframe_pose_bone_translation(pole_right, pole_r, frame)

    bpy.ops.object.mode_set(mode="OBJECT")


def animate_rigify_rig(
    rig_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
    frames: list[dict],
    landmark_names: list[str],
    *,
    source: str,
    pose_scale: float,
) -> None:
    if not frames:
        return
    indices = build_landmark_index(landmark_names)
    first_raw = find_first_pose_landmarks(frames, source)
    origin, scale = compute_pose_origin_and_scale(first_raw or [], indices, mesh_obj, pose_scale)
    base_scale = pose_scale if abs(pose_scale) > 1e-6 else 1.0
    root_scale = scale / base_scale

    controls = resolve_rigify_controls(rig_obj)
    if not controls:
        raise RuntimeError("Failed to resolve Rigify control bones")

    ensure_object_mode()
    activate_object(rig_obj)
    bpy.ops.object.mode_set(mode="POSE")

    arm_inv = rig_obj.matrix_world.inverted()

    for frame in frames:
        raw_landmarks = get_pose_landmarks(frame, source)
        if not raw_landmarks:
            continue
        projected_landmarks = None
        if source == "raw":
            projected_landmarks = get_pose_landmarks(frame, "projected")
        if projected_landmarks is None:
            projected_landmarks = raw_landmarks
        blender_frame = int(frame.get("i", 0)) + 1

        hip_left = landmark_vector(raw_landmarks, indices.get("left_hip"), scale=scale, origin=origin)
        hip_right = landmark_vector(raw_landmarks, indices.get("right_hip"), scale=scale, origin=origin)
        shoulder_left = landmark_vector(
            raw_landmarks, indices.get("left_shoulder"), scale=scale, origin=origin
        )
        shoulder_right = landmark_vector(
            raw_landmarks, indices.get("right_shoulder"), scale=scale, origin=origin
        )
        head_rel = landmark_vector(raw_landmarks, indices.get("nose"), scale=scale, origin=origin)
        hand_left_rel = landmark_vector(
            raw_landmarks, indices.get("left_wrist"), scale=scale, origin=origin
        )
        hand_right_rel = landmark_vector(
            raw_landmarks, indices.get("right_wrist"), scale=scale, origin=origin
        )
        foot_left_rel = landmark_vector(
            raw_landmarks, indices.get("left_ankle"), scale=scale, origin=origin
        )
        foot_right_rel = landmark_vector(
            raw_landmarks, indices.get("right_ankle"), scale=scale, origin=origin
        )

        hip_left_proj = landmark_vector(
            projected_landmarks, indices.get("left_hip"), scale=root_scale, origin=None
        )
        hip_right_proj = landmark_vector(
            projected_landmarks, indices.get("right_hip"), scale=root_scale, origin=None
        )
        root_pos = midpoint(hip_left_proj, hip_right_proj)
        if root_pos is None:
            root_pos = Vector((0.0, 0.0, 0.0))

        if "root" in controls:
            torso_rot = orientation_from_torso(hip_left, hip_right, shoulder_left, shoulder_right)
            rot_arm = arm_inv.to_3x3() @ torso_rot
            root_matrix = Matrix.Translation(arm_inv @ root_pos) @ rot_arm.to_4x4()
            controls["root"].rotation_mode = "QUATERNION"
            set_pose_bone_matrix(controls["root"], root_matrix)
            controls["root"].keyframe_insert(data_path="location", frame=blender_frame)
            controls["root"].keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)

        if head_rel is not None and "head" in controls:
            set_pose_bone_translation(controls["head"], arm_inv @ (root_pos + head_rel))
            controls["head"].keyframe_insert(data_path="location", frame=blender_frame)

        if hand_left_rel is not None and "hand_ik.L" in controls:
            set_pose_bone_translation(controls["hand_ik.L"], arm_inv @ (root_pos + hand_left_rel))
            controls["hand_ik.L"].keyframe_insert(data_path="location", frame=blender_frame)

        if hand_right_rel is not None and "hand_ik.R" in controls:
            set_pose_bone_translation(controls["hand_ik.R"], arm_inv @ (root_pos + hand_right_rel))
            controls["hand_ik.R"].keyframe_insert(data_path="location", frame=blender_frame)

        if foot_left_rel is not None and "foot_ik.L" in controls:
            set_pose_bone_translation(controls["foot_ik.L"], arm_inv @ (root_pos + foot_left_rel))
            controls["foot_ik.L"].keyframe_insert(data_path="location", frame=blender_frame)

        if foot_right_rel is not None and "foot_ik.R" in controls:
            set_pose_bone_translation(controls["foot_ik.R"], arm_inv @ (root_pos + foot_right_rel))
            controls["foot_ik.R"].keyframe_insert(data_path="location", frame=blender_frame)

    bpy.ops.object.mode_set(mode="OBJECT")


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

    pose_source = (args.pose3d_source or "projected").lower().strip()
    if pose_source not in {"projected", "raw"}:
        pose_source = "projected"
    pose_scale = float(args.pose_scale)
    pose_names, pose_bones = get_pose_skeleton(tracks)

    model_path = Path(args.model).expanduser().resolve() if args.model else None
    if model_path:
        mesh_obj = import_model(model_path, args.model_object)
        strip_mesh_rigging(mesh_obj)
        normalize_mesh_object(mesh_obj)
        ensure_rigify_enabled()
        metarig = add_human_metarig()
        fit_metarig_to_mesh(metarig, mesh_obj)
        rig_obj = generate_rigify_rig(metarig)
        parent_mesh_to_rig(mesh_obj, rig_obj)
        hide_rigify_helpers(metarig)
        if args.test_motion:
            animate_test_crouch_motion(rig_obj, mesh_obj)
        elif pose_names:
            animate_rigify_rig(
                rig_obj,
                mesh_obj,
                frames,
                pose_names,
                source=pose_source,
                pose_scale=pose_scale,
            )
    elif pose_names:
        pose_collection = get_or_create_collection("OD_Pose3D")
        pose_display_size = plane_width * 0.01
        pose_empties = create_pose_empties(
            pose_names,
            pose_collection,
            pose_display_size,
            prefix="pose",
        )
        initial_landmarks = find_first_pose_landmarks(frames, pose_source)
        origin = None
        scale = pose_scale
        if pose_source == "raw" and initial_landmarks:
            indices = build_landmark_index(pose_names)
            origin, scale = compute_pose_origin_and_scale(
                initial_landmarks,
                indices,
                None,
                pose_scale,
            )
        if initial_landmarks:
            set_pose_empties(pose_empties, initial_landmarks, scale=scale, origin=origin)
            create_pose_armature(
                pose_names,
                pose_bones,
                initial_landmarks,
                pose_empties,
                pose_collection,
            )
        animate_pose_empties(
            frames,
            pose_empties,
            source=pose_source,
            scale=scale,
            origin=origin,
        )

    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
