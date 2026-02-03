"""PMX file parser - full implementation for PMX 2.0/2.1 format."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

# =============================================================================
# Enums and Constants
# =============================================================================


class PmxWeightType:
    """Weight deformation types."""

    BDEF1 = 0
    BDEF2 = 1
    BDEF4 = 2
    SDEF = 3
    QDEF = 4  # PMX 2.1


class PmxBoneFlag:
    """Bone flags (first byte)."""

    INDEXED_TAIL = 0x0001
    ROTATABLE = 0x0002
    TRANSLATABLE = 0x0004
    VISIBLE = 0x0008
    ENABLED = 0x0010
    IK = 0x0020
    INHERIT_ROTATION = 0x0100
    INHERIT_TRANSLATION = 0x0200
    FIXED_AXIS = 0x0400
    LOCAL_COORD = 0x0800
    EXTERNAL_PARENT = 0x1000


class PmxMorphType:
    """Morph types."""

    GROUP = 0
    VERTEX = 1
    BONE = 2
    UV = 3
    UV1 = 4
    UV2 = 5
    UV3 = 6
    UV4 = 7
    MATERIAL = 8
    FLIP = 9  # PMX 2.1
    IMPULSE = 10  # PMX 2.1


class PmxRigidShape:
    """Rigid body shape types."""

    SPHERE = 0
    BOX = 1
    CAPSULE = 2


class PmxRigidType:
    """Rigid body physics types."""

    STATIC = 0
    DYNAMIC = 1
    DYNAMIC2 = 2  # Connected


class PmxJointType:
    """Joint types."""

    SPRING6DOF = 0
    PMX6DOF = 1  # PMX 2.1
    PMXP2P = 2  # PMX 2.1
    PMXConeTwist = 3  # PMX 2.1
    PMXSlider = 4  # PMX 2.1
    PMXHinge = 5  # PMX 2.1


# =============================================================================
# Data Classes - Header and Globals
# =============================================================================


@dataclass
class PmxHeader:
    """PMX file header."""

    magic: str
    version: float
    globals_count: int


@dataclass
class PmxGlobals:
    """PMX global settings affecting file layout."""

    text_encoding: int  # 0=UTF16LE, 1=UTF8
    additional_vec4_count: int  # 0-4
    vertex_index_size: int  # 1, 2, or 4
    texture_index_size: int  # 1, 2, or 4
    material_index_size: int  # 1, 2, or 4
    bone_index_size: int  # 1, 2, or 4
    morph_index_size: int  # 1, 2, or 4
    rigidbody_index_size: int  # 1, 2, or 4

    @property
    def encoding(self) -> str:
        return "utf-8" if self.text_encoding == 1 else "utf-16-le"


@dataclass
class PmxModelInfo:
    """Model information."""

    name: str
    name_en: str
    comment: str
    comment_en: str


# =============================================================================
# Data Classes - Geometry
# =============================================================================


@dataclass
class PmxVertex:
    """Vertex data."""

    index: int
    position: np.ndarray  # [3] float
    normal: np.ndarray  # [3] float
    uv: np.ndarray  # [2] float
    additional_uvs: list[np.ndarray]  # List of [4] float
    weight_type: int
    weight_bone_indices: list[int]
    weight_values: list[float]
    sdef_c: np.ndarray | None = None  # [3] for SDEF
    sdef_r0: np.ndarray | None = None  # [3] for SDEF
    sdef_r1: np.ndarray | None = None  # [3] for SDEF
    edge_scale: float = field(default=1.0)


@dataclass
class PmxFace:
    """Face (triangle) indices."""

    indices: tuple[int, int, int]


@dataclass
class PmxTexture:
    """Texture path."""

    index: int
    path: str


# =============================================================================
# Data Classes - Materials
# =============================================================================


@dataclass
class PmxMaterial:
    """Material data."""

    index: int
    name: str
    name_en: str
    diffuse: np.ndarray  # [4] RGBA
    specular: np.ndarray  # [3] RGB
    specular_power: float
    ambient: np.ndarray  # [3] RGB
    flags: int
    edge_color: np.ndarray  # [4] RGBA
    edge_scale: float
    texture_index: int
    sphere_texture_index: int
    sphere_mode: int
    toon_mode: int
    toon_index: int  # Internal if toon_mode==1, texture index if toon_mode==0
    meta_data: str
    surface_count: int


# =============================================================================
# Data Classes - Bones (Full)
# =============================================================================


@dataclass
class PmxBoneInherit:
    """Rotation/translation inheritance."""

    parent_index: int
    rate: float


@dataclass
class PmxBoneIkLink:
    """IK link with optional angle limits."""

    bone_index: int
    has_limits: bool
    min_limit: np.ndarray | None = None  # [3] radians
    max_limit: np.ndarray | None = None  # [3] radians


@dataclass
class PmxBoneIk:
    """IK information."""

    target_index: int
    loop_count: int
    limit_radian: float
    links: list[PmxBoneIkLink]


@dataclass
class PmxBone:
    """Complete bone data."""

    index: int
    name: str
    name_en: str
    position: np.ndarray  # [3] float
    parent_index: int
    layer: int
    flags: int
    # Tail (one of these)
    tail_index: int | None = None
    tail_offset: np.ndarray | None = None  # [3] float
    # Inherit
    inherit_rotation: PmxBoneInherit | None = None
    inherit_translation: PmxBoneInherit | None = None
    # Fixed axis
    fixed_axis: np.ndarray | None = None  # [3] float
    # Local coordinate
    local_x_vector: np.ndarray | None = None  # [3] float
    local_z_vector: np.ndarray | None = None  # [3] float
    # External parent
    external_parent: int | None = None
    # IK
    ik: PmxBoneIk | None = None

    @property
    def has_indexed_tail(self) -> bool:
        return bool(self.flags & 0x0001)

    @property
    def is_rotatable(self) -> bool:
        return bool(self.flags & 0x0002)

    @property
    def is_translatable(self) -> bool:
        return bool(self.flags & 0x0004)

    @property
    def is_visible(self) -> bool:
        return bool(self.flags & 0x0008)

    @property
    def is_enabled(self) -> bool:
        return bool(self.flags & 0x0010)

    @property
    def has_ik(self) -> bool:
        return bool(self.flags & 0x0020)

    @property
    def inherit_rotation_flag(self) -> bool:
        return bool(self.flags & 0x0100)

    @property
    def inherit_translation_flag(self) -> bool:
        return bool(self.flags & 0x0200)

    @property
    def has_fixed_axis(self) -> bool:
        return bool(self.flags & 0x0400)

    @property
    def has_local_coord(self) -> bool:
        return bool(self.flags & 0x0800)

    @property
    def has_external_parent(self) -> bool:
        return bool(self.flags & 0x1000)


# =============================================================================
# Data Classes - Morphs
# =============================================================================


@dataclass
class PmxMorphOffset:
    """Base class for morph offsets."""

    pass


@dataclass
class PmxMorphVertexOffset(PmxMorphOffset):
    """Vertex position morph."""

    vertex_index: int
    translation: np.ndarray  # [3] float


@dataclass
class PmxMorphUvOffset(PmxMorphOffset):
    """UV morph."""

    vertex_index: int
    values: np.ndarray  # [4] float


@dataclass
class PmxMorphBoneOffset(PmxMorphOffset):
    """Bone morph."""

    bone_index: int
    translation: np.ndarray  # [3] float
    rotation: np.ndarray  # [4] quaternion (x,y,z,w)


@dataclass
class PmxMorphMaterialOffset(PmxMorphOffset):
    """Material morph."""

    material_index: int
    calc_mode: int  # 0=multiply, 1=add
    diffuse: np.ndarray  # [4] RGBA
    specular: np.ndarray  # [3] RGB
    specular_power: float
    ambient: np.ndarray  # [3] RGB
    edge_color: np.ndarray  # [4] RGBA
    edge_scale: float
    texture_tint: np.ndarray  # [4] RGBA
    sphere_tint: np.ndarray  # [4] RGBA
    toon_tint: np.ndarray  # [4] RGBA


@dataclass
class PmxMorphGroupOffset(PmxMorphOffset):
    """Group morph."""

    morph_index: int
    rate: float


@dataclass
class PmxMorphFlipOffset(PmxMorphOffset):
    """Flip morph (PMX 2.1)."""

    morph_index: int
    rate: float


@dataclass
class PmxMorphImpulseOffset(PmxMorphOffset):
    """Impulse morph (PMX 2.1)."""

    rigidbody_index: int
    local: bool
    translation_velocity: np.ndarray  # [3]
    rotation_torque: np.ndarray  # [3]


@dataclass
class PmxMorph:
    """Complete morph data."""

    index: int
    name: str
    name_en: str
    panel: int  # 1-4 (or 0 for system)
    morph_type: int
    offsets: list[PmxMorphOffset]


# =============================================================================
# Data Classes - Display Frame
# =============================================================================


@dataclass
class PmxDisplayFrameItem:
    """Item in display frame."""

    item_type: int  # 0=bone, 1=morph
    index: int


@dataclass
class PmxDisplayFrame:
    """Display frame (bone/morph group for UI)."""

    index: int
    name: str
    name_en: str
    special: bool  # True for special frames (root, expression)
    items: list[PmxDisplayFrameItem]


# =============================================================================
# Data Classes - Physics
# =============================================================================


@dataclass
class PmxRigidbody:
    """Rigid body for physics."""

    index: int
    name: str
    name_en: str
    bone_index: int
    collision_group: int
    collision_mask: int
    shape: int
    shape_size: np.ndarray  # [3] float
    position: np.ndarray  # [3] float
    rotation: np.ndarray  # [3] radians
    mass: float
    damping_linear: float
    damping_angular: float
    restitution: float
    friction: float
    physics_type: int


@dataclass
class PmxJoint:
    """Joint (constraint) between rigid bodies."""

    index: int
    name: str
    name_en: str
    joint_type: int
    rigidbody_a: int
    rigidbody_b: int
    position: np.ndarray  # [3] float
    rotation: np.ndarray  # [3] radians
    linear_min: np.ndarray  # [3] float
    linear_max: np.ndarray  # [3] float
    angular_min: np.ndarray  # [3] radians
    angular_max: np.ndarray  # [3] radians
    # For spring6dof
    spring_linear: np.ndarray | None = None  # [3]
    spring_angular: np.ndarray | None = None  # [3]


@dataclass
class PmxSoftbody:
    """Soft body (PMX 2.1)."""

    index: int
    name: str
    name_en: str
    shape: int
    material_index: int
    group: int
    collision_mask: int
    flags: int
    b_link_distance: int
    cluster_count: int
    total_mass: float
    collision_margin: float
    aero_model: int
    cfg_k_vcf: float
    cfg_k_dp: float
    cfg_k_dg: float
    cfg_k_lf: float
    cfg_k_pr: float
    cfg_k_vc: float
    cfg_k_df: float
    cfg_k_mt: float
    cfg_k_chr: float
    cfg_k_khr: float
    cfg_k_shr: float
    cfg_k_ahr: float
    cluster_k_srhr_cl: float
    cluster_k_skhr_cl: float
    cluster_k_sshr_cl: float
    cluster_k_sr_splt_cl: float
    cluster_k_sk_splt_cl: float
    cluster_k_ss_splt_cl: float
    cluster_k_vcf: float
    cluster_k_dp: float
    cluster_k_drag: float
    cluster_k_pr: float
    anchors: list[tuple[int, int]]  # (rigidbody_index, is_near) pairs
    vertex_pins: list[int]  # vertex indices


# =============================================================================
# Main Model Class
# =============================================================================


@dataclass
class PmxModel:
    """Complete PMX model data."""

    # Header & Info
    header: PmxHeader
    globals: PmxGlobals
    model_info: PmxModelInfo

    # Geometry
    vertices: list[PmxVertex]
    faces: list[PmxFace]
    textures: list[PmxTexture]
    materials: list[PmxMaterial]

    # Rigging
    bones: list[PmxBone]
    morphs: list[PmxMorph]
    display_frames: list[PmxDisplayFrame]

    # Physics
    rigidbodies: list[PmxRigidbody]
    joints: list[PmxJoint]
    softbodies: list[PmxSoftbody]

    # Helper properties
    @property
    def model_name(self) -> str:
        return self.model_info.name

    @property
    def bone_count(self) -> int:
        return len(self.bones)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def get_bone(self, name: str) -> PmxBone | None:
        """Find bone by name (case-insensitive)."""
        name_lower = name.lower()
        for bone in self.bones:
            if bone.name.lower() == name_lower or bone.name_en.lower() == name_lower:
                return bone
        return None

    def get_bone_by_index(self, index: int) -> PmxBone | None:
        """Get bone by index."""
        if 0 <= index < len(self.bones):
            return self.bones[index]
        return None

    def get_bone_children(self, bone_index: int) -> list[PmxBone]:
        """Get direct children of a bone."""
        return [b for b in self.bones if b.parent_index == bone_index]

    def get_bone_hierarchy(self, bone_index: int) -> list[int]:
        """Get parent hierarchy from root to this bone (excluding itself)."""
        hierarchy = []
        current = bone_index
        while current >= 0:
            bone = self.get_bone_by_index(current)
            if bone is None or bone.parent_index < 0:
                break
            hierarchy.append(bone.parent_index)
            current = bone.parent_index
        return list(reversed(hierarchy))

    def validate(self) -> list[str]:
        """Validate model consistency and return list of issues."""
        issues = []

        # Check bone parent references
        for bone in self.bones:
            if bone.parent_index >= len(self.bones):
                issues.append(
                    f"Bone {bone.index} ({bone.name}) has invalid parent index {bone.parent_index}"
                )

        # Check tail references
        for bone in self.bones:
            if bone.has_indexed_tail and bone.tail_index is not None:
                if bone.tail_index >= len(self.bones):
                    issues.append(
                        f"Bone {bone.index} ({bone.name}) has invalid tail index {bone.tail_index}"
                    )

        # Check IK references
        for bone in self.bones:
            if bone.has_ik and bone.ik:
                if bone.ik.target_index >= len(self.bones):
                    issues.append(
                        f"Bone {bone.index} ({bone.name}) IK has invalid target index {bone.ik.target_index}"
                    )
                for link in bone.ik.links:
                    if link.bone_index >= len(self.bones):
                        issues.append(
                            f"Bone {bone.index} ({bone.name}) IK link has invalid index {link.bone_index}"
                        )

        # Check material texture references
        for mat in self.materials:
            if mat.texture_index >= 0 and mat.texture_index >= len(self.textures):
                issues.append(
                    f"Material {mat.index} ({mat.name}) has invalid texture index {mat.texture_index}"
                )

        # Check face vertex references
        for face in self.faces:
            for vi in face.indices:
                if vi >= len(self.vertices):
                    issues.append(f"Face references invalid vertex index {vi}")

        return issues

    # =========================================================================
    # Initial pose / rest pose helpers for motion capture
    # =========================================================================

    def get_bone_rest_direction(self, bone_index: int) -> np.ndarray:
        """Get rest direction (tail direction) for a bone in T/A-pose.

        Returns normalized direction vector from bone position to tail.
        Falls back to +Y if no tail is defined.
        """
        bone = self.get_bone_by_index(bone_index)
        if bone is None:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)

        if bone.tail_index is not None and bone.tail_index >= 0:
            tail_bone = self.get_bone_by_index(bone.tail_index)
            if tail_bone is not None:
                direction = tail_bone.position - bone.position
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    return direction / norm
        elif bone.tail_offset is not None:
            norm = np.linalg.norm(bone.tail_offset)
            if norm > 1e-8:
                return bone.tail_offset / norm

        # Default to +Y if no valid tail
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def get_bone_world_position(self, bone_index: int) -> np.ndarray:
        """Get world position of a bone in model coordinates.

        In PMX, bone.position is stored in model (global) coordinates.
        """
        bone = self.get_bone_by_index(bone_index)
        if bone is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return bone.position.copy()

    def get_bone_rest_matrix(self, bone_index: int) -> np.ndarray:
        """Get 3x3 rotation matrix representing bone's rest orientation.

        Constructs basis from:
        - If local axes are defined: uses X/Z axes and derives Y
        - Otherwise: uses tail direction as Y and derives X/Z from world axes

        Returns 3x3 rotation matrix (column vectors are local axes).
        """
        bone = self.get_bone_by_index(bone_index)
        if bone is None:
            return np.eye(3, dtype=np.float64)

        if (
            bone.has_local_coord
            and bone.local_x_vector is not None
            and bone.local_z_vector is not None
        ):
            x_axis = bone.local_x_vector.astype(np.float64)
            z_axis = bone.local_z_vector.astype(np.float64)

            x_norm = np.linalg.norm(x_axis)
            z_norm = np.linalg.norm(z_axis)
            if x_norm > 1e-6 and z_norm > 1e-6:
                x_axis = x_axis / x_norm
                z_axis = z_axis / z_norm
                y_axis = np.cross(z_axis, x_axis)
                y_norm = np.linalg.norm(y_axis)
                if y_norm > 1e-6:
                    y_axis = y_axis / y_norm
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
                    z_axis = np.cross(x_axis, y_axis)
                    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
                    return np.column_stack([x_axis, y_axis, z_axis])

        # Fallback to tail direction
        y_axis = self.get_bone_rest_direction(bone_index).astype(np.float64)
        x_axis = np.cross(y_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(y_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

        return np.column_stack([x_axis, y_axis, z_axis])

    def compute_bone_local_basis(self, bone_index: int) -> np.ndarray:
        """Compute basis matrix in parent-local space for motion capture.

        This is the key function for MiKaPo-style retargeting:
        - Returns 3x3 matrix representing bone's orientation relative to parent
        - Used as reference to compute rotation deltas from T/A-pose
        """
        bone = self.get_bone_by_index(bone_index)
        if bone is None or bone.parent_index < 0:
            # Root bone: world orientation is its local orientation
            return self.get_bone_rest_matrix(bone_index)

        # Get parent's world matrix
        parent_matrix = self.get_bone_rest_matrix(bone.parent_index)

        # Get this bone's world matrix
        bone_matrix = self.get_bone_rest_matrix(bone_index)

        # Compute local matrix: parent^-1 * bone
        # Since parent_matrix is orthonormal, inverse is transpose
        local_matrix = parent_matrix.T @ bone_matrix

        return local_matrix

    def get_parent_chain(self, bone_index: int) -> list[PmxBone]:
        """Get list of parent bones from root to immediate parent."""
        hierarchy = []
        current = bone_index
        while True:
            bone = self.get_bone_by_index(current)
            if bone is None or bone.parent_index < 0:
                break
            parent = self.get_bone_by_index(bone.parent_index)
            if parent is None:
                break
            hierarchy.insert(0, parent)
            current = bone.parent_index
        return hierarchy


# =============================================================================
# Binary Parser
# =============================================================================


class PmxBinaryReader:
    """Binary reader for PMX format."""

    def __init__(self, data: bytes) -> None:
        self.stream = BytesIO(data)
        self.encoding = "utf-16-le"
        self.globals: PmxGlobals | None = None

    def read_int8(self) -> int:
        return struct.unpack("b", self.stream.read(1))[0]

    def read_uint8(self) -> int:
        return struct.unpack("B", self.stream.read(1))[0]

    def read_int16(self) -> int:
        return struct.unpack("<h", self.stream.read(2))[0]

    def read_uint16(self) -> int:
        return struct.unpack("<H", self.stream.read(2))[0]

    def read_int32(self) -> int:
        return struct.unpack("<i", self.stream.read(4))[0]

    def read_uint32(self) -> int:
        return struct.unpack("<I", self.stream.read(4))[0]

    def read_float(self) -> float:
        return struct.unpack("<f", self.stream.read(4))[0]

    def read_vec2(self) -> np.ndarray:
        return np.array([self.read_float(), self.read_float()], dtype=np.float32)

    def read_vec3(self) -> np.ndarray:
        return np.array(
            [self.read_float(), self.read_float(), self.read_float()], dtype=np.float32
        )

    def read_vec4(self) -> np.ndarray:
        return np.array(
            [
                self.read_float(),
                self.read_float(),
                self.read_float(),
                self.read_float(),
            ],
            dtype=np.float32,
        )

    def read_string(self) -> str:
        length = self.read_int32()
        if length == 0:
            return ""
        data = self.stream.read(length)
        return data.decode(self.encoding, errors="replace")

    def read_index(self, size: int, signed: bool = True) -> int:
        """Read index of specified size."""
        if size == 1:
            return self.read_int8() if signed else self.read_uint8()
        elif size == 2:
            return self.read_int16() if signed else self.read_uint16()
        elif size == 4:
            return self.read_int32() if signed else self.read_uint32()
        else:
            raise ValueError(f"Invalid index size: {size}")

    def read_bone_index(self) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.bone_index_size, True)

    def read_morph_index(self) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.morph_index_size, True)

    def read_texture_index(self) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.texture_index_size, True)

    def read_material_index(self) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.material_index_size, True)

    def read_rigidbody_index(self) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.rigidbody_index_size, True)

    def read_vertex_index(self, signed: bool = False) -> int:
        if self.globals is None:
            raise RuntimeError("Globals not set")
        return self.read_index(self.globals.vertex_index_size, signed)


class PmxParser:
    """Complete PMX file parser."""

    def __init__(self) -> None:
        self.reader: PmxBinaryReader = PmxBinaryReader(b"")
        self.model: PmxModel | None = None

    def parse(self, data: bytes) -> PmxModel:
        """Parse PMX binary data."""
        self.reader = PmxBinaryReader(data)

        # Header
        header = self._parse_header()
        globals_data = self._parse_globals(header.globals_count)
        self.reader.globals = globals_data
        self.reader.encoding = globals_data.encoding

        # Model info
        model_info = self._parse_model_info()

        # Geometry
        vertices = self._parse_vertices(globals_data.additional_vec4_count)
        faces = self._parse_faces()
        textures = self._parse_textures()
        materials = self._parse_materials()

        # Rigging
        bones = self._parse_bones()
        morphs = self._parse_morphs()
        display_frames = self._parse_display_frames()

        # Physics
        rigidbodies = self._parse_rigidbodies()
        joints = self._parse_joints()

        # Soft body (PMX 2.1)
        softbodies: list[PmxSoftbody] = []
        if header.version >= 2.1:
            softbodies = self._parse_softbodies()

        self.model = PmxModel(
            header=header,
            globals=globals_data,
            model_info=model_info,
            vertices=vertices,
            faces=faces,
            textures=textures,
            materials=materials,
            bones=bones,
            morphs=morphs,
            display_frames=display_frames,
            rigidbodies=rigidbodies,
            joints=joints,
            softbodies=softbodies,
        )

        return self.model

    def _parse_header(self) -> PmxHeader:
        magic = self.reader.stream.read(4).decode("ascii", errors="replace")
        if magic != "PMX ":
            raise ValueError(f"Invalid PMX magic: {magic!r}")

        version = struct.unpack("<f", self.reader.stream.read(4))[0]
        globals_count = self.reader.read_uint8()

        return PmxHeader(magic=magic, version=version, globals_count=globals_count)

    def _parse_globals(self, globals_count: int) -> PmxGlobals:
        if globals_count < 8:
            raise ValueError(f"Invalid globals count: {globals_count}")

        data = self.reader.stream.read(globals_count)

        return PmxGlobals(
            text_encoding=data[0],
            additional_vec4_count=data[1],
            vertex_index_size=data[2],
            texture_index_size=data[3],
            material_index_size=data[4],
            bone_index_size=data[5],
            morph_index_size=data[6],
            rigidbody_index_size=data[7],
        )

    def _parse_model_info(self) -> PmxModelInfo:
        return PmxModelInfo(
            name=self.reader.read_string(),
            name_en=self.reader.read_string(),
            comment=self.reader.read_string(),
            comment_en=self.reader.read_string(),
        )

    def _parse_vertices(self, additional_uv_count: int) -> list[PmxVertex]:
        count = self.reader.read_int32()
        vertices = []

        for i in range(count):
            position = self.reader.read_vec3()
            normal = self.reader.read_vec3()
            uv = self.reader.read_vec2()

            additional_uvs = []
            for _ in range(additional_uv_count):
                additional_uvs.append(self.reader.read_vec4())

            weight_type = self.reader.read_uint8()
            bone_indices = []
            weight_values = []
            sdef_c = None
            sdef_r0 = None
            sdef_r1 = None

            if weight_type == PmxWeightType.BDEF1:
                bone_indices.append(self.reader.read_bone_index())
                weight_values.append(1.0)

            elif weight_type == PmxWeightType.BDEF2:
                bone_indices.append(self.reader.read_bone_index())
                bone_indices.append(self.reader.read_bone_index())
                weight_values.append(self.reader.read_float())
                weight_values.append(1.0 - weight_values[0])

            elif weight_type == PmxWeightType.BDEF4:
                for _ in range(4):
                    bone_indices.append(self.reader.read_bone_index())
                for _ in range(4):
                    weight_values.append(self.reader.read_float())

            elif weight_type == PmxWeightType.SDEF:
                bone_indices.append(self.reader.read_bone_index())
                bone_indices.append(self.reader.read_bone_index())
                weight_values.append(self.reader.read_float())
                weight_values.append(1.0 - weight_values[0])
                sdef_c = self.reader.read_vec3()
                sdef_r0 = self.reader.read_vec3()
                sdef_r1 = self.reader.read_vec3()

            elif weight_type == PmxWeightType.QDEF:
                for _ in range(4):
                    bone_indices.append(self.reader.read_bone_index())
                for _ in range(4):
                    weight_values.append(self.reader.read_float())

            edge_scale = self.reader.read_float()

            vertices.append(
                PmxVertex(
                    index=i,
                    position=position,
                    normal=normal,
                    uv=uv,
                    additional_uvs=additional_uvs,
                    weight_type=weight_type,
                    weight_bone_indices=bone_indices,
                    weight_values=weight_values,
                    sdef_c=sdef_c,
                    sdef_r0=sdef_r0,
                    sdef_r1=sdef_r1,
                    edge_scale=edge_scale,
                )
            )

        return vertices

    def _parse_faces(self) -> list[PmxFace]:
        count = self.reader.read_int32()
        faces = []

        for i in range(count // 3):
            idx0 = self.reader.read_vertex_index()
            idx1 = self.reader.read_vertex_index()
            idx2 = self.reader.read_vertex_index()
            faces.append(PmxFace(indices=(idx0, idx1, idx2)))

        return faces

    def _parse_textures(self) -> list[PmxTexture]:
        count = self.reader.read_int32()
        textures = []

        for i in range(count):
            textures.append(
                PmxTexture(
                    index=i,
                    path=self.reader.read_string(),
                )
            )

        return textures

    def _parse_materials(self) -> list[PmxMaterial]:
        count = self.reader.read_int32()
        materials = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            diffuse = self.reader.read_vec4()
            specular = self.reader.read_vec3()
            specular_power = self.reader.read_float()
            ambient = self.reader.read_vec3()
            flags = self.reader.read_uint8()
            edge_color = self.reader.read_vec4()
            edge_scale = self.reader.read_float()
            texture_index = self.reader.read_texture_index()
            sphere_texture_index = self.reader.read_texture_index()
            sphere_mode = self.reader.read_uint8()
            toon_mode = self.reader.read_uint8()

            if toon_mode == 1:
                toon_index = self.reader.read_uint8()
            else:
                toon_index = self.reader.read_texture_index()

            meta_data = self.reader.read_string()
            surface_count = self.reader.read_int32()

            materials.append(
                PmxMaterial(
                    index=i,
                    name=name,
                    name_en=name_en,
                    diffuse=diffuse,
                    specular=specular,
                    specular_power=specular_power,
                    ambient=ambient,
                    flags=flags,
                    edge_color=edge_color,
                    edge_scale=edge_scale,
                    texture_index=texture_index,
                    sphere_texture_index=sphere_texture_index,
                    sphere_mode=sphere_mode,
                    toon_mode=toon_mode,
                    toon_index=toon_index,
                    meta_data=meta_data,
                    surface_count=surface_count,
                )
            )

        return materials

    def _parse_bones(self) -> list[PmxBone]:
        count = self.reader.read_int32()
        bones = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            position = self.reader.read_vec3()
            parent_index = self.reader.read_bone_index()
            layer = self.reader.read_int32()

            flag1 = self.reader.read_uint8()
            flag2 = self.reader.read_uint8()
            flags = flag1 | (flag2 << 8)

            # Tail
            tail_index = None
            tail_offset = None
            if flags & 0x0001:  # Indexed tail
                tail_index = self.reader.read_bone_index()
            else:
                tail_offset = self.reader.read_vec3()

            # Inherit rotation
            inherit_rotation = None
            if flags & 0x0100:
                inherit_rotation = PmxBoneInherit(
                    parent_index=self.reader.read_bone_index(),
                    rate=self.reader.read_float(),
                )

            # Inherit translation
            inherit_translation = None
            if flags & 0x0200:
                inherit_translation = PmxBoneInherit(
                    parent_index=self.reader.read_bone_index(),
                    rate=self.reader.read_float(),
                )

            # Fixed axis
            fixed_axis = None
            if flags & 0x0400:
                fixed_axis = self.reader.read_vec3()

            # Local coordinate
            local_x_vector = None
            local_z_vector = None
            if flags & 0x0800:
                local_x_vector = self.reader.read_vec3()
                local_z_vector = self.reader.read_vec3()

            # External parent
            external_parent = None
            if flags & 0x1000:
                external_parent = self.reader.read_int32()

            # IK
            ik = None
            if flags & 0x0020:
                target_index = self.reader.read_bone_index()
                loop_count = self.reader.read_int32()
                limit_radian = self.reader.read_float()
                link_count = self.reader.read_int32()
                links = []
                for _ in range(link_count):
                    link_index = self.reader.read_bone_index()
                    has_limits = self.reader.read_uint8()
                    min_limit = None
                    max_limit = None
                    if has_limits:
                        min_limit = self.reader.read_vec3()
                        max_limit = self.reader.read_vec3()
                    links.append(
                        PmxBoneIkLink(
                            bone_index=link_index,
                            has_limits=bool(has_limits),
                            min_limit=min_limit,
                            max_limit=max_limit,
                        )
                    )
                ik = PmxBoneIk(
                    target_index=target_index,
                    loop_count=loop_count,
                    limit_radian=limit_radian,
                    links=links,
                )

            bones.append(
                PmxBone(
                    index=i,
                    name=name,
                    name_en=name_en,
                    position=position,
                    parent_index=parent_index,
                    layer=layer,
                    flags=flags,
                    tail_index=tail_index,
                    tail_offset=tail_offset,
                    inherit_rotation=inherit_rotation,
                    inherit_translation=inherit_translation,
                    fixed_axis=fixed_axis,
                    local_x_vector=local_x_vector,
                    local_z_vector=local_z_vector,
                    external_parent=external_parent,
                    ik=ik,
                )
            )

        return bones

    def _parse_morphs(self) -> list[PmxMorph]:
        count = self.reader.read_int32()
        morphs = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            panel = self.reader.read_uint8()
            morph_type = self.reader.read_uint8()
            offset_count = self.reader.read_int32()

            offsets = []
            for _ in range(offset_count):
                offset = self._parse_morph_offset(morph_type)
                offsets.append(offset)

            morphs.append(
                PmxMorph(
                    index=i,
                    name=name,
                    name_en=name_en,
                    panel=panel,
                    morph_type=morph_type,
                    offsets=offsets,
                )
            )

        return morphs

    def _parse_morph_offset(self, morph_type: int) -> PmxMorphOffset:
        if morph_type == PmxMorphType.GROUP:
            return PmxMorphGroupOffset(
                morph_index=self.reader.read_morph_index(),
                rate=self.reader.read_float(),
            )
        elif morph_type == PmxMorphType.VERTEX:
            return PmxMorphVertexOffset(
                vertex_index=self.reader.read_vertex_index(),
                translation=self.reader.read_vec3(),
            )
        elif morph_type == PmxMorphType.BONE:
            return PmxMorphBoneOffset(
                bone_index=self.reader.read_bone_index(),
                translation=self.reader.read_vec3(),
                rotation=self.reader.read_vec4(),
            )
        elif morph_type in (
            PmxMorphType.UV,
            PmxMorphType.UV1,
            PmxMorphType.UV2,
            PmxMorphType.UV3,
            PmxMorphType.UV4,
        ):
            return PmxMorphUvOffset(
                vertex_index=self.reader.read_vertex_index(),
                values=self.reader.read_vec4(),
            )
        elif morph_type == PmxMorphType.MATERIAL:
            return PmxMorphMaterialOffset(
                material_index=self.reader.read_material_index(),
                calc_mode=self.reader.read_uint8(),
                diffuse=self.reader.read_vec4(),
                specular=self.reader.read_vec3(),
                specular_power=self.reader.read_float(),
                ambient=self.reader.read_vec3(),
                edge_color=self.reader.read_vec4(),
                edge_scale=self.reader.read_float(),
                texture_tint=self.reader.read_vec4(),
                sphere_tint=self.reader.read_vec4(),
                toon_tint=self.reader.read_vec4(),
            )
        elif morph_type == PmxMorphType.FLIP:
            return PmxMorphFlipOffset(
                morph_index=self.reader.read_morph_index(),
                rate=self.reader.read_float(),
            )
        elif morph_type == PmxMorphType.IMPULSE:
            return PmxMorphImpulseOffset(
                rigidbody_index=self.reader.read_rigidbody_index(),
                local=bool(self.reader.read_uint8()),
                translation_velocity=self.reader.read_vec3(),
                rotation_torque=self.reader.read_vec3(),
            )
        else:
            raise ValueError(f"Unknown morph type: {morph_type}")

    def _parse_display_frames(self) -> list[PmxDisplayFrame]:
        count = self.reader.read_int32()
        frames = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            special = bool(self.reader.read_uint8())
            item_count = self.reader.read_int32()

            items = []
            for _ in range(item_count):
                item_type = self.reader.read_uint8()
                if item_type == 0:
                    index = self.reader.read_bone_index()
                else:
                    index = self.reader.read_morph_index()
                items.append(PmxDisplayFrameItem(item_type=item_type, index=index))

            frames.append(
                PmxDisplayFrame(
                    index=i,
                    name=name,
                    name_en=name_en,
                    special=special,
                    items=items,
                )
            )

        return frames

    def _parse_rigidbodies(self) -> list[PmxRigidbody]:
        count = self.reader.read_int32()
        rigidbodies = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            bone_index = self.reader.read_bone_index()
            collision_group = self.reader.read_uint8()
            collision_mask = self.reader.read_uint16()
            shape = self.reader.read_uint8()
            shape_size = self.reader.read_vec3()
            position = self.reader.read_vec3()
            rotation = self.reader.read_vec3()
            mass = self.reader.read_float()
            damping_linear = self.reader.read_float()
            damping_angular = self.reader.read_float()
            restitution = self.reader.read_float()
            friction = self.reader.read_float()
            physics_type = self.reader.read_uint8()

            rigidbodies.append(
                PmxRigidbody(
                    index=i,
                    name=name,
                    name_en=name_en,
                    bone_index=bone_index,
                    collision_group=collision_group,
                    collision_mask=collision_mask,
                    shape=shape,
                    shape_size=shape_size,
                    position=position,
                    rotation=rotation,
                    mass=mass,
                    damping_linear=damping_linear,
                    damping_angular=damping_angular,
                    restitution=restitution,
                    friction=friction,
                    physics_type=physics_type,
                )
            )

        return rigidbodies

    def _parse_joints(self) -> list[PmxJoint]:
        count = self.reader.read_int32()
        joints = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            joint_type = self.reader.read_uint8()
            rigidbody_a = self.reader.read_rigidbody_index()
            rigidbody_b = self.reader.read_rigidbody_index()
            position = self.reader.read_vec3()
            rotation = self.reader.read_vec3()
            linear_min = self.reader.read_vec3()
            linear_max = self.reader.read_vec3()
            angular_min = self.reader.read_vec3()
            angular_max = self.reader.read_vec3()

            spring_linear = None
            spring_angular = None
            if joint_type == 0:  # Spring6DOF
                spring_linear = self.reader.read_vec3()
                spring_angular = self.reader.read_vec3()

            joints.append(
                PmxJoint(
                    index=i,
                    name=name,
                    name_en=name_en,
                    joint_type=joint_type,
                    rigidbody_a=rigidbody_a,
                    rigidbody_b=rigidbody_b,
                    position=position,
                    rotation=rotation,
                    linear_min=linear_min,
                    linear_max=linear_max,
                    angular_min=angular_min,
                    angular_max=angular_max,
                    spring_linear=spring_linear,
                    spring_angular=spring_angular,
                )
            )

        return joints

    def _parse_softbodies(self) -> list[PmxSoftbody]:
        count = self.reader.read_int32()
        softbodies = []

        for i in range(count):
            name = self.reader.read_string()
            name_en = self.reader.read_string()
            shape = self.reader.read_uint8()
            material_index = self.reader.read_material_index()
            group = self.reader.read_uint8()
            collision_mask = self.reader.read_uint16()
            flags = self.reader.read_uint8()
            b_link_distance = self.reader.read_int32()
            cluster_count = self.reader.read_int32()
            total_mass = self.reader.read_float()
            collision_margin = self.reader.read_float()
            aero_model = self.reader.read_int32()

            # Config
            cfg_k_vcf = self.reader.read_float()
            cfg_k_dp = self.reader.read_float()
            cfg_k_dg = self.reader.read_float()
            cfg_k_lf = self.reader.read_float()
            cfg_k_pr = self.reader.read_float()
            cfg_k_vc = self.reader.read_float()
            cfg_k_df = self.reader.read_float()
            cfg_k_mt = self.reader.read_float()
            cfg_k_chr = self.reader.read_float()
            cfg_k_khr = self.reader.read_float()
            cfg_k_shr = self.reader.read_float()
            cfg_k_ahr = self.reader.read_float()

            # Cluster
            cluster_k_srhr_cl = self.reader.read_float()
            cluster_k_skhr_cl = self.reader.read_float()
            cluster_k_sshr_cl = self.reader.read_float()
            cluster_k_sr_splt_cl = self.reader.read_float()
            cluster_k_sk_splt_cl = self.reader.read_float()
            cluster_k_ss_splt_cl = self.reader.read_float()
            cluster_k_vcf = self.reader.read_float()
            cluster_k_dp = self.reader.read_float()
            cluster_k_drag = self.reader.read_float()
            cluster_k_pr = self.reader.read_float()

            # Anchors
            anchor_count = self.reader.read_int32()
            anchors = []
            for _ in range(anchor_count):
                rigidbody_index = self.reader.read_rigidbody_index()
                is_near = bool(self.reader.read_uint8())
                anchors.append((rigidbody_index, int(is_near)))

            # Vertex pins
            pin_count = self.reader.read_int32()
            pins = []
            for _ in range(pin_count):
                pins.append(self.reader.read_vertex_index())

            softbodies.append(
                PmxSoftbody(
                    index=i,
                    name=name,
                    name_en=name_en,
                    shape=shape,
                    material_index=material_index,
                    group=group,
                    collision_mask=collision_mask,
                    flags=flags,
                    b_link_distance=b_link_distance,
                    cluster_count=cluster_count,
                    total_mass=total_mass,
                    collision_margin=collision_margin,
                    aero_model=aero_model,
                    cfg_k_vcf=cfg_k_vcf,
                    cfg_k_dp=cfg_k_dp,
                    cfg_k_dg=cfg_k_dg,
                    cfg_k_lf=cfg_k_lf,
                    cfg_k_pr=cfg_k_pr,
                    cfg_k_vc=cfg_k_vc,
                    cfg_k_df=cfg_k_df,
                    cfg_k_mt=cfg_k_mt,
                    cfg_k_chr=cfg_k_chr,
                    cfg_k_khr=cfg_k_khr,
                    cfg_k_shr=cfg_k_shr,
                    cfg_k_ahr=cfg_k_ahr,
                    cluster_k_srhr_cl=cluster_k_srhr_cl,
                    cluster_k_skhr_cl=cluster_k_skhr_cl,
                    cluster_k_sshr_cl=cluster_k_sshr_cl,
                    cluster_k_sr_splt_cl=cluster_k_sr_splt_cl,
                    cluster_k_sk_splt_cl=cluster_k_sk_splt_cl,
                    cluster_k_ss_splt_cl=cluster_k_ss_splt_cl,
                    cluster_k_vcf=cluster_k_vcf,
                    cluster_k_dp=cluster_k_dp,
                    cluster_k_drag=cluster_k_drag,
                    cluster_k_pr=cluster_k_pr,
                    anchors=anchors,
                    vertex_pins=pins,
                )
            )

        return softbodies


# =============================================================================
# Public API
# =============================================================================


def load_pmx(path: str | Path) -> PmxModel:
    """Load PMX file and return complete model data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PMX file not found: {path}")

    data = path.read_bytes()
    parser = PmxParser()
    return parser.parse(data)
