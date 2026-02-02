"""PMX file adapter for reading bone information."""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, asdict, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

LOG = logging.getLogger(__name__)


@dataclass
class PmxBoneInfo:
    """PMX bone information."""
    index: int
    name: str
    name_en: str
    position: Tuple[float, float, float]
    parent: int
    flags: int = 0
    tail_index: Optional[int] = None
    tail_offset: Optional[Tuple[float, float, float]] = None
    ik_target: Optional[int] = None
    ik_links: List[int] = field(default_factory=list)


class PmxBinaryReader:
    """Binary reader for PMX file format."""
    
    def __init__(self, data: bytes) -> None:
        self.stream = BytesIO(data)
        self.encoding = "utf-16-le"
        self.bone_index_size = 1
    
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
    
    def read_vec3(self) -> Tuple[float, float, float]:
        return (self.read_float(), self.read_float(), self.read_float())
    
    def read_string(self) -> str:
        length = self.read_int32()
        if length == 0:
            return ""
        data = self.stream.read(length)
        return data.decode(self.encoding, errors="replace")
    
    def read_index(self, size: int, signed: bool = True) -> int:
        """Read index of specified size."""
        if size == 1:
            val = self.read_int8() if signed else self.read_uint8()
        elif size == 2:
            val = self.read_int16() if signed else self.read_uint16()
        elif size == 4:
            val = self.read_int32() if signed else self.read_uint32()
        else:
            raise ValueError(f"Invalid index size: {size}")
        return val


class PmxAdapter:
    """Adapter for reading PMX model files."""
    
    def __init__(self, pmx_path: str | Path) -> None:
        """Initialize and load PMX file.
        
        Args:
            pmx_path: Path to PMX file
            
        Raises:
            FileNotFoundError: If PMX file not found
            Exception: If PMX parsing fails
        """
        self.pmx_path = Path(pmx_path)
        self._bones: List[PmxBoneInfo] = []
        self.model_name: str = ""
        
        if not self.pmx_path.exists():
            raise FileNotFoundError(f"PMX file not found: {self.pmx_path}")
        
        self._load()
    
    def _load(self) -> None:
        """Load PMX model from file."""
        LOG.info("Loading PMX file: %s", self.pmx_path)
        
        try:
            data = self.pmx_path.read_bytes()
            self._parse_pmx(data)
            LOG.info("Loaded PMX model with %d bones", len(self._bones))
            
        except Exception as e:
            LOG.error("Failed to load PMX file: %s", e)
            raise
    
    def _parse_pmx(self, data: bytes) -> None:
        """Parse PMX binary data according to PMX 2.0/2.1 specification."""
        reader = PmxBinaryReader(data)
        
        # PMX Header
        magic = reader.stream.read(4)
        if magic != b"PMX ":
            raise ValueError(f"Invalid PMX magic: {magic}")
        
        # Version
        version = struct.unpack("<f", reader.stream.read(4))[0]
        LOG.debug("PMX version: %.1f", version)
        
        # Globals
        globals_count = reader.read_uint8()
        if globals_count < 8:
            raise ValueError(f"Invalid globals count: {globals_count}")
        
        globals_data = reader.stream.read(globals_count)
        
        # Parse globals
        text_encoding = globals_data[0]  # 0 = UTF16LE, 1 = UTF8
        additional_vec4_count = globals_data[1]  # 0-4
        vertex_index_size = globals_data[2]  # 1, 2, or 4
        texture_index_size = globals_data[3]  # 1, 2, or 4
        material_index_size = globals_data[4]  # 1, 2, or 4
        bone_index_size = globals_data[5]  # 1, 2, or 4
        morph_index_size = globals_data[6]  # 1, 2, or 4
        rigidbody_index_size = globals_data[7]  # 1, 2, or 4
        
        reader.encoding = "utf-8" if text_encoding == 1 else "utf-16-le"
        reader.bone_index_size = bone_index_size
        
        LOG.debug("Encoding: %s, Bone index size: %d", reader.encoding, bone_index_size)
        
        # Model info
        model_name = reader.read_string()
        model_name_en = reader.read_string()
        comment = reader.read_string()
        comment_en = reader.read_string()

        self.model_name = model_name
        
        LOG.debug("Model: %s (%s)", model_name, model_name_en)
        
        # Skip vertices
        vertex_count = reader.read_int32()
        LOG.debug("Vertices: %d", vertex_count)
        for _ in range(vertex_count):
            # Position (12 bytes)
            reader.read_vec3()
            # Normal (12 bytes)
            reader.read_vec3()
            # UV (8 bytes)
            reader.read_float()
            reader.read_float()
            # Additional vec4s
            for _ in range(additional_vec4_count):
                reader.stream.read(16)  # 4 floats
            # Weight deform type (1 byte)
            weight_type = reader.read_uint8()
            # Weight deform data
            if weight_type == 0:  # BDEF1
                reader.read_index(bone_index_size, True)
            elif weight_type == 1:  # BDEF2
                reader.read_index(bone_index_size, True)
                reader.read_index(bone_index_size, True)
                reader.read_float()
            elif weight_type == 2:  # BDEF4
                for _ in range(4):
                    reader.read_index(bone_index_size, True)
                for _ in range(4):
                    reader.read_float()
            elif weight_type == 3:  # SDEF
                reader.read_index(bone_index_size, True)
                reader.read_index(bone_index_size, True)
                reader.read_float()
                reader.read_vec3()  # C
                reader.read_vec3()  # R0
                reader.read_vec3()  # R1
            elif weight_type == 4:  # QDEF (PMX 2.1)
                for _ in range(4):
                    reader.read_index(bone_index_size, True)
                for _ in range(4):
                    reader.read_float()
            # Edge scale (4 bytes)
            reader.read_float()
        
        # Skip faces
        face_count = reader.read_int32()
        LOG.debug("Faces: %d", face_count)
        for _ in range(face_count):
            reader.read_index(vertex_index_size, False)
        
        # Skip textures
        texture_count = reader.read_int32()
        LOG.debug("Textures: %d", texture_count)
        for _ in range(texture_count):
            reader.read_string()
        
        # Skip materials
        material_count = reader.read_int32()
        LOG.debug("Materials: %d", material_count)
        for _ in range(material_count):
            reader.read_string()  # Name
            reader.read_string()  # Name EN
            reader.stream.read(16)  # Diffuse (RGBA)
            reader.stream.read(16)  # Specular (RGB) + power (4 floats)
            reader.stream.read(12)  # Ambient (RGB)
            reader.read_uint8()  # Flags
            reader.stream.read(16)  # Edge color (RGBA)
            reader.read_float()  # Edge scale
            reader.read_index(texture_index_size, True)
            reader.read_index(texture_index_size, True)
            reader.read_uint8()  # Environment blend mode
            toon_ref = reader.read_uint8()  # Toon reference
            if toon_ref == 1:  # Internal toon
                reader.read_uint8()
            else:  # Texture toon
                reader.read_index(texture_index_size, True)
            reader.read_string()  # Meta data
            reader.read_int32()  # Surface count
        
        # Parse bones
        bone_count = reader.read_int32()
        LOG.debug("Bones: %d", bone_count)
        self._bones = []
        
        for i in range(bone_count):
            name = reader.read_string()
            name_en = reader.read_string()
            position = reader.read_vec3()
            parent_index = reader.read_index(bone_index_size, True)
            
            # Layer (4 bytes)
            reader.read_int32()
            
            # Flags (2 bytes)
            flag1 = reader.read_uint8()
            flag2 = reader.read_uint8()
            flags = flag1 | (flag2 << 8)
            
            # Tail position or bone index (depends on flag)
            tail_index: Optional[int] = None
            tail_offset: Optional[Tuple[float, float, float]] = None
            if flag1 & 0x01:  # Indexed tail position
                tail_index = reader.read_index(bone_index_size, True)
            else:
                tail_offset = reader.read_vec3()
            
            # Inherit rotation (flag bit 8)
            if flag2 & 0x01:
                reader.read_index(bone_index_size, True)
                reader.read_float()
            
            # Inherit translation (flag bit 9)
            if flag2 & 0x02:
                reader.read_index(bone_index_size, True)
                reader.read_float()
            
            # Fixed axis (flag bit 10)
            if flag2 & 0x04:
                reader.read_vec3()
            
            # Local coordinate (flag bit 11)
            if flag2 & 0x08:
                reader.read_vec3()  # X vector
                reader.read_vec3()  # Z vector
            
            # External parent (flag bit 12)
            if flag2 & 0x10:
                reader.read_int32()
            
            # IK (flag bit 5)
            ik_target: Optional[int] = None
            ik_links: List[int] = []
            if flag1 & 0x20:
                ik_target = reader.read_index(bone_index_size, True)  # Target
                reader.read_int32()  # Loop count
                reader.read_float()  # Limit radian
                link_count = reader.read_int32()
                for _ in range(link_count):
                    link_index = reader.read_index(bone_index_size, True)
                    ik_links.append(link_index)
                    has_limits = reader.read_uint8()
                    if has_limits:
                        reader.read_vec3()  # Min
                        reader.read_vec3()  # Max
            
            bone_info = PmxBoneInfo(
                index=i,
                name=name,
                name_en=name_en,
                position=position,
                parent=parent_index,
                flags=flags,
                tail_index=tail_index,
                tail_offset=tail_offset,
                ik_target=ik_target,
                ik_links=ik_links,
            )
            self._bones.append(bone_info)
    
    def get_bone_list(self) -> List[PmxBoneInfo]:
        """Get list of all bones in the model.
        
        Returns:
            List of PmxBoneInfo objects
        """
        return self._bones.copy()
    
    def get_bone_names(self) -> List[str]:
        """Get list of bone names.
        
        Returns:
            List of bone names (Japanese)
        """
        return [bone.name for bone in self._bones]
    
    def find_bone(self, name: str) -> Optional[PmxBoneInfo]:
        """Find bone by name (case-insensitive).
        
        Args:
            name: Bone name to search for
            
        Returns:
            PmxBoneInfo if found, None otherwise
        """
        name_lower = name.lower()
        
        for bone in self._bones:
            if bone.name.lower() == name_lower or bone.name_en.lower() == name_lower:
                return bone
        
        return None
    
    def print_bone_list(self) -> None:
        """Print formatted bone list to stdout."""
        if not self._bones:
            print("No bones found in PMX file.")
            return
        
        print(f"\nPMX File: {self.pmx_path}")
        print(f"Total Bones: {len(self._bones)}")
        print("-" * 70)
        print(f"{'Index':<6} {'Name':<30} {'Name(EN)':<20} {'Parent'}")
        print("-" * 70)
        
        for bone in self._bones:
            parent_str = str(bone.parent) if bone.parent >= 0 else "-"
            print(f"{bone.index:<6} {bone.name:<30} {bone.name_en:<20} {parent_str}")
        
        print("-" * 70)
    
    def export_bones_json(self, output_path: Optional[str | Path] = None) -> str:
        """Export bone list as JSON.
        
        Args:
            output_path: Output file path (optional, prints to stdout if None)
            
        Returns:
            JSON string
        """
        data = {
            "pmx_file": str(self.pmx_path),
            "total_bones": len(self._bones),
            "bones": [asdict(bone) for bone in self._bones],
        }
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json_str, encoding="utf-8")
            LOG.info("Exported bone list to: %s", output_path)
        
        return json_str
