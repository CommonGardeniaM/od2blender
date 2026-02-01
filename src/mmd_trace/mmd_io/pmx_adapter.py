"""PMX loading utilities using pypmxvmd."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import contextlib
import io
from pathlib import Path
import struct

import pypmxvmd


@dataclass
class PmxBone:
    name: str
    position: List[float]
    parent_index: int


@dataclass
class PmxModelData:
    name: str
    bones: List[PmxBone]
    bone_index_map: Dict[str, int]


def load_pmx_model(path: str) -> PmxModelData:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = pypmxvmd.load_pmx(path)
    if not model.bones:
        return _load_pmx_fallback(path)

    bones = []
    bone_index_map: Dict[str, int] = {}
    for idx, bone in enumerate(model.bones):
        name = bone.name_jp or bone.name_en
        bones.append(PmxBone(name=name, position=list(bone.position), parent_index=bone.parent_index))
        bone_index_map[name] = idx
    model_name = model.header.name_jp or model.header.name_en
    return PmxModelData(name=model_name, bones=bones, bone_index_map=bone_index_map)


def _load_pmx_fallback(path: str) -> PmxModelData:
    """Fallback PMX parser for bone list only."""

    pmx_path = Path(path)
    bones: List[PmxBone] = []
    bone_index_map: Dict[str, int] = {}
    model_name = ""
    with pmx_path.open("rb") as f:
        sig = f.read(4)
        if sig != b"PMX ":
            raise ValueError("Not a PMX file")
        f.read(4)  # version
        header_size = struct.unpack("<B", f.read(1))[0]
        header = f.read(header_size)
        encoding = header[0]
        additional_uv = header[1]
        vertex_index_size = header[2]
        texture_index_size = header[3]
        bone_index_size = header[5]

        def read_text() -> str:
            length = struct.unpack("<I", f.read(4))[0]
            data = f.read(length)
            if encoding == 0:
                return data.decode("utf-16-le", errors="replace")
            return data.decode("utf-8", errors="replace")

        name_jp = read_text()
        name_en = read_text()
        read_text()
        read_text()
        model_name = name_jp or name_en

        def read_index(size: int) -> int:
            if size == 1:
                return struct.unpack("b", f.read(1))[0]
            if size == 2:
                return struct.unpack("<h", f.read(2))[0]
            return struct.unpack("<i", f.read(4))[0]

        vert_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(vert_count):
            f.read(4 * 3)
            f.read(4 * 3)
            f.read(4 * 2)
            if additional_uv:
                f.read(4 * 4 * additional_uv)
            weight_type = struct.unpack("<B", f.read(1))[0]
            if weight_type == 0:
                f.read(bone_index_size)
            elif weight_type == 1:
                f.read(bone_index_size * 2)
                f.read(4)
            elif weight_type == 2:
                f.read(bone_index_size * 4)
                f.read(4 * 4)
            elif weight_type == 3:
                f.read(bone_index_size * 2)
                f.read(4)
                f.read(4 * 3 * 3)
            elif weight_type == 4:
                f.read(bone_index_size * 4)
                f.read(4 * 4)
            f.read(4)

        face_count = struct.unpack("<I", f.read(4))[0]
        f.read(face_count * vertex_index_size)

        tex_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(tex_count):
            read_text()

        mat_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(mat_count):
            read_text()
            read_text()
            f.read(4 * 4)
            f.read(4 * 3)
            f.read(4)
            f.read(4 * 3)
            f.read(1)
            f.read(4 * 4)
            f.read(4)
            f.read(texture_index_size)
            f.read(texture_index_size)
            f.read(1)
            toon_flag = struct.unpack("<B", f.read(1))[0]
            if toon_flag == 0:
                f.read(texture_index_size)
            else:
                f.read(1)
            read_text()
            f.read(4)

        bone_count = struct.unpack("<I", f.read(4))[0]
        bones = []
        bone_index_map = {}
        for idx in range(bone_count):
            bone_name_jp = read_text()
            bone_name_en = read_text()
            pos = struct.unpack("<fff", f.read(12))
            parent = read_index(bone_index_size)
            f.read(4)  # deform layer
            flags = struct.unpack("<H", f.read(2))[0]
            if flags & 0x0001:
                f.read(bone_index_size)
            else:
                f.read(12)
            if flags & 0x0100:
                f.read(bone_index_size)
                f.read(4)
            if flags & 0x0400:
                f.read(12)
            if flags & 0x0800:
                f.read(12)
                f.read(12)
            if flags & 0x2000:
                f.read(4)
            if flags & 0x0020:
                f.read(bone_index_size)
                f.read(4)
                f.read(4)
                link_count = struct.unpack("<I", f.read(4))[0]
                for _ in range(link_count):
                    f.read(bone_index_size)
                    has_limit = struct.unpack("<B", f.read(1))[0]
                    if has_limit:
                        f.read(24)
            name = bone_name_jp or bone_name_en
            bones.append(PmxBone(name=name, position=list(pos), parent_index=parent))
            bone_index_map[name] = idx

    return PmxModelData(name=model_name, bones=bones, bone_index_map=bone_index_map)
