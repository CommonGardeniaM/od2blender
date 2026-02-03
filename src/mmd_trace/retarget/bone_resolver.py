"""PMX bone name resolution and rest references."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmd_trace.io.pmx import PmxBone, PmxModel

BONE_CANDIDATES: dict[str, list[str]] = {
    "center": ["センター", "center"],
    "groove": ["グルーブ", "groove"],
    "lower_body": ["下半身", "lower body"],
    "upper_body": ["上半身", "upper body"],
    "upper_body2": ["上半身2", "上半身２", "upper body2", "upper body 2"],
    "neck": ["首", "neck"],
    "head": ["頭", "head"],
    "shoulder_L": ["左肩", "shoulder_L", "shoulder.L"],
    "shoulder_R": ["右肩", "shoulder_R", "shoulder.R"],
    "arm_L": ["左腕", "arm_L", "arm.L"],
    "arm_R": ["右腕", "arm_R", "arm.R"],
    "elbow_L": ["左ひじ", "左肘", "elbow_L", "elbow.L"],
    "elbow_R": ["右ひじ", "右肘", "elbow_R", "elbow.R"],
    "wrist_L": ["左手首", "wrist_L", "wrist.L"],
    "wrist_R": ["右手首", "wrist_R", "wrist.R"],
    "leg_L": ["左足", "leg_L", "leg.L"],
    "leg_R": ["右足", "leg_R", "leg.R"],
    "knee_L": ["左ひざ", "左膝", "knee_L", "knee.L"],
    "knee_R": ["右ひざ", "右膝", "knee_R", "knee.R"],
    "ankle_L": ["左足首", "ankle_L", "ankle.L"],
    "ankle_R": ["右足首", "ankle_R", "ankle.R"],
}


def find_bone_name(model: PmxModel, candidates: list[str]) -> str | None:
    for cand in candidates:
        bone = model.get_bone(cand)
        if bone is not None:
            return bone.name
    return None


@dataclass
class BoneResolver:
    """Resolve PMX bone names and rest directions."""

    model: PmxModel
    bones: dict[str, str | None] = field(init=False)
    name_to_bone: dict[str, PmxBone] = field(init=False)

    def __post_init__(self) -> None:
        self.name_to_bone = {bone.name: bone for bone in self.model.bones}
        self.bones = {
            key: find_bone_name(self.model, names)
            for key, names in BONE_CANDIDATES.items()
        }

    def bone(self, key: str) -> str | None:
        return self.bones.get(key)

    def get_bone(self, name: str | None) -> PmxBone | None:
        if name is None:
            return None
        return self.name_to_bone.get(name)

    def parent_name(self, name: str | None) -> str | None:
        bone = self.get_bone(name)
        if bone is None or bone.parent_index < 0:
            return None
        parent = self.model.get_bone_by_index(bone.parent_index)
        if parent is None:
            return None
        return parent.name

    def rest_dir_world(self, name: str | None) -> np.ndarray | None:
        bone = self.get_bone(name)
        if bone is None:
            return None
        return self.model.get_bone_rest_direction(bone.index)

    def ref_dir_in_parent(self, name: str | None) -> np.ndarray | None:
        bone = self.get_bone(name)
        if bone is None:
            return None
        ref_world = self.model.get_bone_rest_direction(bone.index)
        if bone.parent_index < 0:
            return ref_world
        parent_rest = self.model.get_bone_rest_matrix(bone.parent_index)
        return parent_rest.T @ ref_world
