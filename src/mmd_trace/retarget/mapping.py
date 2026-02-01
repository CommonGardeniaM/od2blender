"""Joint-to-bone mapping and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BoneMapping:
    """Bone mapping and joint pairs for direction vectors."""

    bone_map: Dict[str, str]

    def vector_pairs(self) -> List[Tuple[str, str, str]]:
        """Return (bone_key, parent_joint, child_joint) entries."""

        return [
            ("lower_body", "pelvis", "spine"),
            ("upper_body", "spine", "chest"),
            ("upper_body2", "chest", "neck"),
            ("neck", "neck", "head"),
            ("head", "neck", "head"),
            ("left_arm", "l_shoulder", "l_elbow"),
            ("left_elbow", "l_elbow", "l_wrist"),
            ("right_arm", "r_shoulder", "r_elbow"),
            ("right_elbow", "r_elbow", "r_wrist"),
            ("left_leg", "l_hip", "l_knee"),
            ("left_knee", "l_knee", "l_ankle"),
            ("right_leg", "r_hip", "r_knee"),
            ("right_knee", "r_knee", "r_ankle"),
        ]

    def shoulder_keys(self) -> List[str]:
        return ["left_shoulder", "right_shoulder"]
