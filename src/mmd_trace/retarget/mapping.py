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
            ("left_shoulder", "spine", "l_shoulder"),
            ("left_arm", "l_shoulder", "l_elbow"),
            ("left_elbow", "l_elbow", "l_wrist"),
            ("left_wrist", "l_elbow", "l_wrist"),
            ("right_shoulder", "spine", "r_shoulder"),
            ("right_arm", "r_shoulder", "r_elbow"),
            ("right_elbow", "r_elbow", "r_wrist"),
            ("right_wrist", "r_elbow", "r_wrist"),
            ("left_leg", "l_hip", "l_knee"),
            ("left_knee", "l_knee", "l_ankle"),
            ("left_ankle", "l_knee", "l_ankle"),
            ("right_leg", "r_hip", "r_knee"),
            ("right_knee", "r_knee", "r_ankle"),
            ("right_ankle", "r_knee", "r_ankle"),
        ]

    def shoulder_keys(self) -> List[str]:
        return ["left_shoulder", "right_shoulder"]

    def bone_pairs(self) -> List[Tuple[str, str, str]]:
        """Return (bone_key, parent_bone_key, child_bone_key) for model rest vectors."""

        return [
            ("lower_body", "lower_body", "upper_body"),
            ("upper_body", "upper_body", "upper_body2"),
            ("upper_body2", "upper_body2", "neck"),
            ("neck", "neck", "head"),
            ("head", "neck", "head"),
            ("left_shoulder", "left_shoulder", "left_arm"),
            ("left_arm", "left_arm", "left_elbow"),
            ("left_elbow", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_shoulder", "right_arm"),
            ("right_arm", "right_arm", "right_elbow"),
            ("right_elbow", "right_elbow", "right_wrist"),
            ("left_leg", "left_leg", "left_knee"),
            ("left_knee", "left_knee", "left_ankle"),
            ("right_leg", "right_leg", "right_knee"),
            ("right_knee", "right_knee", "right_ankle"),
        ]
