"""VPD writer for MMD pose data."""
from __future__ import annotations

import numpy as np


def write_vpd(
    path: str,
    model_name: str,
    bone_quat_local: dict[str, np.ndarray],
    bone_trans: dict[str, tuple[float, float, float]],
) -> str:
    """Write VPD file and return its content as text."""
    ordered: list[str] = []
    for key in bone_trans.keys():
        if key not in ordered:
            ordered.append(key)
    for key in bone_quat_local.keys():
        if key not in ordered:
            ordered.append(key)

    lines: list[str] = []
    lines.append("[Vocaloid Pose Data file")
    lines.append("")
    lines.append(f"{model_name}.osm;\t\t// 親ファイル名")
    lines.append(f"{len(ordered)};\t\t\t\t// 総ポーズボーン数")
    lines.append("")

    for idx, bone_name in enumerate(ordered):
        tx, ty, tz = bone_trans.get(bone_name, (0.0, 0.0, 0.0))
        quat = bone_quat_local.get(
            bone_name,
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        lines.append(f"Bone{idx}{{{bone_name}")
        lines.append(f"  {tx:.6f},{ty:.6f},{tz:.6f};\t\t\t\t// trans x,y,z")
        lines.append(f"  {quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f};\t\t// Quatanion x,y,z,w")
        lines.append("}")

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="shift_jis", newline="\n") as file_handle:
        file_handle.write(text)
    return text
