"""Center/groove translation computation."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..io_pose import PoseSequence


def compute_center_groove(
    seq: PoseSequence,
    axis_map: Dict[str, object],
    pattern: str = "A",
    lowpass_alpha: float = 0.1,
    root_joint: str = "pelvis",
    scale: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return per-frame center and groove translation vectors."""

    positions = []
    for frame in seq.frames:
        if root_joint not in frame.joints:
            positions.append(np.zeros(3, dtype=np.float64))
            continue
        joint = frame.joints[root_joint]
        positions.append(np.array([joint.x, joint.y, joint.z], dtype=np.float64))

    if not positions:
        return [], []

    root0 = positions[0]
    delta = [pos - root0 for pos in positions]
    delta = [apply_axis_map(d * scale, axis_map) for d in delta]

    if pattern.upper() == "B":
        groove_base = _lowpass(delta, lowpass_alpha)
        center = [d - g for d, g in zip(delta, groove_base)]
        groove = [np.array([g[0], 0.0, g[2]], dtype=np.float64) for g in groove_base]
        center = [np.array([c[0], d[1], c[2]], dtype=np.float64) for c, d in zip(center, delta)]
        return center, groove

    groove = [np.array([d[0], 0.0, d[2]], dtype=np.float64) for d in delta]
    center = [np.array([0.0, d[1], 0.0], dtype=np.float64) for d in delta]
    return center, groove


def apply_axis_map(vec: np.ndarray, axis_map: Dict[str, object]) -> np.ndarray:
    swap = axis_map.get("swap", ["x", "y", "z"])
    invert = axis_map.get("invert", {"x": False, "y": False, "z": False})
    axis_to_value = {"x": vec[0], "y": vec[1], "z": vec[2]}
    mapped = np.array(
        [axis_to_value[swap[0]], axis_to_value[swap[1]], axis_to_value[swap[2]]],
        dtype=np.float64,
    )
    mapped[0] = -mapped[0] if invert.get("x", False) else mapped[0]
    mapped[1] = -mapped[1] if invert.get("y", False) else mapped[1]
    mapped[2] = -mapped[2] if invert.get("z", False) else mapped[2]
    return mapped


def _lowpass(series: List[np.ndarray], alpha: float) -> List[np.ndarray]:
    out = []
    prev = None
    for value in series:
        if prev is None:
            prev = value
        else:
            prev = alpha * value + (1.0 - alpha) * prev
        out.append(prev.copy())
    return out
