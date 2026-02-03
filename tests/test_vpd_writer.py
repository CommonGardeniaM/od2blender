from __future__ import annotations

from pathlib import Path

import numpy as np

from mmd_trace.io.vpd import write_vpd


def test_write_vpd_crlf_and_spacing(tmp_path: Path) -> None:
    out_path = tmp_path / "pose.vpd"
    bone_trans = {
        "下半身": (0.0, 0.0, 0.0),
        "上半身": (0.0, 0.0, 0.0),
    }
    bone_quat_local = {
        "下半身": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "上半身": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    }

    vpd_text = write_vpd(str(out_path), "test_model", bone_quat_local, bone_trans)

    assert vpd_text.startswith("Vocaloid Pose Data file\r\n\r\n")
    assert "[Vocaloid Pose Data file" not in vpd_text
    assert "\n" not in vpd_text.replace("\r\n", "")
    assert "}\r\n\r\nBone1{" in vpd_text
    assert vpd_text.endswith("\r\n\r\n")

    data = out_path.read_bytes()
    assert data.startswith(b"Vocaloid Pose Data file\r\n\r\n")
    assert b"[Vocaloid Pose Data file" not in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    assert b"}\r\n\r\nBone1{" in data
    assert data.endswith(b"\r\n\r\n")
