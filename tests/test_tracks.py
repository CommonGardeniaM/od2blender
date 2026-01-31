from __future__ import annotations

import pytest

from od2blender.ir.frames import build_empty_frames
from od2blender.process.dedupe import dedupe_detections


def test_build_empty_frames() -> None:
    frames = build_empty_frames(3, fps=30.0)
    assert len(frames) == 3
    assert frames[0]["detections"] == []
    assert frames[1]["t"] == pytest.approx(1 / 30.0)


def test_dedupe_detections_keeps_highest_conf() -> None:
    warnings: list[str] = []
    detections = [
        {"id": 1, "conf": 0.2},
        {"id": 1, "conf": 0.9},
        {"id": 2, "conf": 0.5},
    ]
    deduped = dedupe_detections(detections, frame_index=0, log_fn=warnings.append)
    assert len(deduped) == 2
    conf_by_id = {det["id"]: det["conf"] for det in deduped}
    assert conf_by_id[1] == 0.9
    assert conf_by_id[2] == 0.5
    assert warnings
