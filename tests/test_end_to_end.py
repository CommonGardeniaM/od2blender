import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("MMD_TRACE_VIDEO") or not os.environ.get("MMD_TRACE_PMX"),
    reason="Set MMD_TRACE_VIDEO and MMD_TRACE_PMX to run end-to-end test",
)
def test_end_to_end(tmp_path):
    from mmd_trace.cli import main

    video = os.environ["MMD_TRACE_VIDEO"]
    pmx = os.environ["MMD_TRACE_PMX"]
    out_dir = tmp_path / "out"
    args = [
        "trace",
        "--video",
        video,
        "--pmx",
        pmx,
        "--out",
        str(out_dir),
        "--config",
        "config_example.yaml",
        "--smooth",
    ]
    import sys

    sys.argv = ["mmd-trace"] + args
    main()
    assert (out_dir / "motion.vmd").exists()
