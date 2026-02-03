"""Single image VPD generation pipeline."""
from __future__ import annotations

import logging

import cv2
from mmd_trace.io.pmx import load_pmx
from mmd_trace.io.vpd import write_vpd
from mmd_trace.pose_provider import create_pose_provider
from mmd_trace.retarget.coords import AxisTransform
from mmd_trace.retarget.pipeline import SolveMode, SolveResult, build_rotations

LOG = logging.getLogger(__name__)


def generate_vpd_from_image(
    image_path: str,
    pmx_path: str,
    out_path: str,
    axis: AxisTransform,
    vis_th: float,
    det_conf: float,
    solver: SolveMode,
    print_vpd: bool = False,
    print_debug: bool = False,
) -> tuple[str, SolveResult]:
    model = load_pmx(pmx_path)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Failed to read image. Check --image path.")

    provider = create_pose_provider(det_conf=det_conf)
    bundle = provider.detect(image)

    result = build_rotations(bundle, model, solver, axis, vis_th)
    vpd_text = write_vpd(out_path, model.model_name, result.bone_quat_local, result.bone_trans)

    LOG.info("VPD written: %s", out_path)
    LOG.info("Bones written: %s", len(result.bone_quat_local))

    if print_debug:
        LOG.info("Centers: %s", ", ".join(result.centers.keys()))
    if print_vpd:
        print(vpd_text)

    return vpd_text, result
