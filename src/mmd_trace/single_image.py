"""Single image VPD generation from pose detection."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from mmd_trace.io.pmx import load_pmx
from mmd_trace.io.vpd import write_vpd
from mmd_trace.pose_processor import MP, build_rotations, compute_scale, map_points
from mmd_trace.pose_provider.mediapipe_provider import MediaPipePoseProvider


def generate_vpd_from_image(
    image_path: str,
    pmx_path: str,
    out_path: str,
    task_model: Optional[str],
    model_type: str,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    axis_matrix: Optional[np.ndarray],
    vis_th: float,
    det_conf: float,
    print_vpd: bool,
    print_debug: bool,
) -> str:
    """Generate VPD file from image and PMX model.

    Args:
        image_path: Path to input image
        pmx_path: Path to PMX model file
        out_path: Path to output VPD file
        task_model: Path to MediaPipe task model (optional)
        model_type: Model type for MediaPipe
        axis_x, axis_y, axis_z: Axis scaling factors
        vis_th: Visibility threshold
        det_conf: Detection confidence threshold
        print_vpd: Whether to print VPD content
        print_debug: Whether to print debug info

    Returns:
        VPD file content as string
    """
    # Load PMX model using new API
    model = load_pmx(pmx_path)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Failed to read image. Check --image path.")

    provider = MediaPipePoseProvider(
        model_asset_path=task_model,
        model_type=model_type,
        min_pose_detection_confidence=det_conf,
        min_pose_presence_confidence=det_conf,
        min_tracking_confidence=det_conf,
    )

    result = provider.detect_pose_world_np(image)
    if result is None:
        raise RuntimeError("PoseLandmarker detected no poses.")
    points, vis = result

    Ql, trans = build_rotations(
        points,
        vis,
        model,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_z=axis_z,
        axis_matrix=axis_matrix,
        vis_th=vis_th,
    )

    vpd_text = write_vpd(out_path, model.model_name, Ql, trans)
    print(f"OK: {out_path}")
    print(f"Bones written: {len(set(list(Ql.keys()) + list(trans.keys())))}")

    if print_debug:
        key = [MP["L_SHO"], MP["R_SHO"], MP["L_HIP"], MP["R_HIP"], MP["NOSE"]]
        print("vis(L_SHO,R_SHO,L_HIP,R_HIP,NOSE)=", [float(vis[i]) for i in key])
        points_mapped = map_points(points, axis_x, axis_y, axis_z, axis_matrix)
        sc = compute_scale(points_mapped, model)
        print("scale(mp->pmx)=", sc)

    if print_vpd:
        print("\n----- VPD BEGIN -----")
        print(vpd_text)
        print("----- VPD END -----")

    return vpd_text
