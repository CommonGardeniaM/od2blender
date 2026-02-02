"""Pose to VPD conversion for single images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mmd_trace.io.pmx import PmxAdapter, PmxBoneInfo
from mmd_trace.io.vpd import write_vpd
from mmd_trace.pose_provider.mediapipe_provider import MediaPipePoseProvider

MP = {
    "NOSE": 0,
    "L_EAR": 7,
    "R_EAR": 8,
    "L_SHO": 11,
    "R_SHO": 12,
    "L_ELB": 13,
    "R_ELB": 14,
    "L_WRI": 15,
    "R_WRI": 16,
    "L_PNK": 17,
    "R_PNK": 18,
    "L_IDX": 19,
    "R_IDX": 20,
    "L_THM": 21,
    "R_THM": 22,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNE": 25,
    "R_KNE": 26,
    "L_ANK": 27,
    "R_ANK": 28,
    "L_HEE": 29,
    "R_HEE": 30,
    "L_FOO": 31,
    "R_FOO": 32,
}


def _norm(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    return vec / (length + eps)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64) / (np.dot(q, q) + 1e-9)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _norm(axis)
    s = np.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)], dtype=np.float64)


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _norm(a)
    b = _norm(b)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + d], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-9)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_inv(q))[:3]


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + 1e-9)


def make_basis(right: np.ndarray, up: np.ndarray, forward_hint: Optional[np.ndarray] = None) -> np.ndarray:
    r = _norm(right)
    u = up - np.dot(up, r) * r
    if np.linalg.norm(u) < 1e-6:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, r)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = tmp - np.dot(tmp, r) * r
    u = _norm(u)
    f = np.cross(r, u)
    if np.linalg.norm(f) < 1e-6:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    f = _norm(f)

    if forward_hint is not None and np.dot(f, forward_hint) < 0:
        u = -u
        f = -f
    return np.stack([r, u, f], axis=1)


@dataclass
class Bone:
    idx: int
    name_jp: str
    pos: np.ndarray
    parent: int
    flags: int
    tail_index: Optional[int]
    tail_offset: Optional[np.ndarray]
    ik_target: Optional[int]
    ik_links: List[int]


def _bones_from_adapter(adapter: PmxAdapter) -> Tuple[str, List[Bone], Dict[str, Bone]]:
    bones = []
    for bone in adapter.get_bone_list():
        tail_offset = None
        if bone.tail_offset is not None:
            tail_offset = np.array(bone.tail_offset, dtype=np.float64)
        bones.append(
            Bone(
                idx=bone.index,
                name_jp=bone.name,
                pos=np.array(bone.position, dtype=np.float64),
                parent=bone.parent,
                flags=bone.flags,
                tail_index=bone.tail_index,
                tail_offset=tail_offset,
                ik_target=bone.ik_target,
                ik_links=list(bone.ik_links),
            )
        )
    name_to_bone = {bone.name_jp: bone for bone in bones}
    return adapter.model_name, bones, name_to_bone


def parent_name(bones: List[Bone], bone: Bone) -> Optional[str]:
    if bone.parent is None or bone.parent < 0:
        return None
    return bones[bone.parent].name_jp


def rest_dir(bones: List[Bone], bone: Bone) -> np.ndarray:
    origin = bone.pos
    if bone.tail_index is not None and bone.tail_index >= 0:
        target = bones[bone.tail_index].pos
        direction = target - origin
    elif bone.tail_offset is not None:
        direction = bone.tail_offset
    else:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _norm(direction)


def hand_center(points: np.ndarray, side: str) -> np.ndarray:
    if side == "L":
        return (points[MP["L_IDX"]] + points[MP["L_PNK"]] + points[MP["L_THM"]]) / 3.0
    return (points[MP["R_IDX"]] + points[MP["R_PNK"]] + points[MP["R_THM"]]) / 3.0


def compute_scale(points: np.ndarray, name_to_bone: Dict[str, Bone]) -> float:
    left = (
        np.linalg.norm(points[MP["L_KNE"]] - points[MP["L_HIP"]])
        + np.linalg.norm(points[MP["L_ANK"]] - points[MP["L_KNE"]])
    )
    right = (
        np.linalg.norm(points[MP["R_KNE"]] - points[MP["R_HIP"]])
        + np.linalg.norm(points[MP["R_ANK"]] - points[MP["R_KNE"]])
    )
    mp_leg = (left + right) * 0.5

    pmx_left = (
        np.linalg.norm(name_to_bone["左ひざ"].pos - name_to_bone["左足"].pos)
        + np.linalg.norm(name_to_bone["左足首"].pos - name_to_bone["左ひざ"].pos)
    )
    pmx_right = (
        np.linalg.norm(name_to_bone["右ひざ"].pos - name_to_bone["右足"].pos)
        + np.linalg.norm(name_to_bone["右足首"].pos - name_to_bone["右ひざ"].pos)
    )
    pmx_leg = (pmx_left + pmx_right) * 0.5

    return float(pmx_leg / (mp_leg + 1e-9))


def twist_correct(base_q: np.ndarray, axis: np.ndarray, n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    axis = _norm(axis)
    n1 = quat_rotate(base_q, n_from)

    def proj(n: np.ndarray) -> Optional[np.ndarray]:
        nn = n - np.dot(n, axis) * axis
        if np.linalg.norm(nn) < 1e-6:
            return None
        return _norm(nn)

    a = proj(n1)
    b = proj(n_to)
    if a is None or b is None:
        return base_q

    ang = float(np.arctan2(np.dot(axis, np.cross(a, b)), np.dot(a, b)))
    q_tw = quat_from_axis_angle(axis, ang)
    return quat_mul(q_tw, base_q)


def rot_from_basis(rest_R: np.ndarray, pose_R: np.ndarray) -> np.ndarray:
    return quat_from_matrix(pose_R @ rest_R.T)


def build_rotations(
    points: np.ndarray,
    vis: np.ndarray,
    bones: List[Bone],
    name_to_bone: Dict[str, Bone],
    axis_x: float,
    axis_y: float,
    axis_z: float,
    vis_th: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float, float]]]:
    M = np.diag([axis_x, axis_y, axis_z]).astype(np.float64)
    points_mapped = points @ M.T

    scale = compute_scale(points_mapped, name_to_bone)
    pmx_hip = 0.5 * (name_to_bone["左足"].pos + name_to_bone["右足"].pos)

    def mp_to_pmx(pt: np.ndarray) -> np.ndarray:
        return pmx_hip + scale * pt

    hipC = 0.5 * (points_mapped[MP["L_HIP"]] + points_mapped[MP["R_HIP"]])
    shoC = 0.5 * (points_mapped[MP["L_SHO"]] + points_mapped[MP["R_SHO"]])
    earC = 0.5 * (points_mapped[MP["L_EAR"]] + points_mapped[MP["R_EAR"]])
    nose = points_mapped[MP["NOSE"]]

    Qw: Dict[str, np.ndarray] = {}

    model_forward_hint = _norm(name_to_bone["左つま先"].pos - name_to_bone["左足首"].pos)
    model_forward_hint[1] = 0.0
    if np.linalg.norm(model_forward_hint) < 1e-6:
        model_forward_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    model_forward_hint = _norm(model_forward_hint)

    def torso_basis_pose(right_vec: np.ndarray, up_vec: np.ndarray, forward_hint_vec: np.ndarray) -> np.ndarray:
        f_hint = _norm(forward_hint_vec)
        return make_basis(right_vec, up_vec, f_hint)

    def torso_basis_rest(right_vec: np.ndarray, up_vec: np.ndarray) -> np.ndarray:
        return make_basis(right_vec, up_vec, model_forward_hint)

    if np.min(vis[[MP["L_HIP"], MP["R_HIP"], MP["L_SHO"], MP["R_SHO"], MP["NOSE"]]]) > vis_th:
        pose_right = points_mapped[MP["R_HIP"]] - points_mapped[MP["L_HIP"]]
        pose_up = shoC - hipC
        pose_fhint = nose - shoC
        R_pose = torso_basis_pose(pose_right, pose_up, pose_fhint)

        rest_right = name_to_bone["右足"].pos - name_to_bone["左足"].pos
        rest_up = name_to_bone["上半身"].pos - name_to_bone["下半身"].pos
        R_rest = torso_basis_rest(rest_right, rest_up)
        Qw["下半身"] = rot_from_basis(R_rest, R_pose)

    if np.min(vis[[MP["L_SHO"], MP["R_SHO"], MP["L_HIP"], MP["R_HIP"], MP["NOSE"]]]) > vis_th:
        pose_right = points_mapped[MP["R_SHO"]] - points_mapped[MP["L_SHO"]]
        pose_up = earC - shoC
        pose_fhint = nose - earC
        R_pose = torso_basis_pose(pose_right, pose_up, pose_fhint)

        rest_right = name_to_bone["右肩"].pos - name_to_bone["左肩"].pos
        rest_up = name_to_bone["上半身2"].pos - name_to_bone["上半身"].pos
        R_rest = torso_basis_rest(rest_right, rest_up)
        Qw["上半身"] = rot_from_basis(R_rest, R_pose)

    if np.min(vis[[MP["L_EAR"], MP["R_EAR"], MP["L_SHO"], MP["R_SHO"], MP["NOSE"]]]) > vis_th:
        pose_right = points_mapped[MP["R_SHO"]] - points_mapped[MP["L_SHO"]]
        pose_up = earC - shoC
        pose_fhint = nose - earC
        R_pose = torso_basis_pose(pose_right, pose_up, pose_fhint)

        rest_right = name_to_bone["右肩"].pos - name_to_bone["左肩"].pos
        rest_up = name_to_bone["首"].pos - name_to_bone["上半身2"].pos
        R_rest = torso_basis_rest(rest_right, rest_up)
        Qw["上半身2"] = rot_from_basis(R_rest, R_pose)

    def dir_safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        if np.linalg.norm(d) < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return _norm(d)

    for bn, (a, b) in {
        "首": (shoC, earC),
        "頭": (earC, nose),
    }.items():
        if bn in name_to_bone:
            if bn == "首":
                cond = np.min(vis[[MP["L_SHO"], MP["R_SHO"], MP["L_EAR"], MP["R_EAR"]]])
            else:
                cond = np.min(vis[[MP["L_EAR"], MP["R_EAR"], MP["NOSE"]]])
            if cond > vis_th:
                tdir = dir_safe(a, b)
                Qw[bn] = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn]), tdir)

    def arm_world(side: str) -> None:
        if side == "L":
            sho, elb, wri = MP["L_SHO"], MP["L_ELB"], MP["L_WRI"]
            thm, idx, pnk = MP["L_THM"], MP["L_IDX"], MP["L_PNK"]
            bn_shoulder, bn_upper, bn_elbow, bn_wrist = "左肩", "左腕", "左ひじ", "左手首"
            bn_thumb0, bn_index1, bn_middle1, bn_ring1, bn_pinky1 = (
                "左親指０",
                "左人指１",
                "左中指１",
                "左薬指１",
                "左小指１",
            )
        else:
            sho, elb, wri = MP["R_SHO"], MP["R_ELB"], MP["R_WRI"]
            thm, idx, pnk = MP["R_THM"], MP["R_IDX"], MP["R_PNK"]
            bn_shoulder, bn_upper, bn_elbow, bn_wrist = "右肩", "右腕", "右ひじ", "右手首"
            bn_thumb0, bn_index1, bn_middle1, bn_ring1, bn_pinky1 = (
                "右親指０",
                "右人指１",
                "右中指１",
                "右薬指１",
                "右小指１",
            )

        if np.min(vis[[sho, elb, wri]]) <= vis_th:
            return

        tdir_sh = dir_safe(shoC, points_mapped[sho])
        Qw[bn_shoulder] = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn_shoulder]), tdir_sh)

        upper_dir = dir_safe(points_mapped[sho], points_mapped[elb])
        q_upper = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn_upper]), upper_dir)

        lower_dir = dir_safe(points_mapped[elb], points_mapped[wri])
        pose_n = np.cross(upper_dir, lower_dir)
        if np.linalg.norm(pose_n) < 1e-6:
            pose_n = np.cross(upper_dir, (nose - earC))
        pose_n = _norm(pose_n)

        rest_upper = rest_dir(bones, name_to_bone[bn_upper])
        rest_lower = rest_dir(bones, name_to_bone[bn_elbow])
        rest_n = np.cross(rest_upper, rest_lower)
        if np.linalg.norm(rest_n) < 1e-6:
            rest_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rest_n = _norm(rest_n)

        q_upper = twist_correct(q_upper, upper_dir, rest_n, pose_n)
        Qw[bn_upper] = q_upper

        q_elb = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn_elbow]), lower_dir)
        Qw[bn_elbow] = q_elb

        hC = hand_center(points_mapped, side)
        wrist_dir = dir_safe(points_mapped[wri], hC)
        q_wri = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn_wrist]), wrist_dir)

        v_idx = dir_safe(points_mapped[wri], points_mapped[idx])
        v_pnk = dir_safe(points_mapped[wri], points_mapped[pnk])
        palm_n = np.cross(v_idx, v_pnk)
        if np.linalg.norm(palm_n) < 1e-6:
            palm_n = pose_n
        palm_n = _norm(palm_n)

        if bn_index1 in name_to_bone and bn_pinky1 in name_to_bone:
            rest_vi = _norm(name_to_bone[bn_index1].pos - name_to_bone[bn_wrist].pos)
            rest_vp = _norm(name_to_bone[bn_pinky1].pos - name_to_bone[bn_wrist].pos)
            rest_pn = np.cross(rest_vi, rest_vp)
            if np.linalg.norm(rest_pn) < 1e-6:
                rest_pn = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            rest_pn = _norm(rest_pn)
            q_wri = twist_correct(q_wri, wrist_dir, rest_pn, palm_n)

        Qw[bn_wrist] = q_wri

        if np.min(vis[[wri, thm, idx, pnk]]) > vis_th:
            d_thumb = dir_safe(points_mapped[wri], points_mapped[thm])
            d_index = dir_safe(points_mapped[wri], points_mapped[idx])
            d_pinky = dir_safe(points_mapped[wri], points_mapped[pnk])
            d_mid = _norm(d_index + d_pinky)
            d_ring = _norm(d_index + d_pinky)

            for bn, d in [
                (bn_thumb0, d_thumb),
                (bn_index1, d_index),
                (bn_middle1, d_mid),
                (bn_ring1, d_ring),
                (bn_pinky1, d_pinky),
            ]:
                if bn in name_to_bone:
                    Qw[bn] = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn]), d)

    arm_world("L")
    arm_world("R")

    bone_trans: Dict[str, Tuple[float, float, float]] = {}
    for side in ["L", "R"]:
        if side == "L":
            ank, toe = MP["L_ANK"], MP["L_FOO"]
            bn_ik, bn_toeik = "左足ＩＫ", "左つま先ＩＫ"
        else:
            ank, toe = MP["R_ANK"], MP["R_FOO"]
            bn_ik, bn_toeik = "右足ＩＫ", "右つま先ＩＫ"

        if bn_ik in name_to_bone and bn_toeik in name_to_bone and np.min(vis[[ank, toe]]) > vis_th:
            ank_pmx = mp_to_pmx(points_mapped[ank])
            toe_pmx = mp_to_pmx(points_mapped[toe])

            bone_trans[bn_ik] = tuple((ank_pmx - name_to_bone[bn_ik].pos).tolist())
            bone_trans[bn_toeik] = tuple((toe_pmx - name_to_bone[bn_toeik].pos).tolist())

            foot_dir = toe_pmx - ank_pmx
            foot_dir[1] = 0.0
            if np.linalg.norm(foot_dir) > 1e-6:
                foot_dir = _norm(foot_dir)
                Qw[bn_ik] = quat_from_two_vectors(rest_dir(bones, name_to_bone[bn_ik]), foot_dir)

    if "センター" in name_to_bone:
        rest_toe_y = 0.5 * (
            name_to_bone["左つま先ＩＫ"].pos[1] + name_to_bone["右つま先ＩＫ"].pos[1]
        )
        if np.min(vis[[MP["L_FOO"], MP["R_FOO"]]]) > vis_th:
            toe_y_now = 0.5 * (mp_to_pmx(points_mapped[MP["L_FOO"]])[1] + mp_to_pmx(points_mapped[MP["R_FOO"]])[1])
            dy = toe_y_now - rest_toe_y
            bone_trans["センター"] = (0.0, -float(dy), 0.0)

    if "グルーブ" in name_to_bone and "グルーブ" not in Qw:
        Qw["グルーブ"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def get_parent(bn: str) -> Optional[str]:
        bone = name_to_bone.get(bn)
        if bone is None:
            return None
        return parent_name(bones, bone)

    Ql: Dict[str, np.ndarray] = {}
    for bn, qw in Qw.items():
        parent = get_parent(bn)
        if parent is not None and parent in Qw:
            Ql[bn] = quat_mul(quat_inv(Qw[parent]), qw)
        else:
            Ql[bn] = qw

    return Ql, bone_trans


def generate_vpd_from_image(
    image_path: str,
    pmx_path: str,
    out_path: str,
    task_model: Optional[str],
    model_type: str,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    vis_th: float,
    det_conf: float,
    print_vpd: bool,
    print_debug: bool,
) -> str:
    adapter = PmxAdapter(pmx_path)
    model_name, bones, name_to_bone = _bones_from_adapter(adapter)

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
        bones,
        name_to_bone,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_z=axis_z,
        vis_th=vis_th,
    )

    vpd_text = write_vpd(out_path, model_name, Ql, trans)
    print(f"OK: {out_path}")
    print(f"Bones written: {len(set(list(Ql.keys()) + list(trans.keys())))}")

    if print_debug:
        key = [MP["L_SHO"], MP["R_SHO"], MP["L_HIP"], MP["R_HIP"], MP["NOSE"]]
        print("vis(L_SHO,R_SHO,L_HIP,R_HIP,NOSE)=", [float(vis[i]) for i in key])
        M = np.diag([axis_x, axis_y, axis_z]).astype(np.float64)
        points_mapped = points @ M.T
        sc = compute_scale(points_mapped, name_to_bone)
        print("scale(mp->pmx)=", sc)

    if print_vpd:
        print("\n----- VPD BEGIN -----")
        print(vpd_text)
        print("----- VPD END -----")

    return vpd_text
