import numpy as np
from scipy.spatial.transform import Rotation as R

# Sample data for testing x/y/z calculations
left_shoulder = np.array([0.59662926197052, 0.2969486117362976, -0.15011295676231384])
right_shoulder = np.array(
    [0.4050024449825287, 0.29932692646980286, -0.10676739364862442]
)
left_hip = np.array([0.5633628368377686, 0.5334545373916626, -0.016372714191675186])
right_hip = np.array([0.43847575783729553, 0.535342812538147, 0.01641947217285633])
center_shoulder = (left_shoulder + right_shoulder) / 2


def _norm(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def solve_upper_body_np_scipy(
    left_shoulder,
    right_shoulder,
    left_hip=None,
    right_hip=None,
    use_hip_center=True,
    use_rows_like_babylon=True,
    flip_z=False,
):
    """
    Babylon の solveUpperBody を numpy/scipy で再現。
    入力はワールド座標 (x,y,z) の 3要素（list/np.array）。

    - use_hip_center=True:
        Y軸(spineY) = shoulder_center - hip_center を使う（安定版）
      use_hip_center=False:
        Y軸(spineY) = shoulder_center を使う（元コードの挙動に近い）

    - use_rows_like_babylon=True:
        Babylon の FromValues に合わせて「行＝軸ベクトル」で回転行列を構成
      False:
        「列＝軸ベクトル」で構成（一般的な“基底を列に置く”流儀）

    - flip_z=True: 前後が逆に見えるときの符号調整
    """
    ls = np.asarray(left_shoulder, dtype=float)
    rs = np.asarray(right_shoulder, dtype=float)

    shoulder_center = 0.5 * (ls + rs)

    # X軸：右肩→左肩
    x = _norm(ls - rs)
    if x is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    # Y軸：腰→肩（安定） or 原点→肩中心（元コード寄り）
    if use_hip_center:
        if left_hip is None or right_hip is None:
            raise ValueError(
                "use_hip_center=True の場合、left_hip/right_hip が必要です。"
            )
        lh = np.asarray(left_hip, dtype=float)
        rh = np.asarray(right_hip, dtype=float)
        hip_center = 0.5 * (lh + rh)
        y = _norm(shoulder_center - hip_center)
    else:
        y = _norm(shoulder_center)

    if y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    # 直交化（Gram-Schmidt）：y から x 成分を除去
    y = y - x * np.dot(y, x)
    y = _norm(y)
    if y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    # Z軸：X×Y（前後が逆なら cross 順序 or flip_z）
    z = _norm(np.cross(x, y))
    if z is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    if flip_z:
        z = -z

    # 再計算して完全直交化
    y = _norm(np.cross(z, x))
    if y is None:
        return np.array([0, 0, 0, 1.0]), np.eye(3)

    # 回転行列を構成
    if use_rows_like_babylon:
        # 行が [x; y; z]
        M = np.vstack([x, y, z])
    else:
        # 列が [x y z]
        M = np.column_stack([x, y, z])

    # SciPyで quaternion (x,y,z,w)
    q = R.from_matrix(M).as_quat()
    return q, M


if __name__ == "__main__":
    # 適当な例（腕を水平に広げて正面向きっぽい配置）
    ls = left_shoulder
    rs = right_shoulder
    lh = left_hip
    rh = right_hip

    q, M = solve_upper_body_np_scipy(ls, rs, lh, rh, use_hip_center=True)
    print("quat (x,y,z,w):", q)
    print("M:\n", M)
    # 直交チェック
    print(
        "dot(x,y), dot(y,z), dot(z,x):",
        np.dot(M[0], M[1]),
        np.dot(M[1], M[2]),
        np.dot(M[2], M[0]),
    )
