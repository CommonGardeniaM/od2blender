# mmd-trace (MVP)

単一画像のMediaPipe Pose推定から、MMD VPDを生成する最小構成です。  
モデルは **`models/pose_landmarker_heavy.task` 固定**（repo内）です。

## 要件

- Python 3.11+
- uv

## セットアップ

```bash
uv venv
uv pip install -e ".[dev]"
```

## 使い方（単一画像）

VPD生成（既定は **2Dロール**）:

```bash
uv run python -m mmd_trace pose-vpd \
  --pmx Millial_forMMD_v1.0.0.pmx \
  --image waking.png \
  --out output/pose.vpd \
  --solver 2d_roll
```

3Dソルバ:

```bash
uv run python -m mmd_trace pose-vpd \
  --pmx Millial_forMMD_v1.0.0.pmx \
  --image waking.png \
  --out output/pose.vpd \
  --solver 3d
```

デバッグ可視化:

```bash
uv run python -m mmd_trace debug-visualize \
  --pmx Millial_forMMD_v1.0.0.pmx \
  --image waking.png \
  --out output/debug_viz \
  --solver 2d_roll \
  --mode both
```

`output/debug_viz` に以下が出力されます:

- `debug_2d.png`（画像上の骨軸可視化）
- `debug_3d.png`（3Dプロット）

## 注意

- `models/pose_landmarker_heavy.task` が存在しない場合は起動時にエラーになります。
- 入力は **単一画像固定** です。

## MiKaPo solver (pose-only)
- `--solver mikapo` uses MiKaPo-compatible pose logic (no hands/fingers).
- For `mikapo`, visibility threshold is ignored and centering is disabled.
- Defaults switch to `axis_z=1` and `leg_mode=fk` for `mikapo` (explicit flags win).

```bash
uv run python -m mmd_trace pose-vpd \
  --pmx Millial_forMMD_v1.0.0.pmx \
  --image waking.png \
  --out output/pose_mikapo.vpd \
  --solver mikapo
```

## Pose skeleton image (MiKaPo-style)
```bash
uv run python -m mmd_trace pose-image \
  --image waking.png \
  --out output/pose_mikapo_skeleton.png
```
