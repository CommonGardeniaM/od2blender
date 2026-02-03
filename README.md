# mmd-trace (MVP)

単眼動画からMMD VMDモーションを生成するMVPツールです。処理は以下の3段階で構成されます。

1. ポーズ抽出（JSON化）
2. 平滑化 / 欠損補間
3. PMXボーンへのリターゲットとVMD出力（センター/グルーブ含む）

## インストール

uvで依存関係をインストールしてCLIを実行します。

```bash
uv venv
uv pip install -e .
```

MediaPipe pose landmarkerモデルのダウンロード:

```bash
python - <<'PY'
from pathlib import Path
import urllib.request

url = "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
target = Path("models") / "pose_landmarker.task"
target.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, target)
print("Saved", target)
PY
```

## 使い方

```bash
python -m mmd_trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --smooth
```

### デバッグ可視化（基準点 + ボーン軸）

`debug-visualize` で、基準点とボーン軸を単一画像にオーバーレイできます。3Dプロットも出力可能です。

```bash
uv run python -m mmd_trace debug-visualize \
  --pmx model.pmx \
  --image input.png \
  --out output/debug_viz \
  --mode both
```

`output/debug_viz` の出力内容:

- `debug_2d.png`（画像オーバーレイ）
- `debug_3d.png`（3Dプロット）

オプション:

- `--mode 2d|3d|both`（デフォルト: both）
- `--project 2d|world` 2Dオーバーレイの投影方式（デフォルト: 2d）
- `--axis_scale 1.0` 軸長の倍率
- `--no_fingers` 指ボーンを非表示（デフォルトは表示）
- `--dpi 150` 3Dプロットの出力DPI

補足:

- `--project 2d` はMediaPipeの画像正規化座標で重ねるため、画像とズレにくいです。
- 基準点はポーズ由来の位置（腰/肩/耳/目/鼻）を使用します。
- ボーン軸はポーズ由来の原点（肩/肘/手首/腰/膝/足首/つま先）に描画します。

平滑化を無効化:

```bash
python -m mmd_trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --no-smooth
```

`out_dir` の出力内容:

- `pose_raw.json`
- `pose_smooth.json`
- `motion.vmd`
- `debug_overlay.mp4`（任意）

## 設定例

全設定は `config_example.yaml` を参照してください。重要な注意点:

- ボーン名はPMXモデルと一致している必要があります。
- 軸変換は `axis_map` で適用されます。
- センター/グルーブのパターンは `A`（デフォルト）または `B` を選べます。

## ライセンスと利用上の注意

このプロジェクトは以下のアイデアやワークフローを参考にしています。

- https://github.com/miu200521358/mmd-auto-trace-4
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/01.-%E6%A6%82%E8%A6%81
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/02.%E4%BD%BF%E7%94%A8%E6%8A%80%E8%A1%93
- https://github.com/miu200521358/mmd-auto-trace-4/wiki/03.%E5%88%A9%E7%94%A8%E8%A6%8F%E7%B4%84

利用規約を確認し、生成物が条件に沿っていることを確認してください。

## 補足

- MVPにはMediaPipeプロバイダーのみが含まれます。
- IK / 指 / 表情モーフはMVPの対象外です。
