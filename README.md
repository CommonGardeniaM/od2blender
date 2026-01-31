od2blender
==========

動画1本から物体検出+追跡、または3D姿勢推定を行い、Blenderで再生できる `scene.blend` を生成するCLIツールです。
生成された `scene.blend` を開いてタイムライン再生するだけで、
背景動画の再生とEmptyの追従を同じシーン内で確認できます。

要件
----
- Python 3.10+
- Blender（ユーザー環境にインストール済みであること）
- 必要に応じて ffmpeg（codecの互換性問題がある場合）

インストール（uv）
----------------
```bash
uv pip install -e .
```

使い方
------
```bash
od2blender /path/to/video.mp4
```

3D姿勢推定を使う場合
```bash
od2blender /path/to/video.mp4 --pose3d
```

uvを使う場合
```bash
uv run od2blender /path/to/video.mp4
```

主なオプション
- `--out-dir PATH` : 出力先（既定: `<video_stem>_od2blender`）
- `--blender PATH` : Blender実行ファイルのフルパス
- `--open` : 生成後に `blender scene.blend` でUI起動
- `--no-blend` : Blender生成をスキップし tracks.json まで
- `--pose3d` : MediaPipeで3D姿勢推定を実行し、関節Empty/アーマチュアを生成
- `--pose-scale FLOAT` : 3D姿勢のスケール（既定: 1.0）
- `--pose-model PATH` : pose landmarkerモデル(.task)のパス（既定: 自動ダウンロード）
- `--test-motion` : モデルを初期姿勢から屈ませて戻すテストモーションを生成（`--model` 時のみ）

出力
----
`out_dir/` に以下が生成されます。
- `media/source.<ext>`（入力動画コピー、共通）
- `<YYYYMMDD_HHMM>_<mode>/`
  - `run.log`
  - `meta.json`
  - `tracks.json`
  - `scene.blend`（`--no-blend` 時は無し）

終了コード
---------
- 0: success
- 10: video open/meta failure
- 20: detection/tracking failure
- 30: blender executable not found / launch failure
- 31: blender importer script failure
- 40: output write failure

トラブルシュート
--------------
- Blenderが見つからない: `--blender` または `BLENDER_BIN` を指定
- 動画が読み込めない: codec不一致の可能性（ffmpegで再エンコード推奨）
- GPUを使いたい: CUDA対応環境で `ultralytics` がCUDAを認識する必要あり
- pose3dが動かない: `mediapipe` がインストールされているか確認
- pose3d初回実行が遅い: モデル(.task)のダウンロードが走るため
- 詳細は `run.log` を確認

手動E2E手順（最低限）
------------------
1. 短い動画で `od2blender` を実行
2. `out_dir/scene.blend` をBlenderで開く
3. タイムライン再生で動画とEmptyの追従を確認

開発
----
```bash
uv pip install -e .[dev]
pytest
```
