od2blender
==========

動画1本から物体検出+追跡、または3D姿勢推定を行い、Blenderで再生できる `scene.blend` を生成するCLIツールです。
生成された `scene.blend` を開いてタイムライン再生するだけで、背景動画の再生とEmpty/ポーズの追従を同じシーン内で確認できます。

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

uvを使う場合
```bash
uv run od2blender /path/to/video.mp4
```

モジュール実行
```bash
python -m od2blender.cli /path/to/video.mp4
```

例
--
物体検出+追跡（YOLO）
```bash
od2blender /path/to/video.mp4 --backend yolo
```

3Dモデルにリターゲット
```bash
od2blender /path/to/video.mp4 --model /path/to/character.fbx
```

Blender出力をスキップ（tracks.jsonのみ）
```bash
od2blender /path/to/video.mp4 --no-blend
```

主なオプション
--------------
- `--out-dir PATH` : 出力先（既定: `<video_stem>_od2blender`）
- `--blender PATH` : Blender実行ファイルのフルパス（既定: config -> `BLENDER_BIN` -> PATH）
- `--open` : 生成後に Blender UI を起動（Blender export有効時のみ）
- `--no-blend` : Blender生成をスキップし tracks.json まで
- `--backend ID` : 推論バックエンド（既定: config または `pose3d`）
- `--processors IDS` : 追加処理（カンマ区切り、既定: backendの既定）
- `--exporters IDS` : 出力先（カンマ区切り、既定: `tracks,blender`）
- `--pose-scale FLOAT` : 3D姿勢のスケール（既定: 1.0）
- `--pose-model PATH` : pose landmarkerモデル(.task)のパス（既定: 自動ダウンロード）
- `--pose3d-source projected|raw` : pose3d出力の参照元（projectedは平面投影、rawはワールド座標）
- `--model PATH` : Rigify用の3Dモデルパス（.blend/.fbx/.glb/.gltf/.obj）
- `--model-object NAME` : モデル内のオブジェクト名（任意）
- `--test-motion` : テスト用の屈みモーションを生成（`--model` 指定時のみ）

利用可能なID
-----------
- backends: `pose3d`, `yolo`
- processors: `pose3d_projector`, `dedupe_detections`
- exporters: `tracks`, `blender`

設定
----
- 実行時に使ったパス/設定は `.config/config.json` に保存され、次回の既定値になります。
- `video_path` を省略するには `default_video_path` が必要です。
- 既定値の詳細は `.config/config.json` を参照してください（実パスは環境依存のため例ではプレースホルダを使用）。
- 主なキー: `default_video_path`, `default_blender_path`, `default_backend`, `default_processors`, `default_exporters`, `default_pose3d_source`, `default_model_path`, `default_model_object`

例
```json
{
  "default_video_path": "/path/to/video.mp4",
  "default_blender_path": "/path/to/blender",
  "default_backend": "pose3d",
  "default_processors": ["pose3d_projector"],
  "default_exporters": ["tracks", "blender"],
  "default_pose3d_source": "projected",
  "default_model_path": "/path/to/model.fbx",
  "default_model_object": "Character"
}
```

出力
----
`out_dir/` に以下が生成されます。
- `media/source.<ext>`（入力動画コピー、共通）
- `<YYYYMMDD_HHMM>_<mode>[_NN]/`
  - `run.log`
  - `meta.json`
  - `tracks.json`
  - `scene.blend`（Blender export有効時のみ）

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
- pose3dが動かない: `mediapipe` がインストールされているか、`--pose-model` のパスが正しいか確認
- pose3d初回実行が遅い: モデル(.task)のダウンロードが走るため（`~/.cache/od2blender`）
- `--open` が効かない: `--no-blend` または `--exporters` で `blender` が無効になっている
- `--model` で失敗する: Rigifyアドオン有効化が必要。モデル形式や `--model-object` を確認
- 詳細は `run.log` を確認

手動E2E手順（最低限）
------------------
1. 短い動画で `od2blender` を実行
2. `out_dir/<YYYYMMDD_HHMM>_<mode>/scene.blend` をBlenderで開く
3. タイムライン再生で動画とEmpty/ポーズの追従を確認

開発
----
```bash
uv pip install -e ".[dev]"
uv run pytest
```
