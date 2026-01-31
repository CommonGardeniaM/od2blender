# Agents Notes

## プロジェクト概要
- **エントリポイント**: `od2blender.cli:app` (Typerベース; `od2blender VIDEO_PATH` で動作)
- **パイプライン**: `od2blender/pipeline.py` が終了コードと `run.log` へのロギングを管理
- **Blenderスクリプト**: `od2blender/blender/importer.py` はstdlib + `bpy` のみ使用
- **出力契約**: `run.log`, `meta.json`, `tracks.json`, `scene.blend`, `media/source.<ext>`
- **テスト**: `tests/` に配置、pytestで実行

## ビルド・テストコマンド

### インストール・依存管理
```bash
# 依存を追加
uv add <package>

# 開発依存を追加
uv add --dev pytest

# 依存を同期（pyproject.tomlから）
uv sync

# 開発依存も含めて同期
uv sync --dev
```

### テスト実行
```bash
# 全テスト
uv run pytest

# 特定のテストファイル
uv run pytest tests/test_tracks.py

# 特定のテスト関数
uv run pytest tests/test_tracks.py::test_build_empty_frames
uv run pytest tests/test_tracks.py::test_dedupe_detections_keeps_highest_conf

# 詳細表示
uv run pytest -v
```

### アプリケーション実行
```bash
uv run od2blender /path/to/video.mp4

# または
uv run python -m od2blender /path/to/video.mp4
```

### リント・型チェック
- **現在設定なし**: ruff, mypy, black等の設定ファイルは存在しません
- 必要に応じて`ruff check .`や`mypy od2blender`を実行可能

## コードスタイルガイドライン

### インポート順序
```python
from __future__ import annotations  # 必須

# 1. 標準ライブラリ（sorted）
import json
import sys
from pathlib import Path
from typing import Callable

# 2. サードパーティライブラリ（sorted）
import cv2
import typer
from loguru import logger

# 3. ローカルモジュール
from od2blender.tracks import build_tracks_payload
```

### 型ヒントと型定義
- `from __future__ import annotations` は全ファイルで必須
- Python 3.10+構文: `list[dict]` や `Path | None` を使用（`typing.List`や`Optional`は使用しない）
- 複雑な型は`Callable[[str], None] | None`のように記述
- データクラスには`@dataclass(frozen=True)`を推奨

### 命名規則
- **関数・変数**: snake_case (`run_pipeline`, `video_meta`)
- **クラス**: CapWords (`VideoMeta`, `Pose3DConfig`)
- **定数**: UPPER_CASE (`DEFAULT_PLANE_WIDTH`, `TRACKS_VERSION`)
- **例外**: `*Error`接尾辞 (`VideoMetaError`, `TrackingError`)

### エラーハンドリング
- ドメイン固有エラーはRuntimeErrorを継承したカスタム例外を定義
  ```python
  class TrackingError(RuntimeError):
      pass
  ```
- パイプラインは例外を捕捉して`ExitCode`に変換
- `logger.exception()`でスタックトレースを記録

### ロギング
- `loguru`を使用（pipeline.py内）
- 書式: `{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}`
- レベル: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`

### パス処理
- 常に`pathlib.Path`を使用（os.pathは使用しない）
- パスは`.expanduser().resolve()`で正規化
- Blender用相対パス: `make_blender_relpath()`で`//`プレフィックス付きパスを生成

### JSON出力
- `json.dump(data, handle, indent=2, ensure_ascii=False)`
- エンコーディング: UTF-8
- `save_json()`ユーティリティ関数を使用

### Blenderスクリプト制約（importer.py）
- **stdlibのみ** + `bpy`
- サードパーティライブラリは使用不可（typer, loguru, cv2等）
- BlenderのAPI規約に従う（コレクション、マテリアル、キーフレーム操作）

## 終了コード（pipeline.py）
| コード | 意味 |
|--------|------|
| 0 | 成功 |
| 10 | 動画オープン/メタデータ失敗 |
| 20 | 検出/追跡失敗 |
| 30 | Blender実行ファイル未検出/起動失敗 |
| 31 | Blenderインポータースクリプト失敗 |
| 40 | 出力書き込み失敗 |

## 外部設定・ルール
- **Cursorルール**: `.cursor/rules/` または `.cursorrules` は存在しません
- **Copilot指示**: `.github/copilot-instructions.md` は存在しません
