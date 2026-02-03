# AGENTS.md

`mmd-trace`リポジトリで作業するAIコーディングエージェント用ガイドライン。

## ビルド・テストコマンド

**パッケージマネージャー**: uv

```bash
# 環境セットアップ
uv venv
uv pip install -e ".[dev]"

# 依存関係の追加（pyproject.tomlに自動記録）
uv add numpy pandas

# 開発依存の追加
uv add --dev pytest black

# 全テスト実行
uv run pytest

# 単一テスト実行
uv run pytest tests/test_axis_transform.py::test_drop_z -v

# カバレッジ付き実行
uv run pytest --cov=src/mmd_trace
```

## コードスタイルガイドライン

### インポート
- **必ず** `from __future__ import annotations` を最初のインポートとして含める
- インポート順序: 標準ライブラリ → サードパーティ → ローカル（相対インポート）
- 明示的なインポートを使用（`from module import *` は避ける）
- サードパーティ: `import numpy as np`（標準エイリアス）
- ローカル: `from .module import Class` または `from ..module import func`

### フォーマット
- 4スペースインデント
- 行長: 既存のファイルスタイルに従う（約100文字）
- トップレベルクラス/関数間は2行空行、メソッド内は1行空行
- 複数行コレクションには末尾カンマを使用

### 型ヒント（Python 3.11）
- **Python 3.11+ 必須**
- `list[str]` / `dict[str, int]` / `tuple[int, int]` のようにビルトインジェネリクスを使う
- `Optional[T]` ではなく `T | None` を使う
- `from typing import ...` は `Protocol` など必要最小限のみ
- 関数シグネチャには戻り値型が必須: `def func() -> ReturnType:`
- Dataclassフィールドは可変デフォルトに `field(default_factory=...)` を使用

### 命名規則
- **クラス**: PascalCase（例: `PoseSequence`, `BoneMapping`）
- **関数/変数**: snake_case（例: `compute_center_groove`）
- **定数**: モジュールレベルでUPPER_SNAKE_CASE
- **プライベート**: アンダースコア接頭辞（例: `_safe_normalize`）
- **モジュール名**: snake_case（例: `center_groove.py`）

### ドキュメンテーション
- モジュールdocstring: `"""簡単な説明。"""`
- クラスdocstring: 目的と主要属性を説明
- メソッドdocstring: ロジックが自明でない場合のみ
- Googleスタイルまたはシンプルな1行docstringを使用

### エラー処理
- 深いネストよりも早期リターン/raiseを優先
- `raise ValueError("説明的なメッセージ")` を無効な入力に使用
- `None` を明示的にガード句で処理

### ログ出力
- モジュールレベルのロガー: `LOG = logging.getLogger(__name__)`
- f-stringまたは%フォーマット: `LOG.info("メッセージ %s", value)`
- ログレベル: INFOは進捗、DEBUGは詳細、WARNINGは問題

### Dataclassパターン
```python
from dataclasses import dataclass, field

@dataclass
class ConfigSection:
    """セクションの説明。"""
    enabled: bool = True
    values: dict[str, float] = field(default_factory=dict)
```

### テスト
- テストファイル: `tests/test_<module>.py`
- テスト関数: `def test_<feature>():`
- `assert` 文を使用（pytestスタイル）
- パッケージルートからインポート: `from mmd_trace.module import Class`

## プロジェクト構成（現在）

```
src/mmd_trace/
  app/
    single_image_pipeline.py
    spec.py
  io/
    pmx.py
    vpd.py
  pose_provider/
    base.py
    mediapipe_provider.py
  retarget/
    indices.py
    coords.py
    quat.py
    bone_resolver.py
    solver_2d_roll.py
    solver_3d.py
    pipeline.py
  viz/
    landmarks_overlay.py
    solver_debug.py
  cli.py
tests/
scripts/
```

## CLI使用例（単一画像）

```bash
# VPD生成（2Dロール既定）
uv run python -m mmd_trace pose-vpd \
  --pmx model.pmx \
  --image input.png \
  --out output/pose.vpd \
  --solver 2d_roll

# デバッグ可視化
uv run python -m mmd_trace debug-visualize \
  --pmx model.pmx \
  --image input.png \
  --out output/debug_viz \
  --solver 2d_roll \
  --mode both
```

## 依存関係

主要な依存関係:
- numpy
- opencv-python
- mediapipe
- scipy
- pydantic
- pypmxvmd
- pyyaml
