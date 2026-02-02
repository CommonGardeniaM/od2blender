# AGENTS.md

`mmd-trace`リポジトリで作業するAIコーディングエージェント用のガイドライン。

## ビルド・テストコマンド

**パッケージマネージャー**: uv (モダンPython)

```bash
# 環境セットアップ
uv venv
uv pip install -e ".[dev]"

# 依存関係の追加（pyproject.tomlに自動記録される）
uv add numpy pandas

# 開発依存の追加
uv add --dev pytest black

# 全テスト実行
uv run pytest

# 単一テスト実行
uv run pytest tests/test_center_groove.py::test_center_groove_pattern_a -v

# カバレッジ付き実行
uv run pytest --cov=src/mmd_trace

# 開発依存のみインストール
uv pip install pytest

# CLI使用例
python -m mmd_trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml
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

### 型ヒント
- Python 3.10+ が必要 - モダンな構文を使用
- `from typing import Dict, List, Optional, Any` を使用
- 関数シグネチャには戻り値型が必須: `def func() -> ReturnType:`
- ヌル許容パラメータ/戻り値には `Optional[Type]` を使用
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
- `Optional[Type]` でヌル許容の戻り値を示す

### ログ出力
- モジュールレベルのロガー: `LOG = logging.getLogger(__name__)`
- f-stringまたは%フォーマット: `LOG.info("メッセージ %s", value)`
- ログレベル: INFOは進捗、DEBUGは詳細、WARNINGは問題

### Dataclassパターン
```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ConfigSection:
    """セクションの説明。"""
    enabled: bool = True
    values: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSection":
        return cls(
            enabled=bool(data.get("enabled", True)),
            values=dict(data.get("values", {})),
        )
```

### テスト
- テストファイル: `tests/test_<module>.py`
- テスト関数: `def test_<feature>():`
- `assert` 文を使用（pytestスタイル）
- パッケージルートからインポート: `from mmd_trace.module import Class`
- docstringよりも説明的なテスト名を優先

### 設定
- `.yaml` 拡張子のYAML設定ファイル
- 小文字のキーとアンダースコアを使用
- ローダーでYAMLとJSONの両方をサポート
- デフォルトの骨マップは日本語の骨名を使用（例: `"センター"`, `"グルーブ"`）

## プロジェクト構成

```
src/mmd_trace/
  __init__.py          # バージョン付きパッケージ初期化
  cli.py               # argparseを使用したCLIエントリーポイント
  config.py            # Config dataclassと読み込み
  io_pose.py           # Pose JSON I/O with dataclasses
  io_video.py          # 動画処理
  smoothing.py         # ポーズ平滑化アルゴリズム
  diagnose.py          # 診断収集/レポート
  axis_auto.py         # 自動軸推論
  pose_provider/       # ポーズ検出プロバイダー
    base.py
    mediapipe_provider.py
  mmd_io/              # MMD形式I/O
    pmx_adapter.py
    vmd_writer.py
  retarget/            # リターゲティングロジック
    mapping.py         # 骨マッピング定義
    kinematics.py      # クォータニオン・ユーティリティ
    center_groove.py   # センター/グルーブ計算
    retargeter.py      # メインリターゲティングロジック
tests/                 # pytestテストファイル
```

## CLI使用例

```bash
# フルパイプライン
mmd-trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --smooth

# 平滑化なし
mmd-trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --no-smooth

# 診断付き
mmd-trace trace --video input.mp4 --pmx model.pmx --out out_dir --config config.yaml --diagnose
```

## 依存関係

主要な依存関係:
- numpy (配列操作)
- opencv-python (動画I/O)
- mediapipe (ポーズ検出)
- pypmxvmd (MMD VMD形式)
- pyyaml (設定読み込み)
