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

# 単一テストファイル実行
uv run pytest tests/test_mediapipe.py -v

# 単一テスト関数実行
uv run pytest tests/test_mediapipe.py::test_detect_pose -v

# カバレッジ付き実行
uv run pytest --cov=src/mmd_trace

# 開発依存のみインストール
uv pip install pytest

# CLI使用例
python -m mmd_trace list-bones --pmx model.pmx --format table
python -m mmd_trace pose-vpd --pmx model.pmx --image input.png --out pose.vpd
```

## コードスタイルガイドライン

### インポート
- **必ず** `from __future__ import annotations` を最初のインポートとして含める
- インポート順序: 標準ライブラリ → サードパーティ → ローカル（絶対インポートを優先）
- 明示的なインポートを使用（`from module import *` は避ける）
- サードパーティ: `import numpy as np`, `import cv2`（標準エイリアス）
- ローカル: `from mmd_trace.module import Class`（パッケージルートから絶対インポート）

### フォーマット
- 4スペースインデント
- 行長: 既存のファイルスタイルに従う（約100文字）
- トップレベルクラス/関数間は2行空行、メソッド内は1行空行
- 複数行コレクションには末尾カンマを使用
- 文字列: f-stringを優先

### 型ヒント
- Python 3.10+ が必要 - モダンな構文を使用
- `from typing import Dict, List, Optional, Any, Tuple` を使用
- 関数シグネチャには戻り値型が必須: `def func() -> ReturnType:`
- ヌル許容パラメータ/戻り値には `Optional[Type]` を使用
- Union型には `|` 構文を使用: `str | Path`
- Dataclassフィールドは可変デフォルトに `field(default_factory=...)` を使用

### 命名規則
- **クラス**: PascalCase（例: `PmxAdapter`, `MediaPipePoseProvider`）
- **関数/変数**: snake_case（例: `detect_pose`, `bone_list`）
- **定数**: モジュールレベルでUPPER_SNAKE_CASE（例: `LANDMARK_NAMES`）
- **プライベート**: アンダースコア接頭辞（例: `_load`, `_parse_pmx`）
- **モジュール名**: snake_case（例: `mediapipe_provider.py`）
- **型変数**: 大文字キャメルケース（例: `PoseFrame3D`）

### ドキュメンテーション
- モジュールdocstring: `"""簡単な説明。"""`
- クラスdocstring: 目的と主要属性を説明
- メソッドdocstring: ロジックが自明でない場合のみ
- Googleスタイルまたはシンプルな1行docstringを使用
- 複雑な関数にはArgsとReturnsセクションを含める

### エラー処理
- 深いネストよりも早期リターン/raiseを優先
- `raise ValueError("説明的なメッセージ")` を無効な入力に使用
- `raise FileNotFoundError` をファイル読み込み失敗に使用
- `None` を明示的にガード句で処理
- `Optional[Type]` でヌル許容の戻り値を示す

### ログ出力
- モジュールレベルのロガー: `LOG = logging.getLogger(__name__)`
- f-stringまたは%フォーマット: `LOG.info("メッセージ %s", value)`
- ログレベル: INFOは進捗、DEBUGは詳細、WARNINGは問題

### Dataclassパターン
```python
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class PoseFrame3D:
    """1フレームの33点3Dポーズデータ。"""
    frame_id: int
    landmarks: List[Landmark3D] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseFrame3D":
        return cls(
            frame_id=int(data.get("frame_id", 0)),
            landmarks=[Landmark3D.from_dict(lm) for lm in data.get("landmarks", [])],
        )
```

### テスト
- テストファイル: `tests/test_<module>.py`（現在は未整備、テスト用スクリプトは `src/mmd_trace/test_*.py`）
- テスト関数: `def test_<feature>():`
- `assert` 文を使用（pytestスタイル）
- パッケージルートからインポート: `from mmd_trace.module import Class`
- docstringよりも説明的なテスト名を優先

## プロジェクト構成

```
src/mmd_trace/
  __init__.py              # バージョン付きパッケージ初期化
  cli.py                   # argparseを使用したCLIエントリーポイント
  test_mediapipe.py        # MediaPipeプロバイダーのテストスクリプト
  debug_visualizer.py      # デバッグ可視化ツール
  io/                      # I/Oパッケージ
    __init__.py
    pmx.py                 # PMXファイル読み込みアダプタ
    vpd.py                 # VPDファイル書き出し
    pose.py                # ポーズ検出とVPD生成
  pose_provider/           # ポーズ検出プロバイダー
    __init__.py
    mediapipe_provider.py  # MediaPipe Tasks API実装
```

## CLI使用例

```bash
# PMXファイルのボーン一覧を表示
mmd-trace list-bones --pmx model.pmx --format table
mmd-trace list-bones --pmx model.pmx --format json --out bones.json

# 単一画像からVPDポーズを生成
mmd-trace pose-vpd --pmx model.pmx --image input.png --out pose.vpd --print_vpd --print_debug

# モデルタイプ指定（lite/full/heavy）
mmd-trace pose-vpd --pmx model.pmx --image input.png --out pose.vpd --model_type heavy
```

## 依存関係

主要な依存関係:
- numpy (配列操作)
- opencv-python (動画/画像I/O)
- mediapipe (ポーズ検定 - MediaPipe Tasks API)
- pymeshio (PMXファイル読み込み)

## 重要な実装詳細

### MediaPipe PoseLandmarker
- デフォルトモデルパス: `models/pose_landmarker_heavy.task`
- 33点ランドマークをサポート
- ワールド座標（メートル単位）と正規化座標の両方を提供

### PMXファイル読み込み
- PMX 2.0/2.1仕様に準拠
- UTF-16LEとUTF-8の両方のエンコーディングをサポート
- ボーン情報のみを抽出（頂点・面・マテリアルはスキップ）

### VPD出力
- Shift-JISエンコーディングで出力
- ローカル回転（クォータニオン）と移動をサポート
- ボーン名はPMXモデルの日本語名を使用
