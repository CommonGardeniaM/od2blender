from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# パッケージルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmd_trace.pose_provider.mediapipe_provider import MediaPipePoseProvider
from mmd_trace.debug_visualizer import DebugVisualizer

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """MediaPipe 33点3Dポーズ推定のテスト実行。"""
    
    # 入力・出力パス
    input_image = "waking.png"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    pose_json_path = output_dir / "pose_33points.json"
    debug_image_path = output_dir / "debug_overlay.png"
    
    LOG.info("=" * 50)
    LOG.info("MediaPipe 33-point 3D Pose Estimation")
    LOG.info("=" * 50)
    
    # 1. MediaPipeプロバイダー初期化
    LOG.info("\n[1] Initializing MediaPipe PoseLandmarker...")
    provider = MediaPipePoseProvider(
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    # 2. ポーズ検出
    LOG.info(f"\n[2] Detecting pose from: {input_image}")
    try:
        pose_frame = provider.detect_from_file(input_image)
    except Exception as e:
        LOG.error(f"Failed to detect pose: {e}")
        sys.exit(1)
    
    if pose_frame is None:
        LOG.error("No pose detected in image")
        sys.exit(1)
    
    # 3. JSON保存
    LOG.info(f"\n[3] Saving pose data to: {pose_json_path}")
    pose_data = {
        "frame_count": 1,
        "frames": [pose_frame.to_dict()],
        "landmark_count": len(pose_frame.landmarks),
    }
    
    with open(pose_json_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, indent=2, ensure_ascii=False)
    LOG.info(f"Saved {len(pose_frame.landmarks)} landmarks to JSON")
    
    # 4. デバッグ可視化
    LOG.info(f"\n[4] Generating debug visualization: {debug_image_path}")
    visualizer = DebugVisualizer(show_labels=True, show_confidence=True)
    visualizer.visualize_from_file(input_image, pose_frame, str(debug_image_path))
    
    # 5. 結果サマリー
    LOG.info("\n" + "=" * 50)
    LOG.info("Results Summary")
    LOG.info("=" * 50)
    LOG.info(f"Input image: {input_image}")
    LOG.info(f"Landmarks detected: {len(pose_frame.landmarks)}")
    LOG.info(f"Output JSON: {pose_json_path}")
    LOG.info(f"Debug image: {debug_image_path}")
    
    # 主要ランドマークの座標を表示
    LOG.info("\n[Key Landmarks]")
    key_indices = [0, 11, 12, 23, 24, 27, 28]  # 鼻、両肩、両腰、両足首
    for lm in pose_frame.landmarks:
        if lm.index in key_indices:
            LOG.info(f"  {lm.name:20s}: x={lm.x:+.3f}, y={lm.y:+.3f}, z={lm.z:+.3f}, v={lm.visibility:.2f}")
    
    LOG.info("\n" + "=" * 50)
    LOG.info("Processing complete!")
    LOG.info("=" * 50)


if __name__ == "__main__":
    main()
