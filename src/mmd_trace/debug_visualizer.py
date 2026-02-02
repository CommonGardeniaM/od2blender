from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mmd_trace.pose_provider.mediapipe_provider import (
    Landmark3D,
    PoseFrame3D,
    POSE_CONNECTIONS,
)

LOG = logging.getLogger(__name__)


class DebugVisualizer:
    """MediaPipe 33点ポーズ検出結果のデバッグ可視化。"""
    
    # 描画設定
    LANDMARK_RADIUS = 5
    LANDMARK_COLOR = (0, 255, 0)  # 緑
    LANDMARK_COLOR_LOW_CONF = (0, 0, 255)  # 赤（低信頼度）
    
    CONNECTION_COLOR = (255, 255, 255)  # 白
    CONNECTION_THICKNESS = 2
    
    TEXT_COLOR = (255, 255, 0)  # シアン
    TEXT_SCALE = 0.4
    TEXT_THICKNESS = 1
    
    VISIBILITY_THRESHOLD = 0.5
    
    def __init__(self, show_labels: bool = True, show_confidence: bool = True) -> None:
        self.show_labels = show_labels
        self.show_confidence = show_confidence
    
    def visualize(
        self,
        image: np.ndarray,
        pose_frame: PoseFrame3D,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """ポーズ検出結果を画像にオーバーレイ描画。"""
        # コピーを作成（元画像を変更しない）
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # ランドマークがない場合
        if not pose_frame.landmarks:
            LOG.warning("No landmarks to visualize")
            cv2.putText(
                vis_image,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            if output_path:
                cv2.imwrite(output_path, vis_image)
                LOG.info(f"Saved visualization to: {output_path}")
            return vis_image
        
        # ランドマークインデックスマップ
        landmark_map = {lm.index: lm for lm in pose_frame.landmarks}
        
        # 骨格接続線を描画
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in landmark_map and end_idx in landmark_map:
                start_lm = landmark_map[start_idx]
                end_lm = landmark_map[end_idx]
                
                # 正規化座標 -> ピクセル座標
                x1 = int(start_lm.x * w)
                y1 = int(start_lm.y * h)
                x2 = int(end_lm.x * w)
                y2 = int(end_lm.y * h)
                
                cv2.line(vis_image, (x1, y1), (x2, y2), self.CONNECTION_COLOR, self.CONNECTION_THICKNESS)
        
        # ランドマーク点を描画
        for lm in pose_frame.landmarks:
            # 正規化座標 -> ピクセル座標
            x = int(lm.x * w)
            y = int(lm.y * h)
            
            # 信頼度に応じた色
            color = self.LANDMARK_COLOR if lm.visibility >= self.VISIBILITY_THRESHOLD else self.LANDMARK_COLOR_LOW_CONF
            
            # 点を描画
            cv2.circle(vis_image, (x, y), self.LANDMARK_RADIUS, color, -1)
            
            # ラベル表示
            if self.show_labels or self.show_confidence:
                label_parts = []
                if self.show_labels:
                    label_parts.append(f"{lm.index}:{lm.name}")
                if self.show_confidence:
                    label_parts.append(f"v={lm.visibility:.2f}")
                
                label = " ".join(label_parts)
                cv2.putText(
                    vis_image,
                    label,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.TEXT_SCALE,
                    self.TEXT_COLOR,
                    self.TEXT_THICKNESS,
                )
        
        # サマリー情報
        summary = f"Landmarks: {len(pose_frame.landmarks)}"
        cv2.putText(
            vis_image,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            LOG.info(f"Saved visualization to: {output_path}")
        
        return vis_image
    
    def visualize_from_file(
        self,
        image_path: str,
        pose_frame: PoseFrame3D,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """画像ファイルから可視化。"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        if output_path is None:
            # デフォルト出力パス
            import os
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_debug{ext}"
        
        return self.visualize(image, pose_frame, output_path)
