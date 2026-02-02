from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerResult

LOG = logging.getLogger(__name__)

# MediaPipe モデルのURL
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def download_model(model_type: str = "full", cache_dir: Optional[str] = None) -> str:
    """MediaPipeモデルをダウンロードしてキャッシュする。"""
    if model_type not in MODEL_URLS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_URLS.keys())}")
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mmd_trace" / "models"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / f"pose_landmarker_{model_type}.task"
    
    if model_path.exists():
        LOG.info(f"Using cached model: {model_path}")
        return str(model_path)
    
    url = MODEL_URLS[model_type]
    LOG.info(f"Downloading MediaPipe model ({model_type})...")
    LOG.info(f"  From: {url}")
    LOG.info(f"  To: {model_path}")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        LOG.info(f"Model downloaded successfully")
    except Exception as e:
        LOG.error(f"Failed to download model: {e}")
        if model_path.exists():
            model_path.unlink()
        raise
    
    return str(model_path)


@dataclass
class Landmark3D:
    """MediaPipe 33点ランドマークの3D座標とメタデータ。"""
    index: int
    name: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
            "presence": self.presence,
        }
    
    @classmethod
    def from_mediapipe_landmark(
        cls,
        index: int,
        name: str,
        landmark: NormalizedLandmark,
    ) -> "Landmark3D":
        # MediaPipeはOptional[float]を返す可能性があるためデフォルト値を使用
        return cls(
            index=index,
            name=name,
            x=landmark.x if landmark.x is not None else 0.0,
            y=landmark.y if landmark.y is not None else 0.0,
            z=landmark.z if landmark.z is not None else 0.0,
            visibility=landmark.visibility if landmark.visibility is not None else 0.0,
            presence=landmark.presence if landmark.presence is not None else 0.0,
        )


@dataclass
class PoseFrame3D:
    """1フレーム（画像）の33点3Dポーズデータ。"""
    frame_id: int
    landmarks: List[Landmark3D] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }


# MediaPipe 33点のランドマーク名（日本語対応）
LANDMARK_NAMES = [
    "nose",                           # 0
    "left_eye_inner",                 # 1
    "left_eye",                       # 2
    "left_eye_outer",                 # 3
    "right_eye_inner",                # 4
    "right_eye",                      # 5
    "right_eye_outer",                # 6
    "left_ear",                       # 7
    "right_ear",                      # 8
    "mouth_left",                     # 9
    "mouth_right",                    # 10
    "left_shoulder",                  # 11
    "right_shoulder",                 # 12
    "left_elbow",                     # 13
    "right_elbow",                    # 14
    "left_wrist",                     # 15
    "right_wrist",                    # 16
    "left_pinky",                     # 17
    "right_pinky",                    # 18
    "left_index",                     # 19
    "right_index",                    # 20
    "left_thumb",                     # 21
    "right_thumb",                    # 22
    "left_hip",                       # 23
    "right_hip",                      # 24
    "left_knee",                      # 25
    "right_knee",                     # 26
    "left_ankle",                     # 27
    "right_ankle",                    # 28
    "left_heel",                      # 29
    "right_heel",                     # 30
    "left_foot_index",                # 31
    "right_foot_index",               # 32
]


# MediaPipeの骨格接続定義
POSE_CONNECTIONS = [
    # 顔
    (0, 1), (1, 2), (2, 3), (3, 7),  # 左目周辺
    (0, 4), (4, 5), (5, 6), (6, 8),  # 右目周辺
    (9, 10),                         # 口
    # 胴体
    (11, 12),                        # 肩
    (11, 23), (12, 24),             # 肩-腰
    (23, 24),                        # 腰
    # 左腕
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # 右腕
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # 左脚
    (23, 25), (25, 27), (27, 29), (27, 31),
    # 右脚
    (24, 26), (26, 28), (28, 30), (28, 32),
    # 頭と体の接続
    (0, 11), (0, 12),
]


class MediaPipePoseProvider:
    """MediaPipeを使用した33点3Dポーズ推定プロバイダー。"""
    
    def __init__(
        self,
        model_asset_path: Optional[str] = None,
        model_type: str = "full",
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # モデルパスが指定されていない場合は自動ダウンロード
        if model_asset_path is None:
            model_asset_path = download_model(model_type)
        
        # MediaPipe PoseLandmarkerの初期化
        base_options = mp.tasks.BaseOptions(model_asset_path=model_asset_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.detector = PoseLandmarker.create_from_options(options)
        LOG.info("MediaPipe PoseLandmarker initialized")
    
    def detect_pose(self, image: np.ndarray) -> Optional[PoseFrame3D]:
        """単一画像から33点3Dポーズを検出。"""
        # BGR -> RGB 変換
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # MediaPipe Image形式に変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # 検出実行
        result: PoseLandmarkerResult = self.detector.detect(mp_image)
        
        if not result.pose_landmarks:
            LOG.warning("No pose detected in image")
            return None
        
        # 最初の検出結果を使用
        landmarks = result.pose_landmarks[0]
        
        pose_frame = PoseFrame3D(frame_id=0)
        
        for i, landmark in enumerate(landmarks):
            name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"landmark_{i}"
            pose_frame.landmarks.append(
                Landmark3D.from_mediapipe_landmark(i, name, landmark)
            )
        
        LOG.info(f"Detected {len(pose_frame.landmarks)} landmarks")
        return pose_frame
    
    def detect_from_file(self, image_path: str) -> Optional[PoseFrame3D]:
        """画像ファイルから33点3Dポーズを検出。"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        LOG.info(f"Processing image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return self.detect_pose(image)
