"""Configuration loading and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import yaml


@dataclass
class AxisMap:
    """Axis swap/invert mapping for coordinate conversion."""

    swap: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    invert: Dict[str, bool] = field(
        default_factory=lambda: {"x": False, "y": True, "z": False}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxisMap":
        swap = data.get("swap", ["x", "y", "z"])
        invert = data.get("invert", {"x": False, "y": False, "z": False})
        return cls(swap=list(swap), invert=dict(invert))


@dataclass
class ProviderConfig:
    """Pose provider settings."""

    name: str = "mediapipe"
    min_confidence: float = 0.2
    model_complexity: int = 1
    model_path: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        return cls(
            name=str(data.get("name", "mediapipe")),
            min_confidence=float(data.get("min_confidence", 0.2)),
            model_complexity=int(data.get("model_complexity", 1)),
            model_path=str(data.get("model_path", "")),
        )


@dataclass
class SmoothingConfig:
    """Smoothing settings."""

    enabled: bool = True
    max_gap: int = 5
    ema_alpha: float = 0.3
    min_confidence: float = 0.2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmoothingConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_gap=int(data.get("max_gap", 5)),
            ema_alpha=float(data.get("ema_alpha", 0.3)),
            min_confidence=float(data.get("min_confidence", 0.2)),
        )


@dataclass
class RetargetConfig:
    """Retargeting settings."""

    min_confidence: float = 0.2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetargetConfig":
        return cls(min_confidence=float(data.get("min_confidence", 0.2)))


@dataclass
class CenterGrooveConfig:
    """Center/groove translation settings."""

    pattern: str = "A"
    lowpass_alpha: float = 0.1
    root_joint: str = "pelvis"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CenterGrooveConfig":
        return cls(
            pattern=str(data.get("pattern", "A")).upper(),
            lowpass_alpha=float(data.get("lowpass_alpha", 0.1)),
            root_joint=str(data.get("root_joint", "pelvis")),
        )


@dataclass
class Config:
    """Top-level configuration."""

    axis_map: AxisMap = field(default_factory=AxisMap)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    center_groove: CenterGrooveConfig = field(default_factory=CenterGrooveConfig)
    bone_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        axis_map = AxisMap.from_dict(data.get("axis_map", {}))
        provider = ProviderConfig.from_dict(data.get("provider", {}))
        smoothing = SmoothingConfig.from_dict(data.get("smoothing", {}))
        retarget = RetargetConfig.from_dict(data.get("retarget", {}))
        center_groove = CenterGrooveConfig.from_dict(data.get("center_groove", {}))
        bone_map = default_bone_map()
        bone_map.update(data.get("bone_map", {}) or {})
        return cls(
            axis_map=axis_map,
            provider=provider,
            smoothing=smoothing,
            retarget=retarget,
            center_groove=center_groove,
            bone_map=bone_map,
        )


def default_bone_map() -> Dict[str, str]:
    """Default PMX bone name mapping."""

    return {
        "center": "センター",
        "groove": "グルーブ",
        "lower_body": "下半身",
        "upper_body": "上半身",
        "upper_body2": "上半身2",
        "neck": "首",
        "head": "頭",
        "left_shoulder": "左肩",
        "left_arm": "左腕",
        "left_elbow": "左ひじ",
        "left_wrist": "左手首",
        "right_shoulder": "右肩",
        "right_arm": "右腕",
        "right_elbow": "右ひじ",
        "right_wrist": "右手首",
        "left_leg": "左足",
        "left_knee": "左ひざ",
        "left_ankle": "左足首",
        "right_leg": "右足",
        "right_knee": "右ひざ",
        "right_ankle": "右足首",
    }


def load_config(path: Optional[str]) -> Config:
    """Load configuration from YAML or JSON file."""

    if not path:
        return Config.from_dict({})

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    return Config.from_dict(data)
