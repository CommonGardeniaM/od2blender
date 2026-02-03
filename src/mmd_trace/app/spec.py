"""Pydantic specs for CLI inputs."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from mmd_trace.retarget.coords import AxisTransform
from mmd_trace.retarget.pipeline import LegIkAxes, LegMode, SolveMode


class AxisTransformSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis_x: float = 1.0
    axis_y: float = 1.0
    axis_z: float = -1.0
    axis_matrix: list[float] | str | None = None
    flip_y: bool = True

    @field_validator("axis_matrix", mode="before")
    @classmethod
    def _parse_axis_matrix(cls, value: object) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                floats = [float(part_value) for part_value in parts]
            except ValueError as exc:
                raise ValueError("axis_matrix must be 9 comma-separated floats") from exc
            if len(floats) != 9:
                raise ValueError("axis_matrix must be 9 comma-separated floats")
            return floats
        if isinstance(value, list):
            if len(value) != 9:
                raise ValueError("axis_matrix must have 9 floats")
            return [float(value_item) for value_item in value]
        raise ValueError("axis_matrix must be a comma-separated string or list of 9 floats")

    def to_internal(self) -> AxisTransform:
        matrix = None
        if self.axis_matrix is not None:
            matrix = np.array(self.axis_matrix, dtype=np.float64).reshape(3, 3)
        return AxisTransform(
            axis_x=self.axis_x,
            axis_y=self.axis_y,
            axis_z=self.axis_z,
            axis_matrix=matrix,
            flip_y=self.flip_y,
        )


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmx: str
    image: str
    out: str
    vis_th: float = 0.2
    det_conf: float = 0.5
    solver: str = SolveMode.ROLL_2D.value
    leg_mode: str = LegMode.IK.value
    leg_ik_axes: str = LegIkAxes.XZ.value
    leg_ik_scale: float = 1.0

    @field_validator("solver")
    @classmethod
    def _validate_solver(cls, value: str) -> str:
        if value not in {SolveMode.ROLL_2D.value, SolveMode.WORLD_3D.value}:
            raise ValueError(f"solver must be one of: {SolveMode.ROLL_2D.value}, {SolveMode.WORLD_3D.value}")
        return value

    @field_validator("leg_mode")
    @classmethod
    def _validate_leg_mode(cls, value: str) -> str:
        if value not in {LegMode.IK.value, LegMode.FK.value}:
            raise ValueError(f"leg_mode must be one of: {LegMode.IK.value}, {LegMode.FK.value}")
        return value

    @field_validator("leg_ik_axes")
    @classmethod
    def _validate_leg_ik_axes(cls, value: str) -> str:
        if value not in {LegIkAxes.X.value, LegIkAxes.XZ.value}:
            raise ValueError(f"leg_ik_axes must be one of: {LegIkAxes.X.value}, {LegIkAxes.XZ.value}")
        return value

    @property
    def solver_mode(self) -> SolveMode:
        return SolveMode(self.solver)

    @property
    def leg_mode_enum(self) -> LegMode:
        return LegMode(self.leg_mode)

    @property
    def leg_ik_axes_enum(self) -> LegIkAxes:
        return LegIkAxes(self.leg_ik_axes)


class DebugSpec(RunSpec):
    model_config = ConfigDict(extra="forbid")

    mode: str = "both"
    project: str = "2d"
    axis_scale: float = 1.0
    dpi: int = 150

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in {"2d", "3d", "both"}:
            raise ValueError("mode must be one of: 2d, 3d, both")
        return value

    @field_validator("project")
    @classmethod
    def _validate_project(cls, value: str) -> str:
        if value not in {"2d", "world"}:
            raise ValueError("project must be one of: 2d, world")
        return value


__all__ = [
    "AxisTransformSpec",
    "RunSpec",
    "DebugSpec",
    "ValidationError",
]
