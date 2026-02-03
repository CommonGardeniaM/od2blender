from __future__ import annotations

import numpy as np
import pytest

from mmd_trace.app.spec import AxisTransformSpec


def test_axis_transform_matrix() -> None:
    spec = AxisTransformSpec(axis_matrix="1,0,0,0,1,0,0,0,-1")
    axis = spec.to_internal()
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    mapped = axis.apply(points)
    assert mapped.shape == (1, 3)
    assert mapped[0, 0] == pytest.approx(1.0)
    assert mapped[0, 1] == pytest.approx(-2.0)
    assert mapped[0, 2] == pytest.approx(-3.0)


def test_axis_transform_invalid_matrix() -> None:
    with pytest.raises(ValueError):
        AxisTransformSpec(axis_matrix="1,0,0")
