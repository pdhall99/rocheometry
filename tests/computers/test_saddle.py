"""The shared angular neck width matches the independent saddle formula."""

from math import sqrt

import pytest

import rocheometry
from rocheometry._computers.saddle import _neck_width


@pytest.mark.parametrize(
    "qi,fill,factor",
    (
        (1e6, 0.3, None),
        (0.01, 0.999999, None),
        (1e-3, 0.99, None),
        (1.0, 1.0, None),
        (1.0, None, 1e-6),
        (1e6, None, 0.5),
        (0.01, None, 0.99),
    ),
)
def test_neck_width_matches_written_saddle_formula(
    qi: float, fill: float | None, factor: float | None
) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    surface = (
        geometry.surface(volume_radius_fill=fill)
        if fill is not None
        else geometry.surface(fill_out_factor=factor)
    )
    saddle = geometry.l1()
    expected = sqrt(2 * abs(surface._depth) / saddle.hessian[0][0]) / abs(
        saddle.position[0]
    )
    assert _neck_width(geometry.field, surface._depth, saddle) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )
