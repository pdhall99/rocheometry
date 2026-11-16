"""The opening limit is the nearest representable depth on its closed side."""

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

import rocheometry
from rocheometry._levels import _levels
from rocheometry._roche.opening import _admissibility_gap


@pytest.mark.parametrize(
    "qi", (1e-8, 1e-6, 1e-3, 0.03, 1.0, 20.0, 1e3, 1e6, 1e8, 3.773494668091228e-5)
)
def test_gap_is_the_inward_rounded_exact_difference(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    levels = _levels(g.field, g._computers)
    exact = levels.l1 - levels.opening
    assert (
        Fraction(levels.admissibility_gap)
        <= exact
        < Fraction(math.nextafter(levels.admissibility_gap, math.inf))
    )
    assert float(exact) - levels.admissibility_gap in (0.0, math.ulp(float(exact)))
    assert levels.admissibility_gap == _admissibility_gap(
        g.field,
        g.l1().position[0],
        g.outer().position[0],
        g._computers.outer(g.field.reciprocal()).position[0],
    )


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1.0, 1e6, 1e8))
def test_f_round_trip_with_inward_gap_is_within_one_ulp(
    accuracy: rocheometry.Fitted | rocheometry.Reference, qi: float
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    for factor in (1e-12, 1e-6, 0.01, 0.5, 0.99, math.nextafter(1.0, 0.0)):
        actual = g.surface(fill_out_factor=factor).fill_out_factor()
        assert abs(actual - factor) <= math.ulp(factor)


@pytest.mark.slow
@pytest.mark.parametrize(
    "qi",
    [
        float.fromhex(q)
        for q in json.loads(
            (Path(__file__).parents[1] / "fixtures/maximum-fill-cases.json").read_text()
        )["qi"]
    ],
)
def test_formerly_blocked_high_order_maximum_fill_constructs(qi: float) -> None:
    g = rocheometry.ComponentGeometry(
        qi, accuracy=rocheometry.Reference(polar_order=512, azimuth_order=192)
    )
    value = g.max_volume_radius_fill()
    assert math.isfinite(value) and value > 1
