"""The maximum part fill is continuous, with a kink at equal masses."""

from functools import lru_cache

import pytest
from crosschecks import cartesian

import rocheometry
from rocheometry import Reference
from rocheometry._levels import _levels
from tests._helpers import _Wired


@lru_cache(maxsize=16)
def maximum(qi: float) -> float:
    return rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).max_volume_radius_fill()


@pytest.mark.slow
@pytest.mark.parametrize("side", (-1, 1))
def test_maximum_fill_one_sided_limits(side: int) -> None:
    h = 1e-6
    extrapolated = 2 * maximum(1 + side * h) - maximum(1 + side * 2 * h)
    assert abs(extrapolated - maximum(1.0)) <= 1e-9


@pytest.mark.slow
def test_maximum_fill_slope_jump() -> None:
    centre = maximum(1.0)
    left = tuple((centre - maximum(1 - h)) / h for h in (1e-3, 2e-3))
    right = tuple((maximum(1 + h) - centre) / h for h in (1e-3, 2e-3))
    spread = max(abs(left[1] - left[0]), abs(right[1] - right[0]))
    jump = abs(sum(left) / 2 - sum(right) / 2)
    assert jump > 10 * spread


def test_equal_mass_maximum_fill_against_cartesian_envelope() -> None:
    # At equal masses the plane x = 1/2 divides the L2-level body in half by
    # symmetry, so x-slices give the maximum fill with no dividing surface and
    # no shared integrator: half the envelope over the lobe, both sliced.
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=Reference())
    gap = _levels(geometry.field, geometry._computers).admissibility_gap
    envelope = cartesian.envelope_estimate(geometry.field, gap)
    lobe = cartesian.estimate(
        lambda order: cartesian.lobe_volume(geometry.field, order)
    )
    assert envelope["error"] < 1e-13 and lobe["error"] < 1e-13
    truth = (envelope["value"] / 2 / lobe["value"]) ** (1 / 3)
    assert geometry.max_volume_radius_fill() == pytest.approx(truth, rel=1e-12)
