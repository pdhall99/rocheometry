"""Depth-coordinate reference roots on both sides of L1 and near L2."""

import math

import pytest
from crosschecks.distance.outlines import normal_error
from crosschecks.radii import exact_radius

import rocheometry
from rocheometry._levels import _levels


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_reference_depth_roots_near_the_saddles(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    gap = _levels(g.field, g._computers).admissibility_gap
    directions = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.01, 0.0),
        (-1.0, 0.01, 0.0),
    )
    for factor in (-1e-4, -1e-8, -1e-12, 1e-12, 1e-8, 1e-4, 1 - 1e-12):
        surface = g._surface_at_depth(factor * gap)
        assert surface.regime is (
            rocheometry.LevelRegime.CONTACT
            if factor > 0
            else rocheometry.LevelRegime.DETACHED
        )
        for direction in directions:
            truth = exact_radius(surface, direction)
            assert surface.radius(direction) == pytest.approx(truth, rel=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_detached_neck_panel_resolves_the_nose_without_a_larger_cap(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    gap = _levels(g.field, g._computers).admissibility_gap
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    nose_errors = []
    for factor in (-1e-4, -1e-8, -1e-12):
        surface = g._surface_at_depth(factor * gap)
        nose_errors.append(g.l1().position[0] - surface.radius((1.0, 0.0, 0.0)))
        for name in ("section", "silhouette"):
            path = getattr(surface, name)(plane)
            error, _, _ = normal_error(surface, path, plane, path, name == "silhouette")
            assert error <= 1e-10
    assert all(b < a for a, b in zip(nose_errors[:-1], nose_errors[1:], strict=True))
