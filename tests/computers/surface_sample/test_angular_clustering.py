"""Common-potential physical convergence of the fixed-budget angular maps."""

from math import atan2, fsum, hypot, pi
from typing import TypedDict

import pytest

import rocheometry
from rocheometry._computers.surface_sample.angular import _azimuth_quadrature


class _Case(TypedDict):
    qi: float
    kind: str
    selector: float
    potential: float
    values: dict[str, float]


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1, 1e3, 1e8))
def test_angular_maps_have_ordered_nodes_and_positive_exact_measure(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    azimuths = _azimuth_quadrature(24, 1 - geometry.l1().position[0])
    assert 0 < azimuths[0][0] < azimuths[-1][0] < pi / 2
    assert all(a[0] < b[0] for a, b in zip(azimuths, azimuths[1:]))
    assert all(w > 0 for _, w in azimuths)
    assert fsum(w for _, w in azimuths) == pytest.approx(pi / 2, rel=2e-16)
    for surface in (
        geometry.surface(volume_radius_fill=0.5),
        geometry.lobe(),
        geometry.surface(fill_out_factor=0.01),
    ):
        sample = geometry._computers.sample(geometry.field, surface._depth)
        assert all(w > 0 for _, w in sample.rays)
        for j in range(24):
            theta = [atan2(hypot(d[1], d[2]), d[0]) for d, _ in sample.rays[j::24]]
            assert all(a < b for a, b in zip(theta, theta[1:]))
        if not sample.part:
            assert fsum(w for _, w in sample.rays) == pytest.approx(4 * pi, rel=3e-16)


@pytest.mark.slow
def test_critical_log_q_sweep_has_no_extra_second_difference() -> None:
    values = []
    for j in range(201):
        qi = 10 ** (5 + 3 * j / 200)
        fitted = rocheometry.ComponentGeometry(qi).lobe().integrated_mean_curvature()
        reference = (
            rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
            .lobe()
            .integrated_mean_curvature()
        )
        values.append((fitted, reference))
    for a, b, c in zip(values, values[1:], values[2:]):
        extra = (a[0] - 2 * b[0] + c[0]) - (a[1] - 2 * b[1] + c[1])
        assert abs(extra / b[1]) < 1e-10
