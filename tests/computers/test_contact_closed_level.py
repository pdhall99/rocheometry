"""Closed readers reject positive depths; part readers bound depth by the gap."""

from math import cos, nextafter, pi, sin, sqrt

import pytest

import rocheometry
from rocheometry._computers.checks import require_closed_depth
from rocheometry._computers.contact import admissibility_depth
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._computers.surface_radius import (
    BisectionSurfaceRadiusComputer,
    BrentSurfaceRadiusComputer,
)
from rocheometry._roche import RocheField
from tests._helpers import RadiusDelegate

QI = (1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6)
DIRECTIONS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) + tuple(
    (
        sqrt(1 - (1 - 2 * (k + 0.5) / 48) ** 2) * cos(k * pi * (3 - sqrt(5))),
        sqrt(1 - (1 - 2 * (k + 0.5) / 48) ** 2) * sin(k * pi * (3 - sqrt(5))),
        1 - 2 * (k + 0.5) / 48,
    )
    for k in range(48)
)


def test_guard_has_no_positive_depth_snapping_band() -> None:
    for depth in (0.0, -nextafter(0.0, 1.0), -1.0):
        assert require_closed_depth(depth) == depth
    for depth in (nextafter(0.0, 1.0), 1e-30, 1e-16, 1.0):
        with pytest.raises(ValueError, match="depth <= 0"):
            require_closed_depth(depth)


@pytest.mark.parametrize(
    "cls", (BrentSurfaceRadiusComputer, BisectionSurfaceRadiusComputer)
)
@pytest.mark.parametrize("qi", QI)
def test_closed_and_contact_depths_solve_every_direction(
    cls: type[BisectionSurfaceRadiusComputer], qi: float
) -> None:
    g = rocheometry.ComponentGeometry(qi)
    computer = cls(
        l1_computer=g._computers.l1,
        outer_computer=g._computers.outer,
        watershed_computer=g._computers.watershed,
    )
    gap = admissibility_depth(g.field, g.l1(), g._computers.outer)
    for depth in (0.0, -1e-6 * gap, 1e-4 * gap):
        assert all(computer(g.field, depth, d) > 0 for d in DIRECTIONS)


@pytest.mark.parametrize(
    "cls", (BrentSurfaceRadiusComputer, BisectionSurfaceRadiusComputer)
)
@pytest.mark.parametrize("qi", QI)
def test_open_depths_raise_in_every_direction(
    cls: type[BisectionSurfaceRadiusComputer], qi: float
) -> None:
    g = rocheometry.ComponentGeometry(qi)
    computer = cls(
        l1_computer=g._computers.l1,
        outer_computer=g._computers.outer,
        watershed_computer=g._computers.watershed,
    )
    gap = admissibility_depth(g.field, g.l1(), g._computers.outer)
    for direction in DIRECTIONS:
        with pytest.raises(ValueError, match="L2 gap"):
            computer(g.field, nextafter(gap, float("inf")), direction)


@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize(
    "cls", (TracedSectionPathComputer, GaugeFixedSectionPathComputer)
)
def test_sections_reject_open_depth_before_solving(
    qi: float,
    cls: type[TracedSectionPathComputer] | type[GaugeFixedSectionPathComputer],
) -> None:
    g = rocheometry.ComponentGeometry(qi)

    def fail_radius(
        field: RocheField, depth: float, direction: tuple[float, float, float], /
    ) -> float:
        raise AssertionError("open depth reached a radius solve")

    computer = cls(
        radius_computer=RadiusDelegate(fail_radius),
        l1_computer=g._computers.l1,
        outer_computer=g._computers.outer,
    )
    gap = admissibility_depth(g.field, g.l1(), g._computers.outer)
    for plane in (
        rocheometry.ORBITAL_PLANE,
        rocheometry.CentrelinePlane(pi / 3),
        rocheometry.AxisNormalPlane("z", 0.05),
    ):
        with pytest.raises(ValueError, match="L2 gap"):
            computer(g.field, nextafter(gap, float("inf")), plane)
