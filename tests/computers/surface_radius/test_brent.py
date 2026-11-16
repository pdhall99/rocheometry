"""Accuracy and integration budgets of the default Brent surface strategies."""

from math import cos, sin, sqrt

import pytest
from typing_extensions import override

import rocheometry
from rocheometry import Reference
from rocheometry._computers.surface_radius import (
    BisectionSurfaceRadiusComputer,
    BrentSurfaceRadiusComputer,
)
from rocheometry._roche.field import RocheField
from tests._helpers import _depth, _Wired, numerical_graph

QI_GRID = (1.0e-4, 0.01, 0.3, 1.0, 3.0, 100.0, 1.0e4)
FILL_GRID = (0.5, 0.9, 0.99, 0.999, 0.99999, 1.0)


@pytest.mark.parametrize("qi", QI_GRID)
@pytest.mark.parametrize("fill", FILL_GRID)
def test_brent_radii_agree_with_bisection_on_all_default_directions(
    qi: float, fill: float
) -> None:
    graph = numerical_graph()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    brent = BrentSurfaceRadiusComputer(l1_computer=graph.l1)
    bisection = BisectionSurfaceRadiusComputer(l1_computer=graph.l1)
    directions = [
        direction
        for direction, _ in graph.sample(
            geometry.field, geometry.surface(volume_radius_fill=fill)._depth
        ).rays
    ]
    directions.extend(
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    for direction in directions:
        assert brent(
            geometry.field, geometry.surface(volume_radius_fill=fill)._depth, direction
        ) == pytest.approx(
            bisection(
                geometry.field,
                geometry.surface(volume_radius_fill=fill)._depth,
                direction,
            ),
            rel=1.0e-12,
            abs=0.0,
        )


@pytest.mark.parametrize("qi", (1e-3, 1.0))
@pytest.mark.parametrize("angle", (2e-8, 1e-7))
def test_outer_critical_cone_rays_meet_the_inward_branch(
    qi: float, angle: float
) -> None:
    graph = Reference()._computers()
    field = RocheField(qi)
    outer = graph.outer(field)
    direction = (-cos(angle), sin(angle), 0.0)
    radius = graph.radius(field, _depth(field, outer.potential), direction)
    assert radius < abs(outer.position[0])
    gradient = field.gradient(*(radius * d for d in direction))
    assert sum(g * d for g, d in zip(gradient, direction, strict=True)) < 0


class _SphereField(RocheField):
    @override
    def potential(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        return 1.0 / sqrt(x * x + y * y + z * z)


@pytest.mark.parametrize("scale", (0.2, 1.0e-8, 1.0e-12))
def test_brent_radius_preserves_custom_fields_and_tiny_intersections(
    scale: float,
) -> None:
    field = _SphereField(0.3)

    def l1(actual: RocheField, /) -> rocheometry.CriticalPoint:
        assert actual is field
        return rocheometry.CriticalPoint(
            (1.0, 0.0, 0.0),
            rocheometry.Potential.jacobi(1.0),
            ((2.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        )

    computer = BrentSurfaceRadiusComputer(l1)
    assert computer(field, 1.0 - 1.0 / scale, (0.0, 1.0, 0.0)) == pytest.approx(
        scale, rel=1.0e-12, abs=0.0
    )


@pytest.mark.parametrize(
    "cls", [BrentSurfaceRadiusComputer, BisectionSurfaceRadiusComputer]
)
def test_ray_solves_reject_open_surfaces(
    cls: type[BisectionSurfaceRadiusComputer],
) -> None:
    graph = numerical_graph()
    field = RocheField(0.3)
    computer = cls(graph.l1)
    level = rocheometry.Potential.jacobi(
        min(
            graph.outer(field).potential.value,
            graph.outer(RocheField(1 / field.qi)).potential.value,
        )
        - 0.1
    )
    with pytest.raises(ValueError, match="L2 gap"):
        computer(field, _depth(field, level), (1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "cls", [BrentSurfaceRadiusComputer, BisectionSurfaceRadiusComputer]
)
def test_unexpected_closed_bracket_failure_remains_a_runtime_error(
    cls: type[BisectionSurfaceRadiusComputer],
) -> None:
    class _EllipsoidField(_SphereField):
        def potential(self, x: float, y: float = 0.0, z: float = 0.0) -> float:  # type: ignore[missing-override-decorator]
            return 1 / sqrt(x * x + y * y / 4 + z * z)

    field = _EllipsoidField(0.3)

    def incorrect_l1(actual: RocheField, /) -> rocheometry.CriticalPoint:
        assert actual is field
        return rocheometry.CriticalPoint(
            (0.1, 0.0, 0.0),
            rocheometry.Potential.jacobi(1.0),
            ((2000.0, 0.0, 0.0), (0.0, -1000.0, 0.0), (0.0, 0.0, -1000.0)),
        )

    # The supplied axial bound is too small for the ellipsoid's y radius.
    computer = cls(l1_computer=incorrect_l1)
    with pytest.raises(RuntimeError, match="L1 distance does not bracket"):
        computer(field, 0.0, (0.0, 1.0, 0.0))
