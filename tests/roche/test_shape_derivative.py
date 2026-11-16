"""Analytic level and radial velocities agree with moving Roche lobes."""

from math import exp, sqrt

import pytest

from rocheometry import Reference
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._roche.shape_derivative import ray_log_qi_velocity


@pytest.mark.parametrize("qi", [0.1, 10.0])
def test_l1_potential_slope_obeys_envelope_theorem(qi: float) -> None:
    graph = Reference()._computers()
    field = RocheField(qi)
    l1 = graph.l1(field)
    step = 1e-4
    upper = graph.l1(RocheField(qi * exp(step))).potential.value
    lower = graph.l1(RocheField(qi * exp(-step))).potential.value
    assert field.log_qi_derivative(*l1.position) == pytest.approx(
        (upper - lower) / (2 * step), abs=1e-9, rel=0
    )


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    "direction", [(-1.0, 0.0, 0.0), (1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))]
)
def test_ray_velocity_matches_moving_lobe_intersection(
    qi: float, direction: Point3D
) -> None:
    graph = Reference()._computers()
    field = RocheField(qi)
    l1 = graph.l1(field)
    radius = graph.radius(field, 0.0, direction)
    actual = ray_log_qi_velocity(
        field, field.log_qi_derivative(*l1.position), direction, radius
    )
    step = 1e-4
    upper_field, lower_field = RocheField(qi * exp(step)), RocheField(qi * exp(-step))
    upper = graph.radius(upper_field, 0.0, direction)
    lower = graph.radius(lower_field, 0.0, direction)
    assert actual == pytest.approx((upper - lower) / (2 * step), rel=1e-7, abs=1e-10)
