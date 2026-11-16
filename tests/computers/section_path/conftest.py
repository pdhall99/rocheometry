"""Independent numerical checks shared by the path-reference tests."""

from collections.abc import Callable
from math import atan2, cos, hypot, pi, sin

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.section_path.reference import SectionReference
from rocheometry._computers.surface_radius import BisectionSurfaceRadiusComputer
from rocheometry._numerics.roots import brent_root
from rocheometry._roche.field import RocheField
from tests._helpers import _depth


def assert_reference_samples(reference: SectionReference) -> None:
    field = RocheField(reference.q if reference.component_id == 1 else 1 / reference.q)
    assert isinstance(reference.plane, rocheometry.CentrelinePlane)
    inclination = reference.plane.inclination
    radius = BisectionSurfaceRadiusComputer(l1_computer=Fitted()._computers().l1)
    for x, u in reference.upper:
        angle = atan2(u, x)
        direction = (
            cos(angle),
            sin(angle) * cos(inclination),
            -sin(angle) * sin(inclination),
        )
        exact = radius(field, _depth(field, reference.surface_potential), direction)
        assert (
            abs(hypot(x, u) - exact) / reference.volume_radius
            <= reference.path_tolerance
        )


def assert_reference_top(reference: SectionReference) -> None:
    field = RocheField(reference.q if reference.component_id == 1 else 1 / reference.q)
    assert isinstance(reference.plane, rocheometry.CentrelinePlane)
    inclination = reference.plane.inclination
    graph = Fitted()._computers()

    def point(angle: float) -> tuple[float, float]:
        radius = graph.radius(
            field,
            _depth(field, reference.surface_potential),
            (cos(angle), sin(angle) * cos(inclination), -sin(angle) * sin(inclination)),
        )
        return (radius * cos(angle), radius * sin(angle))

    def residual(angle: float) -> float:
        x, u = point(angle)
        return field.gradient(x, u * cos(inclination), -u * sin(inclination))[0]

    angle = brent_root(residual, pi / 2, pi)
    exact_top = point(angle)
    assert (
        abs(reference.landmarks.top[1] - exact_top[1]) / reference.volume_radius
        <= reference.path_tolerance
    )
    assert reference.landmarks.top[1] == reference.path.bounds[3]


@pytest.fixture
def reference_sample_check() -> Callable[[SectionReference], None]:
    return assert_reference_samples


@pytest.fixture
def reference_top_check() -> Callable[[SectionReference], None]:
    return assert_reference_top
