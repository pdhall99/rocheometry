"""Critical lobe slopes, including independent differences and sample reuse."""

from collections.abc import Callable
from math import exp, log
from typing import cast

import pytest
from tests._helpers import RadiusDelegate, _Wired, computer_graph

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.lobe_slope import (
    SampledLobeSlopeComputer,
)
from rocheometry._computers.surface_sample import SurfaceSampler
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField

_QI_GRID = tuple(10.0**exponent for exponent in range(-6, 7))


@pytest.mark.parametrize("qi", [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4])
def test_reference_slope_matches_richardson_log_radius_difference(qi: float) -> None:
    graph = Reference()._computers()

    def difference(step: float) -> float:
        upper = (
            rocheometry.ComponentGeometry(qi * exp(step), accuracy=_Wired(graph))
            .lobe()
            .volume_radius()
        )
        lower = (
            rocheometry.ComponentGeometry(qi * exp(-step), accuracy=_Wired(graph))
            .lobe()
            .volume_radius()
        )
        return (log(upper) - log(lower)) / (2 * step)

    richardson = (4 * difference(1e-3) - difference(2e-3)) / 3
    actual = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(graph)
    ).lobe_volume_radius_slope()
    assert actual == pytest.approx(richardson, abs=1e-9, rel=0)


@pytest.mark.parametrize(
    "qi, expected",
    [(0.1, 0.293496464073), (1.0, 0.232171739824), (10.0, 0.134294622754)],
)
def test_lobe_slope_regression(qi: float, expected: float) -> None:
    actual = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).lobe_volume_radius_slope()
    assert actual == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("qi", _QI_GRID)
def test_default_lobe_slope_agrees_with_reference(qi: float) -> None:
    actual = rocheometry.ComponentGeometry(qi).lobe_volume_radius_slope()
    reference = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).lobe_volume_radius_slope()
    assert actual == pytest.approx(reference, abs=1e-6, rel=0)


@pytest.mark.parametrize("factory", [Fitted()._computers, Reference()._computers])
def test_lobe_slopes_stay_within_hill_limit_and_decrease_strictly(
    factory: Callable[[], ComputerSet],
) -> None:
    graph = factory()
    slopes = [
        rocheometry.ComponentGeometry(
            qi, accuracy=_Wired(graph)
        ).lobe_volume_radius_slope()
        for qi in _QI_GRID
    ]
    assert all(0 < value < 1 / 3 for value in slopes)
    assert all(left > right for left, right in zip(slopes, slopes[1:]))


def test_lobe_volume_then_slope_needs_no_additional_radius_calls() -> None:
    calls = 0
    reference_radius = Fitted()._computers().radius

    def radius(field: RocheField, potential: float, direction: Point3D, /) -> float:
        nonlocal calls
        calls += 1
        return reference_radius(field, potential, direction)

    graph = computer_graph(
        radius_computer=RadiusDelegate(radius), polar_order=8, azimuth_order=6
    )
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph))
    assert geometry.lobe().volume() > 0
    assert calls == 8 * 6
    assert geometry.lobe_volume_radius_slope() > 0
    assert calls == 8 * 6
    assert geometry.lobe_volume_radius_slope() > 0
    assert calls == 8 * 6


@pytest.mark.parametrize("name", ["sample", "l1_computer"])
def test_sampled_lobe_slope_requires_callable_dependencies(name: str) -> None:
    graph = Fitted()._computers()
    with pytest.raises(TypeError, match=f"^{name}: expected a callable"):
        SampledLobeSlopeComputer(
            sample=cast(SurfaceSampler, None) if name == "sample" else graph.sample,
            l1_computer=cast(CriticalPointComputer, None)
            if name == "l1_computer"
            else graph.l1,
        )
