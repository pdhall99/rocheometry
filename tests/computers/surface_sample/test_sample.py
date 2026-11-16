"""Shared quadrature samples preserve ray order, symmetry, and radius work."""

from collections.abc import Callable
from math import atan2, fsum, hypot, pi
from typing import cast
from unittest.mock import patch

import pytest

from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.surface_radius import brent
from rocheometry._computers.surface_sample import (
    DEFAULT_FITTED_AZIMUTH_ORDER,
    DEFAULT_FITTED_POLAR_ORDER,
    GaussLegendreSurfaceSampler,
    SurfaceSample,
    SurfaceSampler,
)
from rocheometry._computers.surface_selection import (
    FittedSelection,
    VolumeInversionSelection,
)
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import Potential
from tests._helpers import RadiusDelegate, _depth, computer_graph


def test_standard_sampler_uses_adjacent_roots_only_on_the_node_path() -> None:
    graph = Reference(polar_order=4, azimuth_order=4)._computers()
    shared = cast(Memoized[..., SurfaceSample], graph.sample)
    shared.cache_clear()
    field = RocheField(1.0)
    with patch.object(brent, "_node_root", wraps=brent._node_root) as node_root:
        sample = graph.sample(field, -0.25)
        assert len(sample.radii) == 16
        assert node_root.call_count == 16
        graph.radius(field, -0.25, (0.0, 1.0, 0.0))
        assert node_root.call_count == 16
    shared.cache_clear()


@pytest.mark.parametrize("orders", [(32, 16), (128, 64)])
@pytest.mark.parametrize("offset", (0.0, 1e-10, 1e-6, 0.01, 1.0))
@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3, 1e4, 1e6))
def test_sampler_quadrant_rays_have_full_sphere_weights_and_ordered_radii(
    orders: tuple[int, int],
    offset: float,
    qi: float,
) -> None:
    field = RocheField(qi)
    l1 = Reference()._computers().l1(field)
    level = Potential.jacobi(l1.potential.value + offset)
    calls: list[Point3D] = []

    def radius(actual: RocheField, potential: float, direction: Point3D, /) -> float:
        assert actual is field and potential == _depth(field, level)
        calls.append(direction)
        return 1.0 + len(calls) / 1000.0

    sampler: SurfaceSampler = GaussLegendreSurfaceSampler(
        RadiusDelegate(radius), *orders
    )
    sample = sampler(field, _depth(field, level))
    assert len(sample.rays) == len(sample.radii) == orders[0] * orders[1]
    assert fsum(weight for _, weight in sample.rays) == pytest.approx(
        4 * pi, abs=1e-13, rel=0
    )
    assert calls == [direction for direction, _ in sample.rays]
    assert sample.radii == tuple(1.0 + i / 1000.0 for i in range(1, len(calls) + 1))
    for direction, weight in sample.rays:
        assert hypot(*direction) == pytest.approx(1.0, abs=3e-16, rel=0)
        assert direction[1] >= 0 and direction[2] >= 0 and weight > 0
    assert min(d[0] for d, _ in sample.rays) < 0 < max(d[0] for d, _ in sample.rays)
    previous = 0.0
    for direction, _ in sample.rays[:: orders[1]]:
        theta = atan2(hypot(direction[1], direction[2]), direction[0])
        assert previous < theta < pi
        previous = theta
    for start in range(0, len(sample.rays), orders[1]):
        previous = 0.0
        for direction, _ in sample.rays[start : start + orders[1]]:
            psi = atan2(direction[2], direction[1])
            assert previous < psi < pi / 2
            previous = psi


@pytest.mark.parametrize("orders", [(32, 16), (128, 64)])
@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e6))
@pytest.mark.parametrize("factor", (0.01, 0.5, 0.99))
def test_contact_polar_angles_increase_and_weights_are_positive(
    orders: tuple[int, int], qi: float, factor: float
) -> None:
    graph = Reference()._computers()
    field = RocheField(qi)
    u1 = graph.l1(field).potential.value
    u2 = max(
        graph.outer(field).potential.value,
        graph.outer(RocheField(1 / qi)).potential.value,
    )
    potential = Potential.jacobi(u1 - factor * (u1 - u2))
    sampler = GaussLegendreSurfaceSampler(
        radius_computer=RadiusDelegate(lambda field, potential, direction: 1.0),
        polar_order=orders[0],
        azimuth_order=orders[1],
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        watershed_computer=graph.watershed,
    )
    sample = sampler(field, _depth(field, potential))
    assert sample.part
    assert len(sample.rays) == orders[0] * orders[1]
    previous_psi = 0.0
    for azimuth_index in range(orders[1]):
        rays = sample.rays[azimuth_index :: orders[1]]
        first = rays[0][0]
        psi = atan2(first[2], first[1])
        assert previous_psi < psi < pi / 2
        previous_psi = psi
        rim = graph.watershed.rim(field, _depth(field, potential), psi)
        previous = atan2(hypot(rim[1], rim[2]), rim[0])
        for direction, weight in rays:
            theta = atan2(hypot(direction[1], direction[2]), direction[0])
            assert previous < theta < pi
            assert weight > 0
            assert atan2(direction[2], direction[1]) == pytest.approx(psi, abs=1e-15)
            previous = theta


def test_lobe_volume_then_sample_requires_only_one_radius_pass() -> None:
    calls = 0

    def radius(field: RocheField, potential: float, direction: Point3D, /) -> float:
        nonlocal calls
        calls += 1
        return 0.25

    graph = computer_graph(
        radius_computer=RadiusDelegate(radius), polar_order=8, azimuth_order=6
    )
    field = RocheField(0.3)
    graph.selection.fast_lobe_volume(field)
    assert calls == 8 * 6
    sample = graph.sample(field, 0.0)
    assert calls == 8 * 6
    assert graph.sample(field, 0.0) is sample
    assert calls == 8 * 6


@pytest.mark.parametrize(
    "factory, orders",
    [
        (
            Fitted()._computers,
            (DEFAULT_FITTED_POLAR_ORDER, DEFAULT_FITTED_AZIMUTH_ORDER),
        ),
        (Reference()._computers, (128, 64)),
    ],
)
def test_standard_volume_reads_the_graph_sample(
    factory: Callable[..., ComputerSet], orders: tuple[int, int]
) -> None:
    graph = factory()
    selection = graph.selection
    if isinstance(selection, FittedSelection):
        selection = selection.fallback
    selection = cast(VolumeInversionSelection, selection)
    assert selection.volume_computer is graph.volume
    volume = cast(
        SampledVolumeComputer, cast(Memoized[..., float], graph.volume).computer
    )
    assert volume.sample is graph.sample
    shared_sample = cast(Memoized[..., SurfaceSample], graph.sample)
    assert shared_sample.cache_info().maxsize == 16
    sampler = cast(
        GaussLegendreSurfaceSampler,
        shared_sample.computer,
    )
    assert (sampler.polar_order, sampler.azimuth_order) == orders


@pytest.mark.parametrize(
    "rays,radii",
    [
        ((((1.0, 0.0, 0.0), 1.0),), ()),
        ((), (1.0,)),
        *[
            ((((1.0, 0.0, 0.0), 1.0),), (r,))
            for r in (0.0, -1.0, float("inf"), float("nan"))
        ],
    ],
)
def test_sample_rejects_invalid_values_at_construction(
    rays: tuple[tuple[tuple[float, float, float], float], ...], radii: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="sample"):
        SurfaceSample(rays, radii)


@pytest.mark.parametrize("name", ["polar_order", "azimuth_order"])
@pytest.mark.parametrize(
    "value, error", [(3, ValueError), (True, TypeError), (8.0, TypeError)]
)
def test_sampler_validates_quadrature_controls(
    name: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=name):
        GaussLegendreSurfaceSampler(
            radius_computer=Fitted()._computers().radius,
            polar_order=cast(int, value) if name == "polar_order" else 32,
            azimuth_order=cast(int, value) if name == "azimuth_order" else 16,
        )
