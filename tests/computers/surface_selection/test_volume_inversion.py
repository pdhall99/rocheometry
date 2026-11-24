"""Selection contracts keep critical normalization and detached solves together."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.surface_selection import VolumeInversionSelection
from rocheometry._roche.field import RocheField
from tests._helpers import TestSelection, _depth, _Wired, numerical_graph


@pytest.mark.parametrize("qi", [0.3, 1.0, 3.0])
def test_lobe_uses_l1_even_when_custom_selection_returns_wrong_potential(
    qi: float,
) -> None:
    base = Fitted()._computers()
    field = RocheField(qi)
    l1 = base.l1(field)
    wrong_level = rocheometry.Potential.jacobi(l1.potential.value + 1.0)
    calls: list[tuple[RocheField, float]] = []

    def potential(field: RocheField, fill: float, /) -> float:
        calls.append((field, fill))
        return _depth(field, wrong_level)

    geometry = rocheometry.ComponentGeometry(
        qi,
        accuracy=_Wired(
            replace(
                base,
                selection=TestSelection(
                    base.selection.fast_lobe_volume, potential, base.selection.fill
                ),
            )
        ),
    )
    for surface in (geometry.lobe(), geometry.surface(volume_radius_fill=1.0)):
        assert surface.potential == l1.potential
        assert surface.named_radius("point") == l1.position[0]
        back = surface.named_radius("back")
        side = surface.named_radius("side")
        pole = surface.named_radius("pole")
        for position in ((-back, 0.0, 0.0), (0.0, side, 0.0), (0.0, 0.0, pole)):
            assert geometry.field.potential(*position) == pytest.approx(
                l1.potential.value, abs=5.0e-13
            )
    assert calls == []

    assert geometry.surface(volume_radius_fill=0.8).potential == wrong_level
    assert calls == [(geometry.field, 0.8)]


@pytest.mark.parametrize("qi", [0.01, 1.0, 100.0])
def test_selection_shares_critical_volume_and_orders_detached_potentials(
    qi: float,
) -> None:
    calls: list[tuple[RocheField, float]] = []

    def volume(field: RocheField, level: float, /) -> float:
        calls.append((field, level))
        return (default_l1_computer()(field).potential.value - level) ** -3

    l1 = Memoized(default_l1_computer())
    selection = VolumeInversionSelection(volume_computer=volume, l1_computer=l1)
    field = RocheField(qi)
    critical = l1(field).potential
    assert selection.depth(field, 1.0) == 0.0
    assert calls == []
    lobe_volume = selection.fast_lobe_volume(field)
    assert calls == [(field, 0.0)]
    previous = 0.0
    for fill in (0.999999, 0.8, 0.2):
        level = selection.depth(field, fill)
        assert level < previous
        assert critical.value - level == pytest.approx(
            critical.value / fill, rel=2.0e-8
        )
        assert (critical.value - level) ** -3 == pytest.approx(
            lobe_volume * fill**3, rel=6.0e-8
        )
        before = len(calls)
        assert selection.depth(RocheField(qi), fill) is level
        assert selection.fast_lobe_volume(RocheField(qi)) is lobe_volume
        assert len(calls) == before
        previous = level
    assert all(supplied is field for supplied, _ in calls)


def test_integrated_selection_remains_closed_just_below_critical_fill() -> None:
    graph = numerical_graph(polar_order=8, azimuth_order=6)
    selection = cast(VolumeInversionSelection, graph.selection)
    field = RocheField(0.3)
    previous = 0.0
    for fill in (0.999999, 0.99, 0.8):
        level = selection.depth(field, fill)
        assert level < previous
        assert selection.volume_computer(field, level) == pytest.approx(
            fill**3 * selection.fast_lobe_volume(field), rel=6.0e-8
        )
        assert graph.radius(field, level, (1.0, 0.0, 0.0)) < graph.l1(field).position[0]
        previous = level


@pytest.mark.parametrize("qi", [1.0e-4, 1.0, 1.0e4])
@pytest.mark.parametrize("fill", [1.0e-6, 1.0e-8])
def test_integrated_selection_recovers_tiny_fills(qi: float, fill: float) -> None:
    graph = numerical_graph(polar_order=8, azimuth_order=6)
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(volume_radius_fill=fill)
    selection = cast(VolumeInversionSelection, graph.selection)

    measured_volume = selection.volume_computer(geometry.field, surface._depth)
    assert measured_volume == pytest.approx(surface.volume(), rel=6.0e-8, abs=0.0)
    assert surface.named_radius("side") == pytest.approx(
        surface.volume_radius(), rel=6.0e-8, abs=0.0
    )


def test_selections_have_identity_equality_and_independent_bounded_caches() -> None:
    def volume(field: RocheField, level: float, /) -> float:
        return (default_l1_computer()(field).potential.value - level) ** -3

    first = VolumeInversionSelection(
        volume_computer=volume, l1_computer=default_l1_computer(), maxsize=1
    )
    second = replace(first)
    assert first is not second and first != second
    assert len({first, second}) == 2
    lobe = cast(Memoized[..., float], first.fast_lobe_volume)
    potential = cast(Memoized[..., float], first.depth)
    for qi in (0.3, 1.0, 0.3):
        first.depth(RocheField(qi), 0.8)
    assert lobe.cache_info().misses == potential.cache_info().misses == 3
    assert lobe.cache_info().currsize == potential.cache_info().currsize == 1
    assert (
        cast(Memoized[..., float], second.fast_lobe_volume).cache_info().currsize == 0
    )
    assert cast(Memoized[..., float], second.depth).cache_info().currsize == 0


@pytest.mark.parametrize(
    "factory",
    [rocheometry.Fitted()._computers, rocheometry.Reference()._computers],
    ids=["fitted", "reference"],
)
@pytest.mark.parametrize("qi", [0.03, 1.0, 4.0])
def test_fill_potential_round_trips_and_exact_endpoints(
    factory: Callable[[], ComputerSet], qi: float
) -> None:
    graph = factory()
    selection = cast(VolumeInversionSelection, graph.selection)
    field = RocheField(qi)
    critical = graph.l1(field).potential
    assert selection.depth(field, 1.0) == 0.0
    assert selection.fill(field, _depth(field, critical)) == 1.0
    for fill in (0.1, 0.8, 0.999999):
        potential = selection.depth(field, fill)
        assert selection.fill(field, potential) == pytest.approx(
            fill, rel=selection.volume_radius_fill_tolerance, abs=0.0
        )
        surface = rocheometry.ComponentGeometry(
            qi, accuracy=_Wired(graph)
        )._surface_at_depth(potential)
        assert surface.volume_radius_fill() == selection.fill(field, potential)
        repeated = surface.component._surface_at_depth(potential)
        assert surface == repeated
        assert hash(surface) == hash(repeated)


def test_fill_is_memoized_and_l1_bypasses_volume_measurement() -> None:
    calls: list[tuple[RocheField, float]] = []

    def volume(field: RocheField, potential: float, /) -> float:
        calls.append((field, potential))
        return (default_l1_computer()(field).potential.value - potential) ** -3

    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=default_l1_computer(), maxsize=1
    )
    field = RocheField(0.3)
    critical = selection.l1_computer(field).potential
    assert selection.fill(field, _depth(field, critical)) == 1.0
    assert calls == []
    level = selection.depth(field, 0.8)
    calls.clear()
    fill = selection.fill(field, level)
    assert calls == [(field, level)]
    assert selection.fill(RocheField(0.3), level) == fill
    assert len(calls) == 1
    cache = cast(Memoized[..., float], selection.fill)
    assert cache.cache_info().maxsize == cache.cache_info().currsize == 1


def test_contact_inversion_never_selects_the_l2_measurement_endpoint() -> None:
    from math import nextafter

    from rocheometry._computers.contact import admissibility_depth

    graph = Fitted()._computers()
    field = RocheField(1.0)
    gap = admissibility_depth(field, graph.l1(field), graph.outer)

    def volume(field: RocheField, potential: float, /) -> float:
        return 1 + potential / gap

    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=graph.l1, outer_computer=graph.outer
    )
    maximum = selection.fill(field, gap)
    target = nextafter(maximum, 1.0)
    potential = selection.depth(field, target)
    assert potential < gap
    assert selection.fill(field, potential) == pytest.approx(
        target, rel=selection.volume_radius_fill_tolerance
    )
