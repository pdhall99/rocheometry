"""All three fitted selection members share a domain and memoization policy."""

from typing import NoReturn, cast

import pytest

from rocheometry import Fitted, Reference
from rocheometry._computers.composition import Memoized
from rocheometry._computers.lobe_volume_radius.fitted.computer import (
    InterpolatedLobeVolumeRadiusComputer,
    LinearInterpolator,
)
from rocheometry._computers.packaged import load_packaged_surface_potential_map
from rocheometry._computers.surface_selection import (
    FittedSelection,
    SurfaceSelection,
    VolumeInversionSelection,
)
from rocheometry._roche.field import RocheField
from tests._helpers import TestSelection, numerical_graph


@pytest.mark.parametrize("qi", [1e-9, 1e9])
def test_default_selection_falls_back_for_all_three_members(qi: float) -> None:
    graph = Fitted()._computers()
    selection = cast(FittedSelection, graph.selection)
    assert isinstance(selection.fallback, VolumeInversionSelection)
    fallback = selection.fallback
    field = RocheField(qi)
    level = selection.depth(field, 0.8)
    assert selection.fast_lobe_volume(field) == fallback.fast_lobe_volume(field)
    assert level == fallback.depth(field, 0.8)
    assert selection.fill(field, level) == fallback.fill(field, level)
    assert selection.depth(field, 1.0) == 0.0
    assert selection.fill(field, 0.0) == 1.0


def test_narrower_radius_table_selects_the_fallback_for_every_member() -> None:
    fallback = cast(FittedSelection, Fitted()._computers().selection).fallback
    radius = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, 1.0), radii=(0.3, 0.5), interpolator=LinearInterpolator()
    )
    selection = FittedSelection(
        fallback,
        lobe_volume_radius_computer=radius,
        model=load_packaged_surface_potential_map(),
    )
    field = RocheField(1e-3)
    level = fallback.depth(field, 0.8)
    assert selection.fast_lobe_volume(field) == fallback.fast_lobe_volume(field)
    assert selection.depth(field, 0.8) == level
    assert selection.fill(field, level) == fallback.fill(field, level)


def test_forward_selection_stays_fast_and_reported_fill_uses_measurement() -> None:
    def unexpected(*args: object) -> NoReturn:
        raise AssertionError("fitted selection must not evaluate the fallback")

    measured = []

    def fill_measurement(field: RocheField, depth: float, /) -> float:
        measured.append((field, depth))
        return 0.81

    fallback = cast(
        SurfaceSelection, TestSelection(unexpected, unexpected, fill_measurement)
    )
    graph = numerical_graph()
    selection = FittedSelection(fallback, l1_computer=graph.l1, maxsize=1)
    field = RocheField(0.3)
    level = selection.depth(field, 0.8)
    fill = selection.fill(field, level)
    assert fill == 0.81
    assert measured == [(field, level)]
    volume = selection.fast_lobe_volume(field)
    assert selection.depth(RocheField(0.3), 0.8) is level
    assert selection.fill(RocheField(0.3), level) == fill
    assert selection.fast_lobe_volume(RocheField(0.3)) == volume
    for node in (selection.depth, selection.fill, selection.fast_lobe_volume):
        cache = cast(Memoized[..., object], node)
        assert cache.cache_info().hits == cache.cache_info().misses == 1
        assert cache.cache_info().currsize == cache.cache_info().maxsize == 1


def test_reference_graph_keeps_volume_inversion() -> None:
    graph = Reference()._computers()
    assert isinstance(graph.selection, VolumeInversionSelection)
    assert not isinstance(graph.selection, FittedSelection)
