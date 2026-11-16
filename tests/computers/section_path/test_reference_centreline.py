import math
from collections.abc import Callable, Iterator
from typing import cast

import pytest

import rocheometry._computers.defaults as reference_wiring
from rocheometry import ORBITAL_PLANE, CentrelinePlane
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.critical_point import BisectionCriticalPointComputer
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.lobe_volume.from_surface import (
    LobeVolumeFromSurfaceComputer,
)
from rocheometry._computers.section_path import (
    SectionReference,
    reference_centreline_section,
    reference_component_section,
)
from rocheometry._roche import CriticalPoint, RocheField


def _settings(polar: int, azimuth: int, tolerance: float) -> reference_wiring._Settings:
    return reference_wiring._Settings(
        polar_order=polar,
        azimuth_order=azimuth,
        fill_tolerance=tolerance,
        watershed_tolerance=1e-13,
        fitted_selection=False,
        exact_anchor_everywhere=True,
        fixed_paths=False,
        path_tolerance=1e-10,
    )


@pytest.fixture
def reference_computer_cache() -> Iterator[None]:
    reference_wiring._cached_reference.cache_clear()
    yield
    reference_wiring._cached_reference.cache_clear()


def test_reference_graph_cache_evicts_least_recently_used_configurations(
    reference_computer_cache: None,
) -> None:
    cache = reference_wiring._cached_reference
    capacity = cache.cache_info().maxsize
    assert capacity is not None and capacity >= 2
    graphs = [
        cache(_settings(8, 6, index * 1.0e-8)) for index in range(1, capacity + 1)
    ]
    assert cache(_settings(8, 6, 1.0e-8)) is graphs[0]
    cache(_settings(8, 6, (capacity + 1) * 1.0e-8))
    assert cache.cache_info().currsize == capacity
    assert cache(_settings(8, 6, 1.0e-8)) is graphs[0]
    assert cache(_settings(8, 6, 2.0e-8)) is not graphs[1]
    assert cache.cache_info().currsize == capacity


def test_references_across_fills_and_planes_reuse_one_l1_and_lobe_solve(
    reference_computer_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    l1_calls: list[RocheField] = []
    outer_calls: list[RocheField] = []
    lobe_calls: list[RocheField] = []
    original_l1 = BisectionCriticalPointComputer.__call__
    original_lobe = LobeVolumeFromSurfaceComputer.__call__

    def l1(
        computer: BisectionCriticalPointComputer, field: RocheField
    ) -> CriticalPoint:
        (outer_calls if computer.branch == "outer" else l1_calls).append(field)
        return original_l1(computer, field)

    def lobe(computer: LobeVolumeFromSurfaceComputer, field: RocheField) -> float:
        lobe_calls.append(field)
        return original_lobe(computer, field)

    monkeypatch.setattr(BisectionCriticalPointComputer, "__call__", l1)
    monkeypatch.setattr(LobeVolumeFromSurfaceComputer, "__call__", lobe)
    critical = reference_component_section(
        0.5, 1, 1.0, num_upper_points=9, polar_order=8, azimuth_order=6
    )
    for fill, plane in ((0.7, 0.0), (0.8, 0.5), (0.9, 1.0)):
        section = reference_centreline_section(
            0.5,
            1,
            fill,
            CentrelinePlane(plane),
            num_upper_points=17,
            polar_order=8,
            azimuth_order=6,
        )
        assert section.lobe_volume == critical.lobe_volume
        assert section.volume_radius / section.lobe_volume_radius == pytest.approx(fill)
    exchanged = reference_component_section(
        2.0, 2, 0.8, num_upper_points=9, polar_order=8, azimuth_order=6
    )
    assert exchanged.lobe_volume == critical.lobe_volume
    assert l1_calls == lobe_calls == [RocheField(0.5)]
    assert outer_calls == [RocheField(0.5), RocheField(0.5).reciprocal()]


@pytest.mark.parametrize(
    ("polar_order", "azimuth_order", "tolerance"),
    [(10, 6, 2.0e-8), (8, 8, 2.0e-8), (8, 6, 1.0e-7)],
)
def test_reference_graph_cache_separates_numerical_controls(
    reference_computer_cache: None,
    monkeypatch: pytest.MonkeyPatch,
    polar_order: int,
    azimuth_order: int,
    tolerance: float,
) -> None:
    graphs: list[ComputerSet] = []
    original = reference_wiring._cached_reference.__wrapped__

    def assemble(settings: reference_wiring._Settings) -> ComputerSet:
        computers = original(settings)
        graphs.append(computers)
        return computers

    from functools import lru_cache

    monkeypatch.setattr(
        reference_wiring, "_cached_reference", lru_cache(maxsize=8)(assemble)
    )
    for polar, azimuth, fill_tolerance in (
        (8, 6, 2.0e-8),
        (polar_order, azimuth_order, tolerance),
    ):
        first = reference_component_section(
            0.5,
            1,
            1.0,
            num_upper_points=9,
            polar_order=polar,
            azimuth_order=azimuth,
            volume_radius_fill_tolerance=fill_tolerance,
        )
        second = reference_component_section(
            0.5,
            1,
            1.0,
            num_upper_points=9,
            polar_order=polar,
            azimuth_order=azimuth,
            volume_radius_fill_tolerance=fill_tolerance,
        )
        assert second == first
    assert len(graphs) == 2
    assert graphs[0] is not graphs[1]


@pytest.mark.parametrize(
    "control", ["polar_order", "azimuth_order", "volume_radius_fill_tolerance"]
)
def test_reference_controls_are_checked_before_a_warm_cache_lookup(
    reference_computer_cache: None, control: str
) -> None:
    reference_component_section(
        0.5,
        1,
        1.0,
        num_upper_points=9,
        polar_order=8,
        azimuth_order=6,
        volume_radius_fill_tolerance=1.0,
    )
    polar = 8.0 if control == "polar_order" else 8
    azimuth = 6.0 if control == "azimuth_order" else 6
    tolerance = True if control == "volume_radius_fill_tolerance" else 1.0
    with pytest.raises(TypeError, match=control):
        reference_component_section(
            0.5,
            1,
            1.0,
            num_upper_points=9,
            polar_order=cast(int, polar),
            azimuth_order=cast(int, azimuth),
            volume_radius_fill_tolerance=tolerance,
        )


@pytest.mark.parametrize("volume_radius_fill", [0.83, 1.0])
def test_orbital_plane_boundary_matches_existing_reference_exactly(
    volume_radius_fill: float,
) -> None:
    section = reference_centreline_section(
        0.37,
        1,
        volume_radius_fill,
        ORBITAL_PLANE,
        num_upper_points=33,
        polar_order=16,
        azimuth_order=12,
    )
    orbital = reference_component_section(
        0.37,
        1,
        volume_radius_fill,
        num_upper_points=33,
        polar_order=16,
        azimuth_order=12,
    )

    assert section.upper == orbital.upper
    assert section.landmarks.inner == orbital.landmarks.inner
    assert section.landmarks.top == orbital.landmarks.top
    assert section.landmarks.back == orbital.landmarks.back
    assert section.surface_potential == orbital.surface_potential
    assert section.volume == orbital.volume
    assert section.lobe_volume == orbital.lobe_volume


@pytest.mark.parametrize("inclination", [0.0, math.pi / 6, math.pi / 3, math.pi / 2])
def test_inclined_section_lies_on_one_equipotential(
    inclination: float, reference_sample_check: Callable[[SectionReference], None]
) -> None:
    section = reference_centreline_section(
        0.2,
        1,
        0.8,
        CentrelinePlane(inclination),
        num_upper_points=25,
        polar_order=12,
        azimuth_order=8,
        volume_radius_fill_tolerance=1.0e-7,
    )
    reference_sample_check(section)


@pytest.mark.parametrize("inclination", [0.0, math.pi / 6, math.pi / 3])
def test_critical_inclined_section_starts_exactly_at_l1(
    inclination: float,
) -> None:
    qi = 0.4
    section = reference_centreline_section(
        qi,
        1,
        1.0,
        CentrelinePlane(inclination),
        num_upper_points=17,
        polar_order=12,
        azimuth_order=8,
    )
    l1_x = default_l1_computer()(RocheField(qi)).position[0]

    assert section.upper[0] == (l1_x, 0.0)
    assert section.landmarks.inner == (l1_x, 0.0)


def test_inclined_top_is_true_maximum_transverse_landmark(
    reference_top_check: Callable[[SectionReference], None],
) -> None:
    inclination = math.acos(0.6)
    section = reference_centreline_section(
        1.0,
        1,
        0.8,
        CentrelinePlane(inclination),
        num_upper_points=65,
        polar_order=12,
        azimuth_order=8,
        volume_radius_fill_tolerance=1.0e-7,
    )
    reference_top_check(section)
    top_u = section.landmarks.top[1]
    assert top_u >= max(u for _, u in section.upper) - 1.0e-13


def test_component_exchange_preserves_inclined_local_geometry() -> None:
    first = reference_centreline_section(
        0.125,
        1,
        0.83,
        CentrelinePlane(0.4),
        num_upper_points=25,
        polar_order=12,
        azimuth_order=8,
        volume_radius_fill_tolerance=1.0e-7,
    )
    exchanged = reference_centreline_section(
        8.0,
        2,
        0.83,
        CentrelinePlane(0.4),
        num_upper_points=25,
        polar_order=12,
        azimuth_order=8,
        volume_radius_fill_tolerance=1.0e-7,
    )

    assert first.upper == exchanged.upper
    assert first.landmarks == exchanged.landmarks
    assert first.surface_potential == exchanged.surface_potential
    assert first.volume == exchanged.volume
    assert first.lobe_volume == exchanged.lobe_volume


@pytest.mark.parametrize("inclination", [math.inf, math.nan])
def test_reference_centreline_section_rejects_invalid_inclination(
    inclination: float,
) -> None:
    with pytest.raises(ValueError, match="inclination"):
        reference_centreline_section(
            1.0,
            1,
            0.8,
            CentrelinePlane(inclination),
            num_upper_points=9,
            polar_order=8,
            azimuth_order=8,
        )


def test_reference_centreline_section_rejects_bool_inclination() -> None:
    with pytest.raises(TypeError, match="inclination: expected a real number"):
        reference_centreline_section(
            1.0,
            1,
            0.8,
            CentrelinePlane(cast(float, True)),
            num_upper_points=9,
            polar_order=8,
            azimuth_order=8,
        )
