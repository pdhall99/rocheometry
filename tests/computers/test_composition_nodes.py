"""Explicit wiring keeps chosen dependencies and result checks visible."""

from typing import cast

import pytest

import rocheometry
from rocheometry._computers import defaults
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.defaults import default_l1_computer, default_outer_computer
from rocheometry._computers.lobe_slope import SampledLobeSlopeComputer
from rocheometry._computers.mean_curvature import SampledMeanCurvatureComputer
from rocheometry._computers.mean_gravity import SampledMeanGravityComputer
from rocheometry._computers.quadrature import SampledQuadratureComputer
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.silhouette.paths import GaugeFixedSilhouettePathComputer
from rocheometry._computers.surface_area import SampledAreaComputer
from rocheometry._computers.surface_radius import BrentSurfaceRadiusComputer
from rocheometry._computers.surface_sample import GaussLegendreSurfaceSampler
from rocheometry._computers.surface_selection import VolumeInversionSelection
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import sphere_volume
from tests._helpers import RadiusDelegate, _Wired, numerical_graph


def test_custom_l1_is_used_by_all_default_surface_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = default_l1_computer()(RocheField(0.3))
    fields: list[RocheField] = []

    def l1_producer(field: RocheField, /) -> rocheometry.CriticalPoint:
        fields.append(field)
        return point

    def unused_default(field: RocheField, /) -> rocheometry.CriticalPoint:
        raise AssertionError("default downstream methods must use the supplied L1")

    monkeypatch.setattr(defaults, "default_l1_computer", lambda: unused_default)
    l1 = Memoized(l1_producer)
    outer = Memoized(default_outer_computer())
    watershed = TracedWatershedComputer(l1_computer=l1, outer_computer=outer)
    radius = BrentSurfaceRadiusComputer(
        l1_computer=l1, outer_computer=outer, watershed_computer=watershed
    )
    sample = Memoized(
        GaussLegendreSurfaceSampler(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
            polar_order=4,
            azimuth_order=4,
        ),
        maxsize=16,
    )
    volume = SampledVolumeComputer(sample=sample)
    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=l1, outer_computer=outer
    )
    section = GaugeFixedSectionPathComputer(radius_computer=radius, l1_computer=l1)
    computers = ComputerSet(
        watershed=watershed,
        l1=l1,
        outer=outer,
        radius=radius,
        sample=sample,
        volume=volume,
        lobe_slope=Memoized(SampledLobeSlopeComputer(sample=sample, l1_computer=l1)),
        area=Memoized(SampledAreaComputer(sample=sample)),
        mean_curvature=Memoized(SampledMeanCurvatureComputer(sample=sample)),
        mean_gravity=Memoized(SampledMeanGravityComputer(sample=sample)),
        quadrature=Memoized(SampledQuadratureComputer(sample=sample)),
        selection=selection,
        section=section,
        silhouette=GaugeFixedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        ),
    )
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(computers))
    assert geometry.lobe().volume() > 0.0
    assert geometry.lobe().named_radius("side") > 0.0
    surface = geometry.surface(volume_radius_fill=0.8)
    assert surface.potential.value > point.potential.value
    assert surface.named_radius("side") > 0.0
    assert fields == [geometry.field]


def test_custom_radius_drives_lobe_volume_and_detached_selection() -> None:
    l1_fields: list[RocheField] = []
    radius_fields: list[RocheField] = []

    def l1_producer(field: RocheField, /) -> rocheometry.CriticalPoint:
        l1_fields.append(field)
        return default_l1_computer()(field)

    def radius_producer(
        field: RocheField,
        potential: float,
        direction: tuple[float, float, float],
        /,
    ) -> float:
        radius_fields.append(field)
        return 1.0 / (default_l1_computer()(field).potential.value - potential)

    l1 = Memoized(l1_producer, maxsize=2)
    outer = Memoized(default_outer_computer(), maxsize=2)
    watershed = TracedWatershedComputer(l1_computer=l1, outer_computer=outer)
    radius = RadiusDelegate(radius_producer)
    sample = Memoized(
        GaussLegendreSurfaceSampler(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
            polar_order=8,
            azimuth_order=4,
        ),
        maxsize=16,
    )
    volume = SampledVolumeComputer(sample=sample)
    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=l1, outer_computer=outer, maxsize=2
    )
    section = GaugeFixedSectionPathComputer(radius_computer=radius, l1_computer=l1)
    computers = ComputerSet(
        watershed=watershed,
        l1=l1,
        outer=outer,
        radius=radius,
        sample=sample,
        volume=volume,
        lobe_slope=Memoized(SampledLobeSlopeComputer(sample=sample, l1_computer=l1)),
        area=Memoized(SampledAreaComputer(sample=sample)),
        mean_curvature=Memoized(SampledMeanCurvatureComputer(sample=sample)),
        mean_gravity=Memoized(SampledMeanGravityComputer(sample=sample)),
        quadrature=Memoized(SampledQuadratureComputer(sample=sample)),
        selection=selection,
        section=section,
        silhouette=GaugeFixedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        ),
    )
    assert l1_fields == radius_fields == []
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(computers))
    critical = geometry.l1().potential
    assert geometry.lobe().volume() == pytest.approx(
        sphere_volume(1.0 / critical.value)
    )
    assert radius_fields
    before = len(radius_fields)
    surface = geometry.surface(volume_radius_fill=0.8)
    assert surface.volume() == pytest.approx(
        0.8**3 * sphere_volume(1.0 / critical.value)
    )
    assert len(radius_fields) > before
    before = len(radius_fields)
    assert surface.volume() > 0.0
    assert len(radius_fields) == before
    selected = surface.potential
    assert selected.value == pytest.approx(critical.value / 0.8, rel=2.0e-8)
    assert surface.named_radius("side") == pytest.approx(1.0 / selected.value)
    assert l1_fields == [geometry.field]
    assert all(field is geometry.field for field in radius_fields)
    assert (
        cast(Memoized[..., rocheometry.CriticalPoint], computers.l1)
        .cache_info()
        .maxsize
        == 2
    )
    assert (
        cast(Memoized[..., float], computers.selection.fast_lobe_volume)
        .cache_info()
        .maxsize
        == 2
    )
    assert (
        cast(Memoized[..., rocheometry.Potential], computers.selection.depth)
        .cache_info()
        .maxsize
        == 2
    )


def test_custom_volume_bypasses_radius_integration() -> None:
    calls: list[tuple[RocheField, float]] = []

    def volume_producer(field: RocheField, potential: float, /) -> float:
        calls.append((field, potential))
        return sphere_volume(
            1.0 / (default_l1_computer()(field).potential.value - potential)
        )

    def unused_radius(
        field: RocheField,
        potential: float,
        direction: tuple[float, float, float],
        /,
    ) -> float:
        raise AssertionError("custom volume must not use numerical radius integration")

    l1 = Memoized(default_l1_computer())
    outer = Memoized(default_outer_computer())
    watershed = TracedWatershedComputer(l1_computer=l1, outer_computer=outer)
    radius = RadiusDelegate(unused_radius)
    sample = Memoized(
        GaussLegendreSurfaceSampler(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        ),
        maxsize=16,
    )
    volume = volume_producer
    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=l1, outer_computer=outer
    )
    section = GaugeFixedSectionPathComputer(radius_computer=radius, l1_computer=l1)
    computers = ComputerSet(
        watershed=watershed,
        l1=l1,
        outer=outer,
        radius=radius,
        sample=sample,
        volume=volume,
        lobe_slope=Memoized(SampledLobeSlopeComputer(sample=sample, l1_computer=l1)),
        area=Memoized(SampledAreaComputer(sample=sample)),
        mean_curvature=Memoized(SampledMeanCurvatureComputer(sample=sample)),
        mean_gravity=Memoized(SampledMeanGravityComputer(sample=sample)),
        quadrature=Memoized(SampledQuadratureComputer(sample=sample)),
        selection=selection,
        section=section,
        silhouette=GaugeFixedSilhouettePathComputer(
            radius_computer=radius,
            l1_computer=l1,
            outer_computer=outer,
            watershed_computer=watershed,
        ),
    )
    assert calls == []
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(computers))
    critical = geometry.l1().potential
    assert geometry.lobe().volume() == pytest.approx(
        sphere_volume(1.0 / critical.value)
    )
    assert calls == [(geometry.field, 0.0)]
    surface = geometry.surface(volume_radius_fill=0.6)
    level = surface.potential
    assert level.value == pytest.approx(critical.value / 0.6, rel=2.0e-8)
    assert sphere_volume(1.0 / level.value) == pytest.approx(
        surface.volume(), rel=6.0e-8
    )
    before = len(calls)
    assert (
        rocheometry.ComponentGeometry(0.3, accuracy=_Wired(computers))
        .surface(volume_radius_fill=0.6)
        .potential
        == level
    )
    assert len(calls) == before
    assert all(field is geometry.field for field, _ in calls)


@pytest.mark.parametrize("maxsize", [0, 1, None])
def test_integrated_wiring_controls_sharing_capacity(maxsize: int | None) -> None:
    graph = numerical_graph(maxsize=maxsize)
    for qi in (0.3, 0.3, 1.0, 0.3):
        graph.l1(RocheField(qi))
    l1 = cast(Memoized[..., rocheometry.CriticalPoint], graph.l1)
    assert l1.cache_info().misses == (4 if maxsize == 0 else 3 if maxsize == 1 else 2)
    nodes = (
        graph.l1,
        graph.outer,
        graph.lobe_slope,
        graph.area,
        graph.mean_curvature,
        graph.mean_gravity,
        graph.quadrature,
        graph.selection.fast_lobe_volume,
        graph.selection.depth,
    )
    assert all(
        cast(Memoized[..., object], node).cache_info().maxsize == maxsize
        for node in nodes
    )
