from dataclasses import dataclass
from math import nextafter, pi, sqrt
from typing import cast

import pytest
from typing_extensions import override

import rocheometry
from crosschecks.cartesian import (
    CartesianSliceSurfaceVolumeComputer,
    bounded_surface_root,
    integrate_cross_section_volume,
)
from rocheometry import Fitted
from rocheometry._accuracy import _Accuracy
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.defaults import default_l1_computer, default_outer_computer
from rocheometry._computers.lobe_slope import SampledLobeSlopeComputer
from rocheometry._computers.lobe_volume import LobeVolumeFromSurfaceComputer
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
from rocheometry._computers.surface_volume import (
    SampledVolumeComputer,
    SurfaceVolumeComputer,
)
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._levels import _levels
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import volume_radius_from_volume


@dataclass(frozen=True, slots=True)
class _Wired(_Accuracy):
    """Accuracy setting that hands the library an explicitly built graph."""

    graph: ComputerSet

    def _computers(self) -> ComputerSet:  # type: ignore[missing-override-decorator]
        return self.graph


def _depth(field: RocheField, potential: rocheometry.Potential) -> float:
    """Return the depth of a public level, as the library converts it."""
    if type(field).potential is not RocheField.potential:
        return default_l1_computer()(field).potential.value - potential.value
    return _levels(field, rocheometry.Reference()._computers()).depth(potential)


class _SphericalField(RocheField):
    """Analytic central field for testing the complete slice strategy."""

    @override
    def potential(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        return 1.0 / sqrt(x * x + y * y + z * z)


@pytest.mark.parametrize("radius", [1.0, 0.2, 1.0e-8])
@pytest.mark.parametrize("order", [16, 17])
def test_slice_strategy_integrates_the_actual_field_at_each_surface_scale(
    radius: float,
    order: int,
) -> None:
    field = _SphericalField(0.3)

    def l1(actual: RocheField, /) -> rocheometry.CriticalPoint:
        assert actual is field
        return rocheometry.CriticalPoint(
            (1.0, 0.0, 0.0),
            rocheometry.Potential.jacobi(1.0),
            ((2.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        )

    computer: SurfaceVolumeComputer = CartesianSliceSurfaceVolumeComputer(
        l1_computer=l1, slice_order=order, cross_section_order=order
    )
    actual = computer(field, 1.0 - 1.0 / radius)

    assert actual == pytest.approx(4.0 * pi * radius**3 / 3.0, rel=5.0e-13, abs=0.0)


@pytest.mark.parametrize("qi", [1.0e-4, 0.01, 0.3, 1.0, 3.0, 100.0, 1.0e4])
@pytest.mark.parametrize(
    "potential_offset", [0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 0.05]
)
@pytest.mark.slow
def test_slice_strategy_agrees_with_independent_angular_integration(
    qi: float,
    potential_offset: float,
) -> None:
    """Compare integrators directly, resolving the detached neck near L1."""
    graph = Fitted()._computers()
    field = RocheField(qi)
    potential = rocheometry.Potential.jacobi(
        graph.l1(field).potential.value * (1.0 + potential_offset)
    )
    cartesian = CartesianSliceSurfaceVolumeComputer(
        l1_computer=graph.l1, slice_order=80, cross_section_order=16
    )
    angular = SampledVolumeComputer(sample=graph.sample)

    reference_volume = cartesian(field, _depth(field, potential))
    actual_volume = angular(field, _depth(field, potential))
    assert actual_volume == pytest.approx(reference_volume, rel=3.0e-8, abs=0.0)
    assert volume_radius_from_volume(actual_volume) == pytest.approx(
        volume_radius_from_volume(reference_volume), rel=1.0e-8, abs=0.0
    )


@pytest.mark.parametrize("fill", [0.8, 0.3, 1.0e-6])
def test_slice_strategy_wires_lobe_normalization_and_detached_inversion(
    fill: float,
) -> None:
    l1 = Memoized(default_l1_computer())
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
        ),
        maxsize=16,
    )
    volume = CartesianSliceSurfaceVolumeComputer(
        l1_computer=l1, slice_order=24, cross_section_order=12
    )
    selection = VolumeInversionSelection(
        volume_computer=volume, l1_computer=l1, outer_computer=outer
    )
    section = GaugeFixedSectionPathComputer(radius_computer=radius, l1_computer=l1)
    graph = ComputerSet(
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
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph))
    surface = geometry.surface(volume_radius_fill=fill)

    assert geometry.lobe().volume() == volume(geometry.field, 0.0)
    assert volume(geometry.field, surface._depth) == pytest.approx(
        surface.volume(), rel=6.0e-8, abs=0.0
    )


@pytest.mark.parametrize("name", ["slice_order", "cross_section_order"])
@pytest.mark.parametrize(
    ("order", "error"), [(3, ValueError), (True, TypeError), (8.0, TypeError)]
)
def test_slice_strategy_validates_quadrature_configuration(
    name: str,
    order: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=name):
        CartesianSliceSurfaceVolumeComputer(
            l1_computer=Fitted()._computers().l1,
            slice_order=cast(int, order) if name == "slice_order" else 16,
            cross_section_order=cast(int, order)
            if name == "cross_section_order"
            else 16,
        )


def test_slice_strategy_requires_an_l1_dependency() -> None:
    with pytest.raises(TypeError, match="l1_computer: expected a callable"):
        CartesianSliceSurfaceVolumeComputer(
            l1_computer=cast(CriticalPointComputer, None)
        )


def test_slice_strategy_rejects_an_open_surface_level() -> None:
    graph = Fitted()._computers()
    field = RocheField(1.0)
    volume = CartesianSliceSurfaceVolumeComputer(l1_computer=graph.l1)

    with pytest.raises(ValueError, match="depth <= 0"):
        volume(
            field,
            _depth(
                field,
                rocheometry.Potential.jacobi(graph.l1(field).potential.value - 0.1),
            ),
        )


def _ellipsoid_volume(
    a: float,
    b: float,
    c: float,
    *,
    slice_order: int = 16,
    cross_section_order: int = 16,
) -> float:
    def y_max(x: float) -> float:
        return b * sqrt(max(0.0, 1.0 - (x / a) ** 2))

    def z_max(x: float, y: float) -> float:
        argument = 1.0 - (x / a) ** 2 - (y / b) ** 2
        return c * sqrt(max(0.0, argument))

    return integrate_cross_section_volume(
        -a,
        a,
        y_max,
        z_max,
        slice_order=slice_order,
        cross_section_order=cross_section_order,
    )


def test_cross_section_quadrature_reproduces_sphere_volume() -> None:
    radius = 1.7

    actual = _ellipsoid_volume(radius, radius, radius)
    expected = 4.0 * pi * radius**3 / 3.0

    assert actual == pytest.approx(expected, rel=2.0e-13)


def test_cross_section_quadrature_reproduces_triaxial_ellipsoid_volume() -> None:
    a, b, c = 2.0, 0.7, 1.3

    actual = _ellipsoid_volume(a, b, c)
    expected = 4.0 * pi * a * b * c / 3.0

    assert actual == pytest.approx(expected, rel=2.0e-13)


def test_bounded_surface_root_finds_known_positive_root() -> None:
    root = bounded_surface_root(
        lambda radius: 2.0 - radius,
        3.0,
        scale=1.0,
    )

    assert root == pytest.approx(2.0, abs=2.0e-14)


def test_bounded_surface_root_allows_root_below_old_artificial_offset() -> None:
    expected = 1.0e-14
    root = bounded_surface_root(
        lambda radius: 1.0 - radius / expected,
        1.0,
        scale=1.0,
    )

    assert root == pytest.approx(expected, rel=2.0e-14, abs=0.0)


def test_independent_q1_radius_exercises_roche_slice_roots() -> None:
    l1 = Fitted()._computers().l1
    volume = CartesianSliceSurfaceVolumeComputer(
        l1_computer=l1,
        slice_order=24,
        cross_section_order=12,
    )
    lobe = LobeVolumeFromSurfaceComputer(volume_computer=volume, l1_computer=l1)
    radius = volume_radius_from_volume(lobe(RocheField(1.0)))

    assert radius == pytest.approx(0.379863240927439, rel=5.0e-12)


@pytest.mark.parametrize("qi", [1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6])
def test_closed_cartesian_reader_rejects_every_positive_depth(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    computer = CartesianSliceSurfaceVolumeComputer(
        l1_computer=g._computers.l1, slice_order=4, cross_section_order=4
    )
    for depth in (nextafter(0.0, 1.0), 1e-30, 1e-16, 0.1):
        with pytest.raises(ValueError, match="depth <= 0"):
            computer(g.field, depth)
