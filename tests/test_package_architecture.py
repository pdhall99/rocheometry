"""Geometry values navigate graphs; checks and sharing belong to computers."""

from dataclasses import dataclass, field
from math import pi
from typing import cast

import pytest

import rocheometry
from rocheometry import Reference
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.section_path import (
    SectionReference,
    reference_centreline_section,
    reference_component_section,
)
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import sphere_volume
from tests._helpers import WholeSection, _Wired, computer_graph, numerical_graph


@dataclass
class CountedL1:
    calls: int = 0

    def __call__(self, field: RocheField, /) -> rocheometry.CriticalPoint:
        self.calls += 1
        return default_l1_computer()(field)


@dataclass
class CountedScalar:
    value: float
    calls: int = 0

    def __call__(self, field: RocheField, /) -> float:
        self.calls += 1
        return self.value


def _closed_path() -> rocheometry.CompoundBezierPath:
    return rocheometry.CompoundBezierPath(
        (
            rocheometry.BezierPath(
                (
                    rocheometry.CubicBezier(
                        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
                    ),
                )
            ),
        )
    )


@dataclass
class SectionProbe:
    path_result: object = field(default_factory=_closed_path)
    calls: int = 0
    supplied_fields: list[RocheField] = field(default_factory=list)

    def __call__(
        self,
        field: RocheField,
        potential: float,
        plane: rocheometry.SectionPlane,
        /,
    ) -> rocheometry.CompoundBezierPath:
        self.calls += 1
        self.supplied_fields.append(field)
        return cast(rocheometry.CompoundBezierPath, self.path_result)

    def envelope(
        self, field: RocheField, potential: float, plane: rocheometry.SectionPlane, /
    ) -> rocheometry.CompoundBezierPath:
        return self(field, potential, plane)


@pytest.mark.parametrize("qi", [1.0e-5, 1.0e5])
def test_custom_section_can_use_a_wider_mass_ratio_domain(qi: float) -> None:
    path = _closed_path()
    calls: list[tuple[RocheField, float]] = []

    def section(
        field: RocheField,
        potential: float,
        plane: rocheometry.SectionPlane,
        /,
    ) -> rocheometry.CompoundBezierPath:
        if not 1.0e-6 <= field.qi <= 1.0e6:
            raise ValueError("outside custom section domain")
        calls.append((field, potential))
        return path

    computers = computer_graph(section_computer=WholeSection(section))
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(computers))
    assert geometry.surface(volume_radius_fill=0.8).section() is path
    assert calls == [(geometry.field, geometry.surface(volume_radius_fill=0.8)._depth)]
    with pytest.raises(ValueError, match="outside custom section domain"):
        rocheometry.ComponentGeometry(
            10000000.0, accuracy=_Wired(computers)
        ).lobe().section()


@pytest.mark.parametrize("qi", [1.0e-5, 1.0e5])
def test_integrated_volumes_do_not_depend_on_fitted_section_domain(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(numerical_graph(polar_order=8, azimuth_order=8))
    )
    lobe_volume = geometry.lobe().volume()
    assert lobe_volume > 0.0
    assert geometry.surface(volume_radius_fill=0.8).volume() == pytest.approx(
        0.8**3 * lobe_volume
    )


def test_fitted_section_reuses_the_cached_l1() -> None:
    l1 = CountedL1()
    geometry = rocheometry.ComponentGeometry(
        0.25, accuracy=_Wired(computer_graph(l1_computer=l1))
    )
    geometry.lobe().section()
    geometry.surface(volume_radius_fill=0.8).section()
    assert l1.calls == 1


@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_fast_calibration_is_separate_from_measured_fill_normalization(
    fill: float,
) -> None:
    lobe = CountedScalar(sphere_volume(0.123))
    geometry = rocheometry.ComponentGeometry(
        1.0, accuracy=_Wired(computer_graph(lobe_volume_computer=lobe))
    )
    surface = geometry.surface(volume_radius_fill=fill)
    assert geometry.lobe_volume_radius() == pytest.approx(0.123, rel=2e-15)
    assert surface.volume() == pytest.approx(
        surface.volume_radius_fill() ** 3 * geometry.lobe().volume()
    )
    if fill < 1:
        assert surface.volume_radius() == pytest.approx(
            fill * geometry.lobe().volume_radius()
        )
    assert 4.0 * pi * surface.volume_radius() ** 3 / 3.0 == pytest.approx(
        surface.volume(), rel=2.0e-15
    )
    assert lobe.calls == 1


def test_measured_normalization_and_sections_share_l1() -> None:
    l1 = CountedL1()

    geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(
            computer_graph(
                l1_computer=l1, lobe_volume_computer=CountedScalar(sphere_volume(0.4))
            )
        ),
    )
    assert geometry.surface(volume_radius_fill=0.8).volume_radius() == pytest.approx(
        0.8 * geometry.lobe().volume_radius()
    )
    assert len(geometry.lobe().section().paths[0].segments) == 52
    assert l1.calls == 1


def test_orbital_reference_is_the_same_result_as_a_centreline_plane() -> None:
    orbital = reference_component_section(
        0.4, 2, 0.8, num_upper_points=9, polar_order=8, azimuth_order=6
    )
    general = reference_centreline_section(
        0.4,
        2,
        0.8,
        rocheometry.ORBITAL_PLANE,
        num_upper_points=9,
        polar_order=8,
        azimuth_order=6,
    )
    assert isinstance(orbital, SectionReference)
    assert orbital == general
    surface = (
        rocheometry.BinaryGeometry(
            0.4, accuracy=_Wired(Reference(polar_order=8, azimuth_order=6)._computers())
        )
        .component(2)
        .intrinsic_geometry.surface(volume_radius_fill=0.8)
    )
    assert orbital.surface_potential == surface.potential
    assert orbital.volume == surface.volume()
    assert orbital.landmarks.inner[0] == surface.named_radius("point")


def test_binary_view_scales_selected_volume_and_radius() -> None:
    computers = computer_graph(lobe_volume_computer=CountedScalar(sphere_volume(0.2)))
    view = (
        rocheometry.BinaryGeometry(0.25, accuracy=_Wired(computers), separation=3.0)
        .component(1)
        .surface(volume_radius_fill=0.8)
    )
    measured_radius = 0.8 * view.component.lobe().volume_radius()
    assert view.component.lobe_volume_radius() == pytest.approx(0.6)
    assert view.volume_radius() == pytest.approx(measured_radius)
    assert view.volume() == pytest.approx(sphere_volume(measured_radius))


def test_direction_normalization_handles_large_finite_vectors() -> None:
    surface = rocheometry.ComponentGeometry(1.0).lobe()
    assert surface.radius((1.0e308, 0.0, 0.0)) == surface.named_radius("point")
