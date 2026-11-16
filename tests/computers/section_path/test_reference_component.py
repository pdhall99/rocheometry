import math
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from typing import cast

import pytest
from typing_extensions import override

import rocheometry._computers.section_path.reference as reference_module
from rocheometry import ORBITAL_PLANE
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.section_path import (
    SectionPathComputer,
    SectionReference,
    reference_component_section,
)
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._geometry import Point3D
from rocheometry._roche import RocheField
from tests._helpers import numerical_graph, single_path


class _ScaledRocheField(RocheField):
    """The component-scaled field, as the exact affine transform of Jacobi U."""

    @override
    def potential(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> float:
        if y == 0.0 and z == 0.0:
            return float(self._axis_potential_exact(x))
        return (1.0 + self.qi) * super().potential(x, y, z) - 0.5 / (1.0 + self.qi)

    @lru_cache(maxsize=4096)
    @override
    def _axis_potential_exact(self, x: float) -> Fraction:
        scale = Fraction(1.0 + self.qi)
        return scale * super()._axis_potential_exact(x) - Fraction(0.5) / scale

    @override
    def gradient(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> tuple[float, float, float]:
        scale = 1.0 + self.qi
        gx, gy, gz = super().gradient(x, y, z)
        return (scale * gx, scale * gy, scale * gz)

    @override
    def hessian(
        self, x: float, y: float = 0.0, z: float = 0.0
    ) -> tuple[Point3D, Point3D, Point3D]:
        scale = 1.0 + self.qi
        a, b, c = super().hessian(x, y, z)
        return (
            (scale * a[0], scale * a[1], scale * a[2]),
            (scale * b[0], scale * b[1], scale * b[2]),
            (scale * c[0], scale * c[1], scale * c[2]),
        )


def test_integrated_strategy_samples_closed_component_local_section() -> None:
    computers = numerical_graph(
        polar_order=8, azimuth_order=8, volume_radius_fill_tolerance=1e-7
    )
    reference = TracedSectionPathComputer(
        radius_computer=computers.radius,
        l1_computer=computers.l1,
    )
    computer: SectionPathComputer = reference
    path = single_path(
        computer(
            RocheField(0.4),
            computers.selection.depth(RocheField(0.4), 0.8),
            ORBITAL_PLANE,
        )
    )
    points = path.sample(17)
    assert computer is reference
    assert path.closed
    assert len(points) == 17
    assert points[0] == points[-1]


@pytest.mark.parametrize("operation", ["volume", "sampler"])
def test_integrated_strategies_use_the_supplied_field_throughout(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplied_field = RocheField(0.43)
    evaluated_fields: list[RocheField] = []
    original_axis = RocheField._axis_potential_exact

    def axis(field: RocheField, x: float) -> Fraction:
        evaluated_fields.append(field)
        return original_axis(field, x)

    monkeypatch.setattr(RocheField, "_axis_potential_exact", axis)
    if operation == "volume":
        computer = numerical_graph(
            polar_order=8, azimuth_order=6
        ).selection.fast_lobe_volume
        assert computer(supplied_field) > 0.0
    else:
        computers = numerical_graph(polar_order=8, azimuth_order=6)
        computer = TracedSectionPathComputer(
            radius_computer=computers.radius,
            l1_computer=computers.l1,
        )
        points = single_path(
            computer(
                supplied_field,
                computers.selection.depth(supplied_field, 0.8),
                ORBITAL_PLANE,
            )
        ).sample(9)
        assert points[0] == points[-1]
        assert len(points) == 9

    assert evaluated_fields
    assert all(field is supplied_field for field in evaluated_fields)


def test_volume_radius_fill_one_is_exact_critical_surface() -> None:
    section = reference_component_section(
        0.37,
        1,
        1.0,
        num_upper_points=65,
        polar_order=16,
        azimuth_order=12,
    )
    field = RocheField(0.37)
    l1 = default_l1_computer()(field)
    l1_x = l1.position[0]

    assert section.surface_potential == l1.potential
    assert section.landmarks.inner == (l1_x, 0.0)
    assert section.upper[0] == (l1_x, 0.0)
    assert section.volume == section.lobe_volume
    assert section.volume_radius / section.lobe_volume_radius == 1.0


@pytest.mark.parametrize("volume_radius_fill", [0.2, 0.5, 0.8, 0.95, 0.99])
def test_requested_volume_radius_fill_is_recovered(volume_radius_fill: float) -> None:
    section = reference_component_section(
        0.4,
        1,
        volume_radius_fill,
        num_upper_points=17,
        polar_order=16,
        azimuth_order=12,
        volume_radius_fill_tolerance=5.0e-8,
    )

    recovered = section.volume_radius / section.lobe_volume_radius
    assert recovered == pytest.approx(volume_radius_fill, abs=5.0e-8)


def test_reference_section_geometry_is_invariant_to_scaled_potential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_radius_fill = 0.83
    volume_radius_fill_tolerance = 5.0e-8

    def solve() -> SectionReference:
        return reference_component_section(
            0.43,
            1,
            volume_radius_fill,
            num_upper_points=33,
            polar_order=12,
            azimuth_order=8,
            volume_radius_fill_tolerance=volume_radius_fill_tolerance,
        )

    expected = solve()

    monkeypatch.setattr(reference_module, "RocheField", _ScaledRocheField)
    actual = solve()
    assert actual.surface_potential.value == pytest.approx(
        (1.0 + actual.q) * expected.surface_potential.value - 0.5 / (1.0 + actual.q),
        abs=2.0e-6,
    )

    expected_fill = expected.volume_radius / expected.lobe_volume_radius
    actual_fill = actual.volume_radius / actual.lobe_volume_radius
    assert expected_fill == pytest.approx(
        volume_radius_fill, abs=volume_radius_fill_tolerance
    )
    assert actual_fill == pytest.approx(
        volume_radius_fill, abs=volume_radius_fill_tolerance
    )
    assert actual_fill == pytest.approx(
        expected_fill, abs=2.0 * volume_radius_fill_tolerance
    )

    # The affine potential changes the fill solve's bisection steps, so the
    # two valid surfaces can stop at different points inside the same fill
    # tolerance. Compare geometry at that physical accuracy rather than roundoff.
    geometry_scale = max(
        expected.lobe_volume_radius,
        actual.lobe_volume_radius,
    )
    geometry_tolerance = 2.0 * volume_radius_fill_tolerance * geometry_scale

    for actual_point, expected_point in zip(
        actual.upper,
        expected.upper,
        strict=True,
    ):
        assert math.dist(actual_point, expected_point) <= geometry_tolerance
    assert math.dist(actual.landmarks.inner, expected.landmarks.inner) <= (
        geometry_tolerance
    )
    assert math.dist(actual.landmarks.top, expected.landmarks.top) <= (
        geometry_tolerance
    )
    assert math.dist(actual.landmarks.back, expected.landmarks.back) <= (
        geometry_tolerance
    )


def test_component_section_lies_on_one_equipotential(
    reference_sample_check: Callable[[SectionReference], None],
) -> None:
    section = reference_component_section(
        0.08,
        1,
        0.9,
        num_upper_points=33,
        polar_order=16,
        azimuth_order=12,
    )

    reference_sample_check(section)


def test_component_exchange_preserves_section_geometry() -> None:
    first = reference_component_section(
        0.125,
        1,
        0.83,
        num_upper_points=33,
        polar_order=16,
        azimuth_order=12,
    )
    exchanged = reference_component_section(
        8.0,
        2,
        0.83,
        num_upper_points=33,
        polar_order=16,
        azimuth_order=12,
    )

    assert first.upper == exchanged.upper
    assert first.landmarks == exchanged.landmarks
    assert first.surface_potential == exchanged.surface_potential
    assert first.volume == exchanged.volume
    assert first.lobe_volume == exchanged.lobe_volume


def test_section_top_is_true_maximum_y_landmark(
    reference_top_check: Callable[[SectionReference], None],
) -> None:
    section = reference_component_section(
        1.0,
        1,
        0.8,
        num_upper_points=65,
        polar_order=16,
        azimuth_order=12,
    )
    reference_top_check(section)
    top_y = section.landmarks.top[1]
    assert top_y >= max(y for _, y in section.upper) - 1.0e-13


def test_low_fill_surface_is_nearly_spherical() -> None:
    section = reference_component_section(
        1.0,
        1,
        0.1,
        num_upper_points=65,
        polar_order=12,
        azimuth_order=8,
        volume_radius_fill_tolerance=1.0e-7,
    )
    radii = tuple(math.hypot(x, y) for x, y in section.upper)

    assert (max(radii) - min(radii)) / section.volume_radius < 2.0e-3


def test_lobe_volume_quadrature_converges() -> None:
    coarse = numerical_graph(
        polar_order=24, azimuth_order=16
    ).selection.fast_lobe_volume(RocheField(1.0))
    fine = numerical_graph(polar_order=32, azimuth_order=20).selection.fast_lobe_volume(
        RocheField(1.0)
    )

    assert abs((coarse / fine) ** (1.0 / 3.0) - 1.0) < 2.0e-5


@pytest.mark.parametrize("volume_radius_fill", [0.0, -0.1, math.inf, math.nan])
def test_reference_component_section_rejects_invalid_volume_radius_fill(
    volume_radius_fill: float,
) -> None:
    with pytest.raises(ValueError, match="volume_radius_fill"):
        reference_component_section(
            1.0,
            1,
            volume_radius_fill,
            polar_order=8,
            azimuth_order=8,
        )


def test_reference_component_section_rejects_bool_volume_radius_fill() -> None:
    with pytest.raises(TypeError, match="volume_radius_fill: expected a real number"):
        reference_component_section(
            1.0,
            1,
            cast(float, True),
            polar_order=8,
            azimuth_order=8,
        )


@pytest.mark.parametrize("order", [3, 0, -1])
def test_reference_component_section_rejects_invalid_quadrature_order(
    order: int,
) -> None:
    with pytest.raises(ValueError, match="polar_order"):
        reference_component_section(
            1.0,
            1,
            0.8,
            polar_order=order,
            azimuth_order=8,
        )


@pytest.mark.parametrize("order", [True, 4.5])
def test_reference_component_section_rejects_wrong_quadrature_order_type(
    order: object,
) -> None:
    with pytest.raises(TypeError, match="polar_order"):
        reference_component_section(
            1.0,
            1,
            0.8,
            polar_order=cast(int, order),
            azimuth_order=8,
        )
