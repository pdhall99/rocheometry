import math
from sys import float_info
from typing import cast

import pytest
from typing_extensions import override

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.surface_potential import (
    BisectionSurfacePotentialComputer,
)
from rocheometry._computers.surface_radius import BisectionSurfaceRadiusComputer
from rocheometry._computers.surface_sample import GaussLegendreSurfaceSampler
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics import bisect_root
from rocheometry._roche import (
    CriticalPoint,
    Potential,
    RocheField,
    volume_radius_from_volume,
)
from rocheometry._validation import normalize_volume_radius_fill
from tests._helpers import RadiusDelegate, _depth, _Wired, numerical_graph


class _ScaledRocheField(RocheField):
    """The component-scaled Roche potential: the same field in other units."""

    @override
    def hessian(
        self, x: float, y: float = 0.0, z: float = 0.0
    ) -> tuple[Point3D, Point3D, Point3D]:
        # Saddle widths use both endpoints, so the scaled chart's Hessian,
        # like its potential and gradient, must use component-scaled units.
        return cast(
            tuple[Point3D, Point3D, Point3D],
            tuple(
                tuple((1 + self.qi) * value for value in row)
                for row in super().hessian(x, y, z)
            ),
        )

    @override
    def potential(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> float:
        r1 = math.sqrt(x * x + y * y + z * z)
        dx2 = x - 1.0
        r2 = math.sqrt(dx2 * dx2 + y * y + z * z)
        return self.qi / r1 + 1.0 / r2 + 0.5 * (1.0 + self.qi) * (x * x + y * y) - x

    @override
    def gradient(
        self,
        x: float,
        y: float = 0.0,
        z: float = 0.0,
    ) -> tuple[float, float, float]:
        r1_sq = x * x + y * y + z * z
        dx2 = x - 1.0
        r2_sq = dx2 * dx2 + y * y + z * z
        r1_cubed = r1_sq * math.sqrt(r1_sq)
        r2_cubed = r2_sq * math.sqrt(r2_sq)
        return (
            -self.qi * x / r1_cubed - dx2 / r2_cubed + (1.0 + self.qi) * x - 1.0,
            -self.qi * y / r1_cubed - y / r2_cubed + (1.0 + self.qi) * y,
            -self.qi * z / r1_cubed - z / r2_cubed,
        )


class _ScaledSurfaceRadiusComputer(BisectionSurfaceRadiusComputer):
    """Radial bracketing and residual scaling in component-scaled units."""

    def __call__(
        self,
        field: RocheField,
        potential: float,
        direction: tuple[float, float, float],
    ) -> float:
        level = self.l1_computer(field).potential.value - potential
        direction_x, direction_y, direction_z = direction
        l1 = self.l1_computer(field)
        inner_radius = min(
            l1.position[0] * 1.0e-12,
            0.25 / max(1.0, abs(level)),
        )
        inner_radius = max(inner_radius, 16.0 * float_info.min)

        def residual(radius: float) -> float:
            return (
                field.potential(
                    radius * direction_x,
                    radius * direction_y,
                    radius * direction_z,
                )
                - level
            )

        for _ in range(self.max_inner_contractions):
            if residual(inner_radius) > 0.0:
                break
            inner_radius *= 0.1
        else:
            raise RuntimeError(
                "could not bracket equipotential near the component centre"
            )

        outer_residual = residual(l1.position[0])
        tolerance = self.relative_tolerance * max(1.0, abs(level))
        if outer_residual > tolerance:
            raise RuntimeError("L1 distance does not bracket this closed equipotential")
        if outer_residual >= 0.0:
            return l1.position[0]
        return bisect_root(residual, inner_radius, l1.position[0])


class _ScaledSurfacePotentialComputer(BisectionSurfacePotentialComputer):
    """The fill-to-potential initial bracket in component-scaled units."""

    def __call__(
        self,
        field: RocheField,
        volume_radius_fill: float,
    ) -> Potential:
        volume_radius_fill = normalize_volume_radius_fill(volume_radius_fill)
        l1 = self.l1_computer(field)
        if volume_radius_fill == 1.0:
            return l1.potential

        target_volume = self.lobe_volume_computer(field) * volume_radius_fill**3
        target_radius = volume_radius_from_volume(target_volume)

        lower_potential = l1.potential.value
        potential_step = max(1.0, 0.125 * abs(l1.potential.value))
        upper_potential = l1.potential.value + potential_step
        upper_volume = self.volume_computer(
            field,
            _depth(field, Potential.jacobi(upper_potential)),
        )
        for _ in range(self.max_bracket_expansions):
            if upper_volume <= target_volume:
                break
            potential_step *= 2.0
            upper_potential = l1.potential.value + potential_step
            upper_volume = self.volume_computer(
                field,
                _depth(field, Potential.jacobi(upper_potential)),
            )
        else:
            raise RuntimeError(
                "could not bracket component-enclosing equipotential potential"
            )

        for _ in range(self.max_iterations):
            midpoint = 0.5 * (lower_potential + upper_potential)
            midpoint_volume = self.volume_computer(
                field,
                _depth(field, Potential.jacobi(midpoint)),
            )
            radius_error = abs(
                volume_radius_from_volume(midpoint_volume) / target_radius - 1.0
            )
            if radius_error <= self.volume_radius_fill_tolerance:
                return Potential.jacobi(midpoint)
            if midpoint_volume > target_volume:
                lower_potential = midpoint
            else:
                upper_potential = midpoint

        raise RuntimeError("component volume-fill solve did not converge")


def _scaled_l1_distance(qi: float) -> float:
    field = _ScaledRocheField(qi)
    return bisect_root(
        lambda x: field.gradient(x)[0],
        1.0e-12,
        1.0 - 1.0e-12,
    )


def _scaled_lobes(
    qi: float,
) -> tuple[RocheField, _ScaledRocheField, CriticalPoint, CriticalPoint]:
    field = RocheField(qi)
    scaled = _ScaledRocheField(qi)
    l1 = default_l1_computer()(field)
    scaled_l1_distance = _scaled_l1_distance(qi)
    return (
        field,
        scaled,
        l1,
        CriticalPoint(
            position=(scaled_l1_distance, 0.0, 0.0),
            potential=Potential.jacobi(scaled.potential(scaled_l1_distance)),
            hessian=cast(
                tuple[Point3D, Point3D, Point3D],
                tuple(
                    tuple((1 + qi) * value for value in row)
                    for row in field.hessian(scaled_l1_distance)
                ),
            ),
        ),
    )


@pytest.mark.parametrize("qi", [1.0e-4, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1.0e4])
def test_bisection_l1_satisfies_kopal_quintic(qi: float) -> None:
    x = default_l1_computer()(RocheField(qi)).position[0]
    residual = (
        (qi + 1.0) * x**5
        - (2.0 * qi + 3.0) * x**4
        + (qi + 3.0) * x**3
        - qi * x**2
        + 2.0 * qi * x
        - qi
    )
    assert residual == pytest.approx(0.0, abs=1.0e-10)


@pytest.mark.parametrize("qi", [1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e8])
def test_l1_location_is_invariant_under_potential_convention(qi: float) -> None:
    expected = default_l1_computer()(RocheField(qi)).position[0]
    actual = _scaled_l1_distance(qi)

    assert actual == pytest.approx(expected, abs=5.0e-13)


@pytest.mark.parametrize("qi", [1.0e-4, 0.01, 0.1, 1.0, 10.0, 100.0, 1.0e4])
def test_bisection_l1_obeys_component_exchange_symmetry(qi: float) -> None:
    computer = default_l1_computer()
    original = computer(RocheField(qi)).position[0]
    swapped = computer(RocheField(1.0 / qi)).position[0]

    assert original == pytest.approx(1.0 - swapped, abs=2.0e-13)


def test_equal_mass_l1_is_midpoint() -> None:
    assert default_l1_computer()(RocheField(1.0)).position[0] == pytest.approx(
        0.5,
        abs=1.0e-14,
    )


@pytest.mark.parametrize("qi", [1.0e-4, 1.0, 1.0e4])
@pytest.mark.parametrize("radius_fraction", [1.0e-4, 1.0e-8, 1.0e-12])
@pytest.mark.parametrize(
    "direction", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
)
def test_surface_radius_retains_relative_accuracy_for_tiny_intersections(
    qi: float, radius_fraction: float, direction: tuple[float, float, float]
) -> None:
    field = RocheField(qi)
    l1 = default_l1_computer()(field)
    expected = l1.position[0] * radius_fraction
    level = Potential.jacobi(field.potential(*(expected * x for x in direction)))
    computer = BisectionSurfaceRadiusComputer(l1_computer=lambda field: l1)

    assert computer(field, _depth(field, level), direction) == pytest.approx(
        expected, rel=2.0e-12, abs=0.0
    )


@pytest.mark.parametrize("qi", [1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e8])
@pytest.mark.parametrize("radius_fraction", [0.5, 1.0e-4])
def test_surface_radius_is_invariant_under_potential_convention(
    qi: float,
    radius_fraction: float,
) -> None:
    field, scaled, l1, scaled_l1 = _scaled_lobes(qi)
    target_radius = l1.position[0] * radius_fraction
    direction = (1.0, 0.0, 0.0)
    computer = BisectionSurfaceRadiusComputer(l1_computer=lambda field: l1)
    scaled_computer = _ScaledSurfaceRadiusComputer(l1_computer=lambda field: scaled_l1)

    radius = computer(
        field,
        _depth(field, Potential.jacobi(field.potential(target_radius))),
        direction,
    )
    scaled_radius = scaled_computer(
        scaled,
        scaled_l1.potential.value - scaled.potential(target_radius),
        direction,
    )

    assert radius == pytest.approx(target_radius, abs=3.0e-14)
    assert scaled_radius == pytest.approx(target_radius, abs=3.0e-14)
    assert radius == pytest.approx(scaled_radius, abs=3.0e-14)


@pytest.mark.parametrize("qi", [1.0e-4, 1.0, 1.0e4])
@pytest.mark.parametrize("volume_radius_fill", [0.2, 0.95])
def test_volume_radius_fill_solve_geometry_is_invariant_under_potential_convention(
    qi: float,
    volume_radius_fill: float,
) -> None:
    field, scaled, l1, scaled_l1 = _scaled_lobes(qi)
    radius_computer = BisectionSurfaceRadiusComputer(l1_computer=lambda field: l1)
    scaled_radius_computer = _ScaledSurfaceRadiusComputer(
        l1_computer=lambda field: scaled_l1
    )
    volume_computer = SampledVolumeComputer(
        sample=GaussLegendreSurfaceSampler(
            radius_computer=radius_computer,
            l1_computer=lambda field: l1,
            polar_order=8,
            azimuth_order=6,
        )
    )
    scaled_volume_computer = SampledVolumeComputer(
        sample=GaussLegendreSurfaceSampler(
            radius_computer=scaled_radius_computer,
            l1_computer=lambda field: scaled_l1,
            polar_order=8,
            azimuth_order=6,
        )
    )
    potential_computer = BisectionSurfacePotentialComputer(
        volume_computer=volume_computer,
        l1_computer=lambda field: l1,
        lobe_volume_computer=lambda field: lobe_volume,
        volume_radius_fill_tolerance=1.0e-11,
    )
    scaled_potential_computer = _ScaledSurfacePotentialComputer(
        volume_computer=scaled_volume_computer,
        l1_computer=lambda field: scaled_l1,
        lobe_volume_computer=lambda field: scaled_lobe_volume,
        volume_radius_fill_tolerance=1.0e-11,
    )

    lobe_volume = volume_computer(
        field,
        0.0,
    )
    scaled_lobe_volume = scaled_volume_computer(
        scaled,
        0.0,
    )
    potential = potential_computer(
        field,
        volume_radius_fill,
    )
    scaled_potential = scaled_potential_computer(
        scaled,
        volume_radius_fill,
    )
    volume = volume_computer(
        field,
        potential,
    )
    scaled_volume = scaled_volume_computer(
        scaled,
        _depth(scaled, scaled_potential),
    )

    assert volume == pytest.approx(scaled_volume, rel=3.0e-10, abs=3.0e-14)


def test_volume_computer_uses_its_constructor_radius_dependency() -> None:
    calls = 0
    levels: list[float] = []

    class CountingRadius:
        def __call__(
            self,
            field: RocheField,
            potential: float,
            direction: tuple[float, float, float],
        ) -> float:
            nonlocal calls
            calls += 1
            levels.append(potential)
            return 0.25

    radius = RadiusDelegate(CountingRadius())
    volume = SampledVolumeComputer(
        sample=GaussLegendreSurfaceSampler(
            radius_computer=radius,
            polar_order=4,
            azimuth_order=4,
        )
    )
    field = RocheField(1.0)
    volume(field, 0.0)
    first_calls = calls
    volume(field, 0.0)

    assert first_calls > 0
    assert calls == 2 * first_calls
    assert all(level == 0.0 for level in levels)


def test_critical_potential_computer_returns_the_l1_potential_object() -> None:
    field = RocheField(0.37)
    l1 = default_l1_computer()(field)

    def unused_volume(field: RocheField, potential: float) -> float:
        raise AssertionError("critical potential does not integrate volume")

    def unused_lobe_volume(field: RocheField) -> float:
        raise AssertionError("critical potential does not resolve lobe volume")

    computer = BisectionSurfacePotentialComputer(
        volume_computer=unused_volume,
        l1_computer=lambda field: l1,
        lobe_volume_computer=unused_lobe_volume,
    )
    assert computer(field, 1.0) == 0.0


def test_potential_inversion_passes_typed_trials_and_returns_the_solved_level() -> None:
    field = RocheField(0.37)
    l1 = default_l1_computer()(field)
    levels: list[float] = []

    def volume(field: RocheField, potential: float) -> float:
        levels.append(potential)
        return (l1.potential.value - potential) ** -3

    computer = BisectionSurfacePotentialComputer(
        volume_computer=volume,
        l1_computer=lambda field: l1,
        lobe_volume_computer=lambda field: l1.potential.value**-3,
    )
    potential = computer(field, 0.8)

    assert l1.potential.value - potential == pytest.approx(
        l1.potential.value / 0.8, rel=2.0e-8
    )
    assert all(isinstance(level, float) and level <= 0 for level in levels)
    assert potential is levels[-1]


def test_default_graph_is_shared_and_variants_have_independent_caches() -> None:
    surface = numerical_graph(
        polar_order=16, azimuth_order=12, volume_radius_fill_tolerance=5e-08
    )
    repeated = numerical_graph(
        polar_order=16, azimuth_order=12, volume_radius_fill_tolerance=5e-08
    )

    assert repeated is not surface
    assert Fitted()._computers() is Fitted()._computers()


def test_integrated_roche_configuration_matches_lobe_volume_computer() -> None:
    geometry = rocheometry.ComponentGeometry(
        0.37, accuracy=_Wired(numerical_graph(polar_order=16, azimuth_order=12))
    )
    computer = numerical_graph(
        polar_order=16, azimuth_order=12
    ).selection.fast_lobe_volume

    assert geometry.lobe().volume() == pytest.approx(computer(RocheField(0.37)))
    l1 = geometry.l1().position[0]
    assert geometry.l1().potential.value == RocheField(0.37).potential(l1)


@pytest.mark.parametrize("volume_radius_fill", [0.2, 0.5, 0.8, 0.95, 1.0])
def test_integrated_surface_recovers_requested_fill(volume_radius_fill: float) -> None:
    geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(
            numerical_graph(
                polar_order=16, azimuth_order=12, volume_radius_fill_tolerance=5e-08
            )
        ),
    )

    assert geometry.surface(volume_radius_fill=volume_radius_fill).volume_radius() / (
        geometry.lobe().volume_radius()
    ) == pytest.approx(volume_radius_fill, abs=5.0e-8)


def test_critical_surface_reuses_lobe_dependencies_exactly() -> None:
    geometry = rocheometry.ComponentGeometry(
        0.2, accuracy=_Wired(numerical_graph(polar_order=12, azimuth_order=8))
    )

    lobe = geometry.lobe()

    assert lobe == geometry.surface(volume_radius_fill=1.0)
    assert lobe.potential == geometry.l1().potential
    assert lobe.volume() > 0.0


def test_surface_geometry_owns_arbitrary_directional_radius() -> None:
    geometry = rocheometry.ComponentGeometry(
        0.6, accuracy=_Wired(numerical_graph(polar_order=12, azimuth_order=8))
    )
    direction = (1.0, 2.0, 1.0)

    critical = geometry.lobe().radius(direction)
    detached = geometry.surface(volume_radius_fill=0.8).radius(direction)

    assert critical > 0.0
    assert detached > 0.0
    assert detached < critical
