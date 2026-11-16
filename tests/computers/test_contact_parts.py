"""Watershed part integrals and the contact-level public API."""

import pickle
from math import atan2, cos, fsum, hypot, pi, sin, sqrt
from typing import cast

import pytest
from crosschecks.cartesian import envelope_volume

import rocheometry
from rocheometry import Reference
from rocheometry._geometry import Point3D
from rocheometry._numerics.quadrature import gauss_legendre
from tests._helpers import _opening_level as outer_level
from tests._helpers import _Wired
from tests.computers.watershed.test_traced import uphill_basin

QI = (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6)
FACTORS = (1e-6, 0.01, 0.5, 0.99)


@pytest.mark.parametrize("qi", QI)
def test_contact_selectors_limits_and_pickle(qi: float) -> None:
    graph = Reference(polar_order=32, azimuth_order=16)._computers()
    geometry = rocheometry.ComponentGeometry(
        qi, accuracy=Reference(polar_order=32, azimuth_order=16)
    )
    surface = geometry.surface(fill_out_factor=0.5)
    assert surface.regime is rocheometry.LevelRegime.CONTACT
    reported = geometry.surface(potential=surface.potential)
    assert reported.potential == surface.potential
    assert geometry.surface(fill_out_factor=surface.fill_out_factor()) == surface
    assert geometry.surface(fill_out_factor=0.6) != surface
    assert surface.fill_out_factor() == pytest.approx(0.5, abs=1e-9)
    assert geometry.surface(fill_out_factor=0) == geometry.lobe()
    assert (
        geometry.surface(fill_out_factor=-1).regime is rocheometry.LevelRegime.DETACHED
    )
    maximum = geometry.max_volume_radius_fill()
    assert maximum > 1
    for selectors in (
        {
            "potential": rocheometry.Potential.jacobi(
                outer_level(geometry.field, graph.outer)
            )
        },
        {"volume_radius_fill": maximum},
        {"fill_out_factor": 1.0},
        {"volume_radius": maximum * geometry.lobe().volume_radius() * 1.001},
    ):
        with pytest.raises(ValueError, match="L2|maximum"):
            geometry.surface(**selectors)  # type: ignore[arg-type]
    copy = pickle.loads(pickle.dumps(surface))
    assert copy == surface
    assert copy.component._computers is graph
    binary = rocheometry.BinaryGeometry(
        qi, separation=7.3, accuracy=geometry.accuracy
    ).component(1)
    assert binary.surface(fill_out_factor=0.5).intrinsic_geometry == surface
    assert binary.max_volume_radius_fill() == maximum
    binary_surface = binary.surface(fill_out_factor=0.5)
    assert pickle.loads(pickle.dumps(binary_surface)) == binary_surface
    assert (
        binary.surface(fill_out_factor=0.5).fill_out_factor()
        == surface.fill_out_factor()
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_tiling_against_cartesian_envelope(qi: float, factor: float) -> None:
    graph = Reference()._computers()
    first = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph)).surface(
        fill_out_factor=factor
    )
    other = rocheometry.ComponentGeometry(1 / qi, accuracy=_Wired(graph)).surface(
        potential=first.potential
    )
    total = first.volume() + other.volume()
    assert total == pytest.approx(
        envelope_volume(first.component.field, first.potential.value), rel=1e-9
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
def test_gauss_volume_against_all_boundary_rays(qi: float) -> None:
    graph = Reference(polar_order=64, azimuth_order=32)._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=0.5)
    sample = graph.sample(geometry.field, surface._depth)
    volume = fsum(
        w * r**3 / 3 for (_, w), r in zip(sample.rays, sample.radii, strict=True)
    )
    nodes, weights = gauss_legendre(64)
    azimuths, aw = gauss_legendre(32)
    for an, wpsi in zip(azimuths, aw, strict=True):
        psi = pi * (an + 1) / 4
        rim = graph.watershed.rim(geometry.field, surface._depth, psi)
        end = atan2(hypot(rim[1], rim[2]), rim[0])
        terms = []
        for node, weight in zip(nodes, weights, strict=True):
            s = (node + 1) / 2
            theta = end * (1 - s * s)
            direction = (cos(theta), sin(theta) * cos(psi), sin(theta) * sin(psi))
            r = hypot(*graph.watershed.sheet_point_on_ray(geometry.field, direction))
            terms.append(pi * wpsi * weight * end * s * sin(theta) * r**3 / 3)
        volume += fsum(terms)
    assert surface.volume() == pytest.approx(volume, rel=1e-8)


@pytest.mark.parametrize("qi", (0.001, 0.05, 1e3))
def test_radius_on_both_sides_of_rim(qi: float) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=0.5)
    assert surface.named_radius("point") == geometry.l1().position[0]
    for psi in (0.0, pi / 6, pi / 3, pi / 2):
        rim = graph.watershed.rim(geometry.field, surface._depth, psi)
        theta_rim = atan2(hypot(rim[1], rim[2]), rim[0])
        for offset in (-0.01, -1e-5, -1e-7, 1e-7, 1e-5, 0.01):
            theta = theta_rim + offset
            direction = (cos(theta), sin(theta) * cos(psi), sin(theta) * sin(psi))
            radius = surface.radius(direction)
            point = tuple(radius * d for d in direction)
            if offset > 0:
                assert geometry.field.potential(*point) == pytest.approx(
                    surface.potential.value, abs=1e-13
                )
            else:
                assert point == pytest.approx(
                    graph.watershed.sheet_point_on_ray(geometry.field, direction),
                    abs=1e-11,
                )
            assert (
                geometry.field.potential(*(0.999 * v for v in point))
                > surface.potential.value
            )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_contact_fill_round_trips(qi: float, factor: float) -> None:
    geometry = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    )
    surface = geometry.surface(fill_out_factor=factor)
    fill = surface.volume_radius_fill()
    recovered = geometry.surface(volume_radius_fill=fill)
    assert recovered.volume_radius_fill() == pytest.approx(fill, rel=1e-13)


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
def test_fill_increases_through_contact(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    )
    fills = [
        geometry.surface(fill_out_factor=factor).volume_radius_fill()
        for factor in (1e-8, *FACTORS)
    ]
    assert 1 < fills[0]
    assert all(a < b for a, b in zip(fills[:-1], fills[1:], strict=True))
    assert fills[-1] < geometry.max_volume_radius_fill()


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_fitted_contact_integrals_match_reference(qi: float, factor: float) -> None:
    reference = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(fill_out_factor=factor)
    fitted = rocheometry.ComponentGeometry(qi).surface(potential=reference.potential)
    errors = {
        reader: getattr(fitted, reader)() / getattr(reference, reader)() - 1
        for reader in ("volume", "area", "integrated_mean_curvature", "mean_gravity")
    }
    assert all(abs(error) <= 1e-6 for error in errors.values()), errors


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
def test_contact_limit_matches_lobe(qi: float) -> None:
    """Curvature is continuous at L1 with a sqrt(F) approach due to the neck."""
    geometry = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    )
    surface = geometry.surface(fill_out_factor=1e-8)
    lobe = geometry.lobe()
    errors = {
        reader: getattr(surface, reader)() / getattr(lobe, reader)() - 1
        for reader in ("volume", "area", "mean_gravity")
    }
    assert all(abs(error) <= 1e-6 for error in errors.values()), errors
    factors = (1e-6, 1e-8)
    surfaces = (geometry.surface(fill_out_factor=factors[0]), surface)
    assert all(s.regime is rocheometry.LevelRegime.CONTACT for s in surfaces), tuple(
        s.regime for s in surfaces
    )
    lobe_curvature = lobe.integrated_mean_curvature()
    rates = tuple(
        (s.integrated_mean_curvature() - lobe_curvature)
        / (lobe_curvature * sqrt(factor))
        for s, factor in zip(surfaces, factors, strict=True)
    )
    assert all(rate < 0 for rate in rates), rates
    assert rates[1] == pytest.approx(rates[0], rel=0.01, abs=0), rates


@pytest.mark.slow
@pytest.mark.parametrize("qi", (0.001, 0.05, 1e3))
def test_star_shape_near_rim_with_independent_basin_classifier(qi: float) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=0.5)
    for psi in (0.0, pi / 4, pi / 2):
        rim = graph.watershed.rim(geometry.field, surface._depth, psi)
        theta_rim = atan2(hypot(rim[1], rim[2]), rim[0])
        for offset in (-1e-3, -1e-5, -1e-7, 1e-7, 1e-5, 1e-3):
            theta = theta_rim + offset
            direction = (cos(theta), sin(theta) * cos(psi), sin(theta) * sin(psi))
            radius = surface.radius(direction)
            inside = []
            for fraction in (
                0.2,
                0.4,
                0.6,
                0.8,
                0.95,
                0.99,
                0.999,
                1.001,
                1.01,
                1.05,
                1.1,
            ):
                point = cast(Point3D, tuple(fraction * radius * d for d in direction))
                inside.append(
                    geometry.field.potential(*point) >= surface.potential.value
                    and uphill_basin(geometry.field, point) == 0
                )
            assert inside == [True] * 7 + [False] * 4


def test_binary_contact_potential_conventions_and_selector_exclusion() -> None:
    binary = rocheometry.BinaryGeometry(0.3, separation=7.3)
    for component_id in (1, 2):
        component = binary.component(component_id)
        surface = component.surface(fill_out_factor=0.5)
        kopal = binary.convert_potential(
            surface.potential,
            convention=rocheometry.PotentialConvention.KOPAL,
            component_id=component_id,
        )
        recovered = component.surface(potential=kopal)
        assert recovered.potential.value == pytest.approx(
            surface.potential.value, abs=1e-15
        )
        assert recovered.fill_out_factor() == pytest.approx(0.5, abs=1e-13)
        assert component.surface(fill_out_factor=0) == component.lobe()
        for selectors in (
            {"potential": surface.potential},
            {"volume_radius_fill": 1.0},
            {"volume_radius": 0.1},
        ):
            with pytest.raises(TypeError, match="exactly one"):
                component.surface(fill_out_factor=0.5, **selectors)  # type: ignore[arg-type]


@pytest.mark.parametrize("qi", (1e-6, 1.0, 1e6))
def test_fill_out_factor_just_below_one_stays_above_l2(qi: float) -> None:
    from math import nextafter

    geometry = rocheometry.ComponentGeometry(qi)
    surface = geometry.surface(fill_out_factor=nextafter(1.0, 0.0))
    from rocheometry._computers.contact import admissibility_depth

    assert surface._depth < admissibility_depth(
        geometry.field, geometry.l1(), geometry._computers.outer
    )
    assert surface.potential.value == outer_level(
        geometry.field, geometry._computers.outer
    )
    assert surface.fill_out_factor() < 1
    assert geometry.surface(fill_out_factor=surface.fill_out_factor()) == surface
    with pytest.raises(ValueError, match="above L2"):
        geometry.surface(potential=surface.potential)
