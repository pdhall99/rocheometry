"""Whole-surface geometry identities and default/reference accuracy checks."""

from dataclasses import replace
from math import fsum, hypot, pi
from unittest.mock import patch

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.lobe_slope import SampledLobeSlopeComputer
from rocheometry._computers.mean_curvature import SampledMeanCurvatureComputer
from rocheometry._computers.surface_area import SampledAreaComputer
from rocheometry._computers.surface_sample import (
    GaussLegendreSurfaceSampler,
    SurfaceSample,
    angular,
)
from rocheometry._computers.surface_sample.terms import ray_terms
from rocheometry._computers.surface_selection import (
    FittedSelection,
    VolumeInversionSelection,
)
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._roche.field import RocheField
from tests._helpers import RadiusDelegate, _depth, _Wired, computer_graph


def _task2_azimuths(
    order: int, companion_width: float
) -> tuple[tuple[float, float], ...]:
    nodes, weights = gauss_legendre(order)
    return tuple(
        (pi * (node + 1) / 4, pi * weight / 4)
        for node, weight in zip(nodes, weights, strict=True)
    )


@pytest.fixture(scope="module")
def task2_default_graph() -> ComputerSet:
    """Retain the task 2 default's selection and plain 64-by-16 sampling."""
    base = Fitted()._computers()
    sampler = GaussLegendreSurfaceSampler(
        base.radius,
        64,
        16,
        l1_computer=base.l1,
        outer_computer=base.outer,
        watershed_computer=base.watershed,
    )

    def sampled(field: RocheField, potential: float, /) -> SurfaceSample:
        with patch.object(angular, "_azimuth_quadrature", _task2_azimuths):
            return sampler(field, potential)

    sample = Memoized(sampled, maxsize=16)
    selection = FittedSelection(
        VolumeInversionSelection(
            volume_computer=SampledVolumeComputer(sample),
            l1_computer=base.l1,
            outer_computer=base.outer,
            volume_radius_fill_tolerance=2e-8,
        ),
        l1_computer=base.l1,
    )
    return replace(
        base,
        sample=sample,
        area=SampledAreaComputer(sample),
        mean_curvature=SampledMeanCurvatureComputer(sample),
        lobe_slope=SampledLobeSlopeComputer(sample=sample, l1_computer=base.l1),
        selection=selection,
    )


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
def test_small_surface_measurement_radii_approach_the_sphere(qi: float) -> None:
    surface = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=0.05)
    radius = surface.volume_radius()
    assert surface.area_radius() == pytest.approx(radius, rel=1e-8, abs=0)
    assert surface.mean_curvature_radius() == pytest.approx(radius, rel=1e-8, abs=0)


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("fill", [0.9, 1.0])
def test_minkowski_integral_identity(qi: float, fill: float) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(volume_radius_fill=fill)
    sample = graph.sample(geometry.field, surface._depth)
    residuals = []
    for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
        area, curvature_area, support = ray_terms(geometry.field, direction, radius)
        residuals.append(weight * (area - curvature_area * support))
    assert abs(fsum(residuals)) / surface.area() <= 1e-12
    # The identity is an integral cancellation, not a pointwise equality.
    assert max(abs(value) for value in residuals) / surface.area() > 1e-6


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("fill", [0.9, 1.0])
def test_mean_gravity_gauss_law_matches_direct_surface_integral(
    qi: float, fill: float
) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(volume_radius_fill=fill)
    sample = graph.sample(geometry.field, surface._depth)
    terms = []
    for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
        gradient = geometry.field.gradient(
            radius * direction[0], radius * direction[1], radius * direction[2]
        )
        area, _, _ = ray_terms(geometry.field, direction, radius)
        terms.append(weight * hypot(*gradient) * area)
    assert surface.mean_gravity() == pytest.approx(
        fsum(terms) / surface.area(), rel=1e-10, abs=0
    )


@pytest.mark.parametrize(
    "qi, area_ratio, curvature_ratio, gravity",
    [
        (0.1, 1.0058720088, 1.0161524749, 1.9943575147),
        (1.0, 1.0038475109, 1.0111879282, 3.1872820054),
        (10.0, 1.0029134651, 1.0078216796, 2.2991967356),
    ],
)
def test_lobe_measurements_regression(
    qi: float, area_ratio: float, curvature_ratio: float, gravity: float
) -> None:
    surface = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).lobe()
    radius = surface.volume_radius()
    assert surface.area_radius() / radius == pytest.approx(area_ratio, rel=1e-9)
    assert surface.mean_curvature_radius() / radius == pytest.approx(
        curvature_ratio, rel=1e-9
    )
    assert surface.mean_gravity() == pytest.approx(gravity, rel=1e-9)


@pytest.mark.parametrize("qi", [1e-3, 0.1, 1.0, 10.0, 1e3])
@pytest.mark.parametrize("fill", [0.5, 0.99, 1.0])
def test_surface_measurement_radii_are_ordered(qi: float, fill: float) -> None:
    surface = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=fill)
    assert (
        surface.volume_radius()
        <= surface.area_radius()
        <= surface.mean_curvature_radius()
    )


@pytest.mark.parametrize(
    "qi, fill",
    [
        *[
            (qi, fill)
            for qi in (1e-3, 0.1, 1.0, 10.0, 1e3)
            for fill in (0.5, 0.99, 1.0)
        ],
        *[(qi, fill) for qi in (1e4, 1e6) for fill in (0.99, 0.999, 1.0)],
    ],
)
def test_default_surface_measurements_agree_with_reference(
    qi: float, fill: float
) -> None:
    reference = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=fill)
    actual = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill)
    assert actual.area() == pytest.approx(reference.area(), rel=1e-6, abs=0)
    assert actual.integrated_mean_curvature() == pytest.approx(
        reference.integrated_mean_curvature(), rel=1e-6, abs=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3))
@pytest.mark.parametrize("fill", (0.999, 0.9999, 0.99999))
def test_near_critical_detached_measurements_agree_with_reference(
    qi: float, fill: float
) -> None:
    reference = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=fill)
    fitted = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill)
    assert fitted.area() == pytest.approx(reference.area(), rel=1e-6, abs=0)
    assert fitted.integrated_mean_curvature() == pytest.approx(
        reference.integrated_mean_curvature(), rel=1e-6, abs=0
    )


@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3))
@pytest.mark.parametrize("graph", (Fitted()._computers(), Reference()._computers()))
def test_detached_measurements_join_the_critical_cone(
    qi: float, graph: ComputerSet
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(volume_radius_fill=1 - 1e-12)
    lobe = geometry.lobe()
    assert surface.area() == pytest.approx(lobe.area(), rel=1e-6, abs=0)
    assert surface.integrated_mean_curvature() == pytest.approx(
        lobe.integrated_mean_curvature(), rel=1e-6, abs=0
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "regime, selector",
    [
        *[("contact", factor) for factor in (1e-6, 0.01, 0.5, 0.99)],
        *[("detached", fill) for fill in (0.99, 0.999, 1.0)],
    ],
)
def test_reference_equatorial_edge_is_azimuth_converged(
    regime: str, selector: float
) -> None:
    geometry = rocheometry.ComponentGeometry(
        1000000.0, accuracy=_Wired(Reference()._computers())
    )
    reference = (
        geometry.surface(fill_out_factor=selector)
        if regime == "contact"
        else geometry.surface(volume_radius_fill=selector)
    )
    finer = rocheometry.ComponentGeometry(
        1000000.0, accuracy=_Wired(Reference(azimuth_order=128)._computers())
    ).surface(potential=reference.potential)
    readers = (
        ("volume", "area", "integrated_mean_curvature", "mean_gravity")
        if regime == "contact"
        else ("area", "integrated_mean_curvature")
    )
    for reader in readers:
        assert getattr(reference, reader)() == pytest.approx(
            getattr(finer, reader)(), rel=1e-8, abs=0
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "qi, regime, selector",
    [
        *[
            (qi, "contact", factor)
            for qi in (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3)
            for factor in (1e-6, 0.01, 0.5, 0.99)
        ],
        *[
            (qi, "detached", fill)
            for qi in (1e-3, 0.1, 1.0, 10.0, 1e3)
            for fill in (0.5, 0.99, 0.999, 1.0)
        ],
    ],
)
def test_comparable_mass_integrals_preserve_task2_accuracy(
    qi: float,
    regime: str,
    selector: float,
    monkeypatch: pytest.MonkeyPatch,
    task2_default_graph: ComputerSet,
) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = (
        geometry.surface(fill_out_factor=selector)
        if regime == "contact"
        else geometry.surface(volume_radius_fill=selector)
    )
    sampler = GaussLegendreSurfaceSampler(
        graph.radius,
        128,
        64,
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        watershed_computer=graph.watershed,
    )
    actual = sampler(geometry.field, surface._depth)

    with monkeypatch.context() as context:
        context.setattr(angular, "_azimuth_quadrature", _task2_azimuths)
        task2 = sampler(geometry.field, surface._depth)

    def measurements(sample: SurfaceSample) -> tuple[float, float, float, float]:
        def sampled(field: RocheField, potential: float, /) -> SurfaceSample:
            return sample

        field, potential = geometry.field, surface.potential
        volume = SampledVolumeComputer(sampled)(field, _depth(field, potential))
        area = SampledAreaComputer(sampled)(field, _depth(field, potential))
        curvature = SampledMeanCurvatureComputer(sampled)(
            field, _depth(field, potential)
        )
        gravity = (4 * pi * field.component_mass - 2 * volume) / area
        return volume, area, curvature, gravity

    assert measurements(actual) == pytest.approx(measurements(task2), rel=1e-9, abs=0)

    new_geometry = rocheometry.ComponentGeometry(qi)
    old_geometry = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(task2_default_graph)
    )
    # Isolate quadrature accuracy from cancellation against fitted-selection
    # error: all three integrals must describe the same equipotential.
    new = new_geometry.surface(potential=surface.potential)
    old = old_geometry.surface(potential=surface.potential)
    readers = (
        ("volume", "area", "integrated_mean_curvature", "mean_gravity")
        if regime == "contact"
        else ("area", "integrated_mean_curvature")
    )
    for reader in readers:
        expected = getattr(surface, reader)()
        old_error = abs(getattr(old, reader)() - expected)
        new_error = abs(getattr(new, reader)() - expected)
        assert new_error <= old_error + 1e-9 * abs(expected), (
            reader,
            old_error,
            new_error,
        )


def test_volume_area_and_mean_curvature_share_one_radius_pass() -> None:
    calls = 0
    reference_radius = Fitted()._computers().radius

    def radius(field: RocheField, potential: float, direction: Point3D, /) -> float:
        nonlocal calls
        calls += 1
        return reference_radius(field, potential, direction)

    graph = computer_graph(
        radius_computer=RadiusDelegate(radius), polar_order=8, azimuth_order=6
    )
    surface = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph)).lobe()
    assert surface.volume() > 0
    assert calls == 8 * 6
    assert surface.area() > 0
    assert surface.integrated_mean_curvature() > 0
    assert calls == 8 * 6
    assert surface.area_radius() > 0
    assert surface.mean_curvature_radius() > 0
    assert surface.mean_gravity() > 0
    assert calls == 8 * 6
