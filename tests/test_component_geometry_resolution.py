"""Absolute levels and log depths stop at their representable resolution."""

import json
from decimal import Decimal, localcontext
from math import atan2, cos, hypot, inf, nextafter, pi, sin, ulp
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest

import rocheometry
from rocheometry._computers.surface_potential import BisectionSurfacePotentialComputer
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._computers.watershed.traced import TracedWatershedComputer
from rocheometry._roche.volume import volume_radius_from_volume
from tests._helpers import _depth
from tests._reference_precision import geometry_at_40_digits, volume_at_40_digits


class _PrecisionCase(TypedDict):
    qi: float
    potential: float
    baseline_fill: float
    mp_fill: str


_PRECISION_CASES = cast(
    list[_PrecisionCase],
    json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("reference-resolution.json")
        .read_text()
    )["cases"],
)


class _GeometryCase(TypedDict):
    reference: bool
    qi: float
    potential: float
    factor: float | None
    baseline_integrated_mean_curvature: float
    baseline_normal_error: float
    baseline_fill: NotRequired[float]


_GEOMETRY_CASES = cast(
    list[_GeometryCase],
    json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("residual-geometry.json")
        .read_text()
    )["cases"],
)


@pytest.mark.slow
@pytest.mark.parametrize("case", _GEOMETRY_CASES)
def test_both_residual_policies_against_40_digit_geometry(case: _GeometryCase) -> None:
    geometry = rocheometry.ComponentGeometry(
        case["qi"],
        accuracy=rocheometry.Reference() if case["reference"] else rocheometry.Fitted(),
    )
    graph, field = geometry._computers, geometry.field
    surface = geometry.surface(
        potential=rocheometry.Potential.jacobi(case["potential"])
    )
    sample = graph.sample(field, surface._depth)
    volume, curvature, normals = geometry_at_40_digits(field, sample, case["potential"])
    with localcontext() as context:
        context.prec = 40
        reference = Decimal(curvature)
        actual = Decimal.from_float(surface.integrated_mean_curvature())
        baseline = Decimal.from_float(case["baseline_integrated_mean_curvature"])
        assert abs(actual - reference) <= abs(baseline - reference) + Decimal(
            "1e-14"
        ) * abs(reference)
        if case["qi"] == 1e-3 and case["factor"] == 1e-6:
            assert abs(actual / reference - 1) < Decimal("1e-11")
        observed = surface.quadrature().normals[::4]
        error = max(
            sum(
                (
                    (Decimal.from_float(a) - Decimal(b)) ** 2
                    for a, b in zip(n, exact, strict=True)
                ),
                Decimal(0),
            ).sqrt()
            for n, exact in zip(observed, normals, strict=True)
        )
        assert error <= Decimal.from_float(case["baseline_normal_error"]) + Decimal(
            "1e-14"
        )
        if "baseline_fill" in case:
            lobe, _ = volume_at_40_digits(
                field,
                graph.sample(field, 0.0),
                geometry.l1().potential.value,
                saddle_x=geometry.l1().position[0],
            )
            reference_fill = (Decimal(volume) / Decimal(lobe)) ** (Decimal(1) / 3)
            sampler = SampledVolumeComputer(sample=graph.sample)
            actual_fill = Decimal.from_float(
                (sampler(field, surface._depth) / sampler(field, 0.0)) ** (1 / 3)
            )
            assert abs(actual_fill - reference_fill) <= abs(
                Decimal.from_float(case["baseline_fill"]) - reference_fill
            )


@pytest.mark.slow
@pytest.mark.parametrize("case", _PRECISION_CASES, ids=lambda case: str(case["qi"]))
def test_exact_anchor_fill_beats_rounded_anchor_at_40_digits(
    case: _PrecisionCase,
) -> None:
    geometry = rocheometry.ComponentGeometry(
        case["qi"], accuracy=rocheometry.Reference()
    )
    surface = geometry.surface(
        potential=rocheometry.Potential.jacobi(case["potential"])
    )
    graph, field = geometry._computers, geometry.field
    lobe, _ = volume_at_40_digits(
        field,
        graph.sample(field, 0.0),
        geometry.l1().potential.value,
        saddle_x=geometry.l1().position[0],
    )
    volume, _ = volume_at_40_digits(
        field, graph.sample(field, surface._depth), surface.potential.value
    )
    with localcontext() as context:
        context.prec = 40
        reference = (Decimal(volume) / Decimal(lobe)) ** (Decimal(1) / 3)
        # The live solve is 40-digit; its recorded value allows for platforms
        # whose libm gives slightly different binary64 quadrature directions.
        assert abs(reference - Decimal(case["mp_fill"])) < Decimal("5e-15")
        actual = Decimal.from_float(surface.volume_radius_fill())
        assert abs(actual - reference) <= abs(
            Decimal.from_float(case["baseline_fill"]) - reference
        )


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3))
def test_critical_rays_have_exact_cone_roots_and_next_level_is_detached(
    qi: float,
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    l1 = geometry.l1()
    graph = geometry._computers
    sample = graph.sample(geometry.field, 0.0)
    _, error = volume_at_40_digits(
        geometry.field, sample, l1.potential.value, saddle_x=l1.position[0]
    )
    # Near-axis cone roots amplify residual roundoff; this remains twentyfold
    # below the traced-path tolerance, including the very first polar ring.
    assert error < 5e-12
    level = rocheometry.Potential.jacobi(nextafter(l1.potential.value, inf))
    # Use the same rays to isolate the level change from level-dependent nodes.
    for (direction, _), radius in zip(sample.rays, sample.radii, strict=True):
        inside = graph.radius(geometry.field, _depth(geometry.field, level), direction)
        assert 0 < inside <= radius * (1 + 1e-13)


@pytest.mark.parametrize("qi", (1e-8, 1e-6, 1e-3))
def test_designated_outer_level_selects_the_inward_back_cone_limit(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    outer = geometry.outer()
    field = geometry.field
    depth = _depth(field, outer.potential)
    offset = field._depth_offset(depth, geometry.l1().position[0], outer.position[0])
    # The largest representable closed-side depth is at most one ULP inward,
    # rather than the nearest-rounded (half-ULP) approximation to the saddle.
    assert -ulp(depth) < offset <= 0.0
    assert geometry._computers.radius(
        field, nextafter(_depth(field, outer.potential), 0.0), (-1.0, 0.0, 0.0)
    ) < abs(outer.position[0])


@pytest.mark.parametrize("qi", (1e-8, 1e-7, 1e-6))
@pytest.mark.parametrize("fill", (0.5, 0.99, 0.999999))
def test_reference_fill_stops_at_the_closest_representable_level(
    qi: float, fill: float
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    surface = geometry.surface(volume_radius_fill=fill)
    target = geometry.lobe().volume_radius() * fill
    level = surface._depth

    def error(value: float) -> float:
        return geometry._surface_at_depth(value).volume_radius() / target - 1

    lower, upper = nextafter(level, -inf), nextafter(level, inf)
    actual, below, above = error(level), error(lower), error(upper)
    if abs(actual) > 1e-13:
        assert abs(actual) <= min(abs(below), abs(above))
    neighbour = below if actual < 0 else above
    floor = abs(actual - neighbour)
    assert abs(actual) <= max(1e-13, floor)
    assert surface.volume_radius_fill() == pytest.approx(fill, rel=max(1e-13, floor))


def test_bisection_fill_uses_the_same_representable_floor() -> None:
    geometry = rocheometry.ComponentGeometry(1e-8, accuracy=rocheometry.Reference())
    graph = geometry._computers
    volume = SampledVolumeComputer(sample=graph.sample)
    computer = BisectionSurfacePotentialComputer(
        volume_computer=volume,
        l1_computer=graph.l1,
        lobe_volume_computer=graph.selection.fast_lobe_volume,
        outer_computer=graph.outer,
        volume_radius_fill_tolerance=1e-13,
    )
    potential = computer(geometry.field, 0.99)
    target = geometry.lobe().volume_radius() * 0.99
    values = [
        nextafter(potential, -inf),
        potential,
        nextafter(potential, inf),
    ]
    errors = [
        volume_radius_from_volume(volume(geometry.field, u)) / target - 1
        for u in values
    ]
    assert abs(errors[1]) <= max(1e-13, abs(errors[0] - errors[2]))
    if abs(errors[1]) > 1e-13:
        assert abs(errors[1]) <= min(abs(errors[0]), abs(errors[2]))


@pytest.mark.parametrize("qi", (1e-8, 1e-6))
def test_reference_contact_fill_uses_adjacent_absolute_levels(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    lower = geometry.surface(fill_out_factor=0.5)
    upper = geometry.surface(
        potential=rocheometry.Potential.jacobi(nextafter(lower.potential.value, inf))
    )
    target = (lower.volume_radius() + upper.volume_radius()) / 2
    fill = target / geometry.lobe().volume_radius()
    selected = geometry.surface(volume_radius_fill=fill)
    error = abs(selected.volume_radius() / target - 1)
    floor = abs(lower.volume_radius() - upper.volume_radius()) / target
    assert selected.potential in (lower.potential, upper.potential)
    assert error <= max(1e-13, floor)
    assert (
        error
        <= min(abs(s.volume_radius() / target - 1) for s in (lower, upper)) + 1e-13
    )


def test_critical_light_component_radii_have_no_absolute_level_noise() -> None:
    geometry = rocheometry.ComponentGeometry(1e-8, accuracy=rocheometry.Reference())
    surface = geometry.lobe()
    # A front-ray switch between factored and unfactored residuals at this
    # angle would show as noise in the radii either side of it.
    angle = 0.14153947758451035
    radii = [
        geometry._computers.radius(
            geometry.field,
            surface._depth,
            (cos(angle + k * 1e-12), sin(angle + k * 1e-12), 0.0),
        )
        for k in range(-4, 5)
    ]
    scatter = (
        max(
            abs(radii[k + 1] - 2 * radii[k] + radii[k - 1])
            for k in range(1, len(radii) - 1)
        )
        / radii[4]
    )
    assert scatter < 1e-12


@pytest.mark.parametrize("qi", (1e-8, 1.0, 1e8))
@pytest.mark.parametrize("theta", (0.01, 0.3))
def test_factored_ray_residual_accepts_the_negative_outer_reference(
    qi: float, theta: float
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    field = geometry.field
    outer = geometry.outer()
    direction = (-cos(theta), sin(theta), 0.0)
    radius = abs(outer.position[0]) * 0.9999
    point = (radius * direction[0], radius * direction[1], 0.0)
    level = nextafter(outer.potential.value, inf)
    expected = field.potential_difference(point, outer.position) + field._depth_offset(
        _depth(field, rocheometry.Potential.jacobi(level)),
        geometry.l1().position[0],
        outer.position[0],
    )
    actual = field._ray_residual(
        direction,
        _depth(field, rocheometry.Potential.jacobi(level)),
        outer.position[0],
        l1_x=geometry.l1().position[0],
    )(radius)
    # The independent general difference loses about eps*theta**2 in the
    # heavy-star nearly axisymmetric limit; retain that absolute roundoff.
    assert actual == pytest.approx(expected, rel=2e-12, abs=1e-17)


@pytest.mark.parametrize("inclination", (0.0, pi / 3))
def test_shallow_heavy_contact_cut_is_representable(inclination: float) -> None:
    geometry = rocheometry.ComponentGeometry(1e8, accuracy=rocheometry.Reference())
    surface = geometry.surface(fill_out_factor=1e-6)
    assert surface.section(rocheometry.CentrelinePlane(inclination)).paths
    watershed = geometry._computers.watershed
    assert isinstance(watershed, TracedWatershedComputer)
    rho = 5.098558872088485e-8
    # A dense line whose log-depth endpoints are adjacent floats, far inside
    # the sheet; its points must still meet an 8e-15 threshold.
    line = watershed(geometry.field, 0.016482509439569464)
    point = line.at_radius(rho)
    assert abs(hypot(*point[1:]) - rho) <= 1.65e-14 * rho
    psi = atan2(point[2], point[1])
    height = watershed.cut_height(geometry.field, rho, psi)
    assert abs(height - point[0]) <= ulp(height)
    rim = watershed.rim(geometry.field, surface._depth, inclination)
    rim_height = watershed.cut_height(
        geometry.field, hypot(*rim[1:]), atan2(rim[2], rim[1])
    )
    assert abs(rim_height - rim[0]) <= ulp(rim_height)
    end = line.at_log_depth(line.log_depths[-1])
    with pytest.raises(
        ValueError, match="transverse radius lies beyond the traced watershed"
    ):
        line.at_radius(nextafter(hypot(*end[1:]), inf))


_LEVELS = (
    ("fill", 0.5),
    ("fill", 0.99),
    ("fill", 0.999999),
    ("fill", 1.0),
    ("contact", 1e-6),
    ("contact", 0.01),
    ("contact", 0.5),
    ("contact", 0.99),
)


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-8, 1e-7, 1e-6, 1e6, 1e7, 1e8))
@pytest.mark.parametrize("kind,value", _LEVELS)
@pytest.mark.parametrize(
    "reader", ("integrals", "orbital", "inclined", "z_slice", "silhouette")
)
def test_extreme_reference_probe(
    qi: float, kind: str, value: float, reader: str
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    surface = (
        geometry.surface(volume_radius_fill=value)
        if kind == "fill"
        else geometry.surface(fill_out_factor=value)
    )
    if reader == "integrals":
        assert surface.volume() > 0
        assert surface.area() > 0
        assert surface.integrated_mean_curvature() > 0
        assert surface.mean_gravity() > 0
        assert len(surface.quadrature()) > 0
    elif reader == "orbital":
        assert surface.section().paths
    elif reader == "inclined":
        assert surface.section(rocheometry.CentrelinePlane(pi / 3)).paths
    elif reader == "z_slice":
        assert surface.section(
            rocheometry.AxisNormalPlane("z", 0.3 * surface.named_radius("pole"))
        ).paths
    else:
        assert surface.silhouette_envelope(rocheometry.CentrelinePlane(pi / 3)).paths
