"""Contact sections close watershed cuts and retain exact envelope subpaths."""

from math import atan2, cos, fsum, hypot, pi, sin

import pytest
from crosschecks.cartesian import bounded_surface_root

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.section_path.gauge_fixed import (
    NECK_SLOT_PLACEMENT_MULTIPLIER,
)
from rocheometry._geometry import Point2D
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._numerics.roots import brent_root
from tests._helpers import _depth, _Wired, gauge_neck_is_uncapped, single_path

QI = (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6)
FACTORS = (1e-6, 0.01, 0.5, 0.99)
INCLINATIONS = (0.0, pi / 6, pi / 3, pi / 2)
CONTINUITY_QI = (1e-8, 1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e4, 1e6)


def check_section(
    qi: float, factor: float, inclination: float, *, reference: bool
) -> tuple[float, float]:
    """Measure independent non-knot radial and height errors, and exact topology."""
    graph = Reference()._computers() if reference else Fitted()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=factor)
    plane = rocheometry.CentrelinePlane(inclination)
    closed = single_path(surface.section(plane))
    envelope = surface.section_envelope(plane).paths[0]
    count = (len(closed.segments) - len(envelope.segments)) // 2
    assert closed.closed and not envelope.closed
    assert count > 0
    assert envelope.segments == closed.segments[count:-count]
    assert (
        closed.segments[0].p0
        == closed.segments[-1].p3
        == (geometry.l1().position[0], 0.0)
    )
    if not reference:
        assert len(closed.segments) == 52
        assert count == 1
    for upper, lower in zip(
        closed.segments[: len(closed.segments) // 2],
        reversed(closed.segments[len(closed.segments) // 2 :]),
        strict=True,
    ):
        assert lower == upper.reversed().transformed(
            rocheometry.AffineTransform2D(yy=-1.0)
        )
    rim = envelope.segments[0].p0
    assert rim[1] > 0
    assert envelope.segments[-1].p3 == (rim[0], -rim[1])
    assert geometry.field.potential(
        rim[0], rim[1] * cos(inclination), -rim[1] * sin(inclination)
    ) == pytest.approx(surface.potential.value, rel=3e-15)
    # Reciprocal charts describe the same stored-mass field and depth.
    companion = rocheometry.BinaryGeometry(qi, accuracy=_Wired(graph)).geometry2
    depth = surface._depth
    other = graph.watershed.rim_at_depth(companion.field, depth, inclination)
    pole = surface.named_radius("pole")
    assert abs(rim[0] - (1 - other[0])) <= 1e-9 * pole
    assert abs(rim[1] - hypot(*other[1:])) <= 1e-9 * pole
    cut_error = 0.0
    for segment in closed.segments[:count]:
        for t in (0.071, 0.193, 0.413, 0.687, 0.919):
            x, y = segment.evaluate(t)
            exact = graph.watershed.cut_height(geometry.field, y, inclination)
            cut_error = max(cut_error, abs(x - exact) / pole)
    radial_error = 0.0
    for segment in envelope.segments[: len(envelope.segments) // 2]:
        for index in range(12):
            x, y = segment.evaluate((index + 0.5) / 12)
            radius = hypot(x, y)
            exact = graph.radius(
                geometry.field,
                surface._depth,
                (
                    x / radius,
                    y / radius * cos(inclination),
                    -y / radius * sin(inclination),
                ),
            )
            radial_error = max(radial_error, abs(radius - exact))
    radial_error /= pole if reference else surface.volume_radius()
    return radial_error, cut_error


@pytest.mark.parametrize(
    "qi,factor,inclination,reference",
    [
        (1e-6, 0.99, 0.0, True),
        (0.05, 0.5, 0.0, True),
        (1.0, 0.01, pi / 3, True),
        (20.0, 0.99, pi / 2, False),
        (0.001, 0.01, pi / 6, False),
    ],
)
def test_fast_contact_subset(
    qi: float, factor: float, inclination: float, reference: bool
) -> None:
    radial, cut = check_section(qi, factor, inclination, reference=reference)
    assert radial <= (1e-10 if reference else 3e-4)
    assert cut <= (1e-10 if reference else 3e-4)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_equal_mass_cut_is_exactly_vertical(
    reference: bool, inclination: float
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    surface = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph)).surface(
        fill_out_factor=0.99
    )
    plane = rocheometry.CentrelinePlane(inclination)
    p, e = single_path(surface.section(plane)), surface.section_envelope(plane).paths[0]
    n = (len(p.segments) - len(e.segments)) // 2
    for segment in p.segments[:n] + p.segments[-n:]:
        assert all(
            abs(point[0] - 0.5) < 1e-12
            for point in (segment.p0, segment.p1, segment.p2, segment.p3)
        )


@pytest.mark.parametrize("qi", CONTINUITY_QI)
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_fitted_controls_continue_through_l1(qi: float, inclination: float) -> None:
    """Check decrease, a finest-depth bound and the asymptotic L1 rate.

    Fixed F depths retain the continuity checks. The ratio uses depths 100
    and 10,000 times below this section's neck-slot transition: a light
    component's fixed coarse depth can still be outside the asymptotic range.
    Positive stored depths remain contact, including on the light side.
    """
    geometry = rocheometry.ComponentGeometry(qi)
    plane = rocheometry.CentrelinePlane(inclination)

    def error(contact: rocheometry.ComponentSurfaceGeometry) -> float:
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        detached = geometry._surface_at_depth(-contact._depth)
        a, b = single_path(contact.section(plane)), single_path(detached.section(plane))
        assert len(a.segments) == len(b.segments) == 52
        return max(
            hypot(p[0] - q[0], p[1] - q[1])
            for sa, sb in zip(a.segments, b.segments, strict=True)
            for p, q in zip(
                (sa.p0, sa.p1, sa.p2, sa.p3), (sb.p0, sb.p1, sb.p2, sb.p3), strict=True
            )
        ) / contact.named_radius("pole")

    errors = [
        error(geometry.surface(fill_out_factor=factor)) for factor in (1e-4, 1e-6, 1e-8)
    ]
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] <= 1e-3, errors
    # sqrt(beta*nu*phi1)=phi1/2 gives nu=phi1/(4*beta), hence
    # delta_t=U_xx*x_L1**2*phi1**2/(32*beta**2).
    m = 26 - 2
    phi1 = pi * (1 / m - sin(2 * pi / m) / (2 * pi))
    l1 = geometry.l1()
    transition = (
        l1.hessian[0][0]
        * l1.position[0] ** 2
        * phi1**2
        / (32 * NECK_SLOT_PLACEMENT_MULTIPLIER**2)
    )
    pair = [
        geometry._surface_at_depth(transition / divisor) for divisor in (100, 10000)
    ]
    assert all(s.regime is rocheometry.LevelRegime.CONTACT for s in pair)
    assert error(pair[1]) <= 0.4 * error(pair[0])


def test_cross_l1_rate_covers_at_least_one_third_of_cases() -> None:
    uncapped_cases = 0
    for qi in CONTINUITY_QI:
        geometry = rocheometry.ComponentGeometry(qi)
        for inclination in INCLINATIONS:
            plane = rocheometry.CentrelinePlane(inclination)
            uncapped = []
            for factor in (1e-6, 1e-8):
                contact = geometry.surface(fill_out_factor=factor)
                assert contact.regime is rocheometry.LevelRegime.CONTACT
                detached = geometry._surface_at_depth(-contact._depth)
                uncapped.extend(
                    (
                        gauge_neck_is_uncapped(contact, plane),
                        gauge_neck_is_uncapped(detached, plane),
                    )
                )
            uncapped_cases += all(uncapped)
    total = len(CONTINUITY_QI) * len(INCLINATIONS)
    assert 3 * uncapped_cases >= total, (uncapped_cases, total)


@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_neck_tangents_converge_to_l1_cone(qi: float, inclination: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    field, l1 = geometry.field, geometry.l1()
    cone = field.critical_upper_tangent(l1.position[0], inclination)
    angles: tuple[list[float], list[float]] = ([], [])
    for factor in (1e-6, 1e-8):
        contact = geometry.surface(fill_out_factor=factor)
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        depth = l1.potential.value - contact.potential.value
        detached = geometry.surface(
            potential=rocheometry.Potential.jacobi(l1.potential.value + depth)
        )
        for errors, surface in zip(angles, (contact, detached), strict=True):
            path = single_path(
                surface.section(rocheometry.CentrelinePlane(inclination))
            )
            x, y = path.segments[2].p0
            gx, gy, gz = field.gradient(x, y * cos(inclination), -y * sin(inclination))
            tx, ty = gy * cos(inclination) - gz * sin(inclination), -gx
            errors.append(
                atan2(abs(tx * cone[1] - ty * cone[0]), tx * cone[0] + ty * cone[1])
            )
    for errors in angles:
        assert errors[1] < errors[0], errors


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize(
    "plane",
    (
        rocheometry.ORBITAL_PLANE,
        rocheometry.CentrelinePlane(pi / 3),
        rocheometry.AxisNormalPlane("y", 0),
        rocheometry.AxisNormalPlane("z", 0.01),
    ),
)
def test_envelope_is_whole_loop_below_contact(
    reference: bool, plane: rocheometry.SectionPlane
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    geometry = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph))
    for u in (geometry.l1().potential.value, geometry.l1().potential.value + 0.1):
        surface = geometry.surface(potential=rocheometry.Potential.jacobi(u))
        assert (
            single_path(surface.section(plane))
            == surface.section_envelope(plane).paths[0]
        )


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize(
    "plane",
    (
        rocheometry.AxisNormalPlane("x", 0),
        rocheometry.AxisNormalPlane("y", 0),
        rocheometry.AxisNormalPlane("z", 0.05),
    ),
)
def test_axis_normal_contact_sections_are_supported(
    reference: bool, plane: rocheometry.SectionPlane
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    surface = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph)).surface(
        fill_out_factor=0.5
    )
    path = single_path(surface.section(plane))
    assert path == single_path(
        graph.section(surface.component.field, surface._depth, plane)
    )
    for arc in surface.section_envelope(plane):
        assert all(segment in path.segments for segment in arc.segments)


def test_binary_envelope_arc_is_transformed_without_closing() -> None:
    binary = rocheometry.BinaryGeometry(0.3, separation=7.0, frame="centre_of_mass")
    for component_id in (1, 2):
        component = binary.component(component_id)
        surface = component.surface(fill_out_factor=0.5)
        for plane in (rocheometry.ORBITAL_PLANE, rocheometry.CentrelinePlane(pi / 3)):
            closed, envelope = (
                single_path(surface.section(plane)),
                surface.section_envelope(plane).paths[0],
            )
            assert not envelope.closed
            assert closed.segments[1:-1] == envelope.segments
            local = surface.intrinsic_geometry.section_envelope(plane).paths[0]
            assert envelope == local.transformed(component._output_transform())


def shoelace(points: tuple[Point2D, ...]) -> float:
    return (
        abs(
            fsum(
                p[0] * q[1] - q[0] * p[1]
                for p, q in zip(points, points[1:] + points[:1], strict=True)
            )
        )
        / 2
    )


def envelope_area(geometry: rocheometry.ComponentGeometry, level: float) -> float:
    """Independent orbital half-width integration, using Cartesian root internals."""
    graph, field = geometry._computers, geometry.field
    companion = rocheometry.ComponentGeometry(1 / geometry.qi, accuracy=_Wired(graph))

    def back(g: rocheometry.ComponentGeometry) -> float:
        return bounded_surface_root(
            g.field._ray_residual(
                (-1.0, 0.0, 0.0),
                _depth(g.field, rocheometry.Potential.jacobi(level)),
                l1_x=g.l1().position[0],
            ),
            abs(graph.outer(g.field).position[0]),
            scale=level,
            inner=1e-15,
        )

    left, right = -back(geometry), 1 + back(companion)

    def width(x: float) -> float:
        minimum = brent_root(
            lambda y: (
                1
                - field.component_mass / hypot(x, y) ** 3
                - field.companion_mass / hypot(1 - x, y) ** 3
            ),
            1e-30,
            2.0,
        )
        return bounded_surface_root(
            lambda y: field.potential(x, y) - level, minimum, scale=level
        )

    nodes, weights = gauss_legendre(256)
    bounds = (left, 0.0, geometry.l1().position[0], 1.0, right)
    # A sine change of variables removes the square-root endpoint behaviour.
    return fsum(
        (b - a)
        * pi
        / 2
        * w
        * cos(pi * t / 2)
        * width((a + b) / 2 + (b - a) / 2 * sin(pi * t / 2))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        for t, w in zip(nodes, weights, strict=True)
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_orbital_part_areas_tile_independent_envelope(qi: float, factor: float) -> None:
    graph = Reference()._computers()
    first = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    second = rocheometry.ComponentGeometry(1 / qi, accuracy=_Wired(graph))
    level = first.surface(fill_out_factor=factor).potential
    paths = tuple(
        single_path(g.surface(potential=level).section()) for g in (first, second)
    )
    # Uniform per-segment samples resolve even the shortest adaptive cubics;
    # convergence here tests polygon area independently of any Bezier integral.
    areas = tuple(
        shoelace(
            tuple(
                segment.evaluate(k / 256)
                for segment in path.segments
                for k in range(256)
            )
        )
        for path in paths
    )
    exact = envelope_area(first, level.value)
    assert sum(areas) == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("qi", (0.001, 0.05, 1.0, 20.0, 1000.0))
def test_contact_control_differences_shrink_with_level_depth(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    critical = single_path(geometry.lobe().section())
    errors = []
    for factor in (1e-6, 1e-9, 1e-12):
        path = single_path(geometry.surface(fill_out_factor=factor).section())
        errors.append(
            max(
                hypot(p[0] - q[0], p[1] - q[1])
                for a, b in zip(path.segments, critical.segments, strict=True)
                for p, q in zip(
                    (a.p0, a.p1, a.p2, a.p3), (b.p0, b.p1, b.p2, b.p3), strict=True
                )
            )
        )
    assert errors[0] > errors[1] > errors[2], errors


def check_cut_midpoint(qi: float, factor: float, inclination: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    surface = geometry.surface(fill_out_factor=factor)
    path = single_path(surface.section(rocheometry.CentrelinePlane(inclination)))
    point = geometry._computers.watershed.rim(
        geometry.field, surface._depth / 4, inclination
    )
    x, t = path.segments[0].evaluate(0.5)
    assert hypot(x - point[0], t - hypot(*point[1:])) <= 1e-12 * surface.named_radius(
        "pole"
    )


@pytest.mark.parametrize("inclination", INCLINATIONS)
@pytest.mark.parametrize("qi", (0.05, 1.0))
def test_fitted_cut_interpolates_middle_square_root_depth(
    qi: float, inclination: float
) -> None:
    check_cut_midpoint(qi, 0.99, inclination)


@pytest.mark.parametrize("qi", (1e3, 1e6))
def test_cut_midpoint_near_l1_in_reciprocal_frame(qi: float) -> None:
    check_cut_midpoint(qi, 1e-6, pi / 6)


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_full_cut_midpoint_grid(qi: float, factor: float, inclination: float) -> None:
    check_cut_midpoint(qi, factor, inclination)
