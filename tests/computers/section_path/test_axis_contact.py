"""Axis-normal sections of common-envelope parts, including cut corners."""

from math import atan2, cos, fsum, hypot, pi, sin, sqrt
from typing import cast

import pytest
from crosschecks.cartesian import bounded_surface_root

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.section_path.axis_contact import AxisContactSection
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import SectionAxis
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired, single_path

QI = (1e-3, 0.05, 1.0, 20.0, 1e3)
FACTORS = (0.01, 0.5, 0.99)


def tracer(graph: ComputerSet) -> TracedSectionPathComputer:
    """Use the graph's own traced strategy and dependencies."""
    computer = graph.section
    if isinstance(computer, PlaneRoutedSectionPathComputer):
        computer = computer.section
    assert isinstance(computer, TracedSectionPathComputer)
    return computer


def check_transverse(
    qi: float, factor: float, axis: SectionAxis, fraction: float = 0.6
) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=factor)
    normal = 1 if axis == "y" else 2
    moving = 3 - normal
    extent = graph.watershed.rim(
        geometry.field, surface._depth, 0.0 if normal == 1 else pi / 2
    )[normal]
    plane = rocheometry.AxisNormalPlane(axis, fraction * extent)
    path = single_path(surface.section(plane))
    arc = surface.section_envelope(plane).paths[0]
    assert path.closed and not arc.closed
    n = (len(path.segments) - len(arc.segments)) // 2
    assert n > 0 and path.segments[n:-n] == arc.segments
    assert path.segments[0].p0[1] == 0.0
    assert path.segments[0].p1[0] == path.segments[0].p0[0]
    pole = surface.named_radius("pole")
    for a, b in zip(
        path.segments[: len(path.segments) // 2],
        reversed(path.segments[len(path.segments) // 2 :]),
        strict=True,
    ):
        assert a.reversed().transformed(rocheometry.AffineTransform2D(yy=-1.0)) == b
    for segment in path.segments[:n]:
        for parameter in (0.137, 0.379, 0.713, 0.947):
            x, t = segment.evaluate(parameter)
            y, z = (plane.offset, t) if normal == 1 else (t, plane.offset)
            exact = graph.watershed.cut_height(geometry.field, hypot(y, z), atan2(z, y))
            assert abs(x - exact) <= 1e-10 * pole
    for segment in arc.segments[: len(arc.segments) // 2]:
        for parameter in (0.137, 0.379, 0.713, 0.947):
            x, t = segment.evaluate(parameter)
            r = hypot(x, t)

            def residual(distance: float) -> float:
                coords = [distance * x / r, 0.0, 0.0]
                coords[normal] = plane.offset
                coords[moving] = distance * t / r
                field = geometry.field
                if qi > 1:
                    field = RocheField(1 / qi)
                    coords[0] = 1 - coords[0]
                l1 = graph.l1(field)
                return field.potential_difference(
                    (coords[0], coords[1], coords[2]), l1.position
                ) + (l1.potential.value - surface.potential.value)  # type: ignore[arg-type]

            exact = brent_root(
                residual,
                0.0,
                min(
                    sqrt(
                        abs(graph.outer(geometry.field).position[0]) ** 2
                        - plane.offset**2
                    ),
                    arc.segments[0].p0[0] * r / x,
                )
                if x > 0
                else sqrt(
                    abs(graph.outer(geometry.field).position[0]) ** 2 - plane.offset**2
                ),
                absolute_tolerance=0.0,
                relative_tolerance=8e-16,
            )
            assert abs(r - exact) <= 1e-10 * pole
    # Offsets of opposite sign have exactly the same planar coordinates.
    assert (
        single_path(surface.section(rocheometry.AxisNormalPlane(axis, -plane.offset)))
        == path
    )
    if qi == 1:
        assert all(
            p[0] == 0.5 for s in path.segments[:n] for p in (s.p0, s.p1, s.p2, s.p3)
        )


@pytest.mark.parametrize("axis", ("y", "z"))
def test_equal_mass_offset_sections(axis: SectionAxis) -> None:
    check_transverse(1.0, 0.5, axis)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_contact_axis_normal_routes_to_tracer(
    reference: bool, axis: SectionAxis
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=0.5)
    plane = rocheometry.AxisNormalPlane(axis, 0.03)
    assert single_path(surface.section(plane)) == single_path(
        tracer(graph)(geometry.field, surface._depth, plane)
    )


def polygon_area(
    path: rocheometry.BezierPath | rocheometry.CompoundBezierPath,
) -> float:
    if isinstance(path, rocheometry.CompoundBezierPath):
        return sum(polygon_area(p) for p in path)
    points = tuple(s.evaluate(k / 256) for s in path.segments for k in range(256))
    return (
        abs(
            fsum(
                p[0] * q[1] - q[0] * p[1]
                for p, q in zip(points, points[1:] + points[:1], strict=True)
            )
        )
        / 2
    )


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("axis", ("y", "z"))
def test_transverse_rim_extent_classification_and_continuity(
    qi: float, factor: float, axis: SectionAxis
) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=factor)
    normal = 1 if axis == "y" else 2
    extent = graph.watershed.rim(
        geometry.field, surface._depth, 0.0 if normal == 1 else pi / 2
    )[normal]
    areas = []
    for fraction in (1 - 1e-7, 1.0, 1 + 1e-7):
        plane = rocheometry.AxisNormalPlane(axis, extent * fraction)
        paths = surface.section(plane)
        envelope = surface.section_envelope(plane)
        assert all(p.closed for p in paths)
        if fraction >= 1:
            assert envelope == paths
        areas.append(polygon_area(paths))
    assert max(areas) - min(areas) <= 1e-5 * areas[-1]


@pytest.mark.parametrize("qi", (1.0, 20.0))
def test_ordinary_and_empty_axis_sections(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    surface = geometry.surface(fill_out_factor=0.5)
    x1 = geometry.l1().position[0]
    for x in (-0.1, 0.0, x1 - 0.01):
        path = single_path(surface.section(rocheometry.AxisNormalPlane("x", x)))
        assert (
            path.closed
            and path
            == surface.section_envelope(rocheometry.AxisNormalPlane("x", x)).paths[0]
        )
    for x in (-2.0, x1, 2.0):
        with pytest.raises(rocheometry.EmptySectionError):
            single_path(surface.section(rocheometry.AxisNormalPlane("x", x)))
    if qi == 1:
        for reader in (surface.section, surface.section_envelope):
            with pytest.raises(rocheometry.EmptySectionError):
                reader(rocheometry.AxisNormalPlane("x", 0.5))


def test_heavier_star_ring_band() -> None:
    geometry = rocheometry.ComponentGeometry(3.0)
    surface = geometry.surface(fill_out_factor=0.99)
    lower = geometry.l1().position[0]
    upper = max(
        geometry._computers.watershed.rim(geometry.field, surface._depth, a)[0]
        for a in (0.0, pi / 2)
    )
    assert lower == pytest.approx(0.6107, abs=1e-3)
    assert upper == pytest.approx(0.6250, abs=1e-3)
    for x in (lower + 1e-8, (lower + upper) / 2, upper - 1e-8):
        with pytest.raises(
            NotImplementedError,
            match="^ring-shaped section of the heavier component's part$",
        ):
            single_path(surface.section(rocheometry.AxisNormalPlane("x", x)))
    for x in (upper, upper + 1e-8):
        with pytest.raises(rocheometry.EmptySectionError):
            single_path(surface.section(rocheometry.AxisNormalPlane("x", x)))


@pytest.mark.parametrize(("qi", "factor"), [(1e-3, 0.5), (1.0, 0.01), (1e3, 0.99)])
def test_axial_contact_section_structure_and_corners(qi: float, factor: float) -> None:
    # Accuracy over the whole contact domain is measured by the outlines
    # contract study; these cases guard the cut/surface split and the arcs.
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    surface = geometry.surface(fill_out_factor=factor)
    rims = [
        graph.watershed.rim(geometry.field, surface._depth, a) for a in (0.0, pi / 2)
    ]
    x = (rims[0][0] + rims[1][0]) / 2 if qi < 1 else geometry.l1().position[0] - 0.01
    path = single_path(surface.section(rocheometry.AxisNormalPlane("x", x)))
    pole = surface.named_radius("pole")
    helper = AxisContactSection(
        tracer(graph),
        geometry.field,
        surface._depth,
        rocheometry.AxisNormalPlane("x", x),
    )
    assert path.closed
    for segment in path.segments[: len(path.segments) // 4]:
        for parameter in (0.137, 0.379, 0.713, 0.947):
            y, z = segment.evaluate(parameter)
            rho, psi = hypot(y, z), atan2(z, y)
            cut = qi < 1 and x > helper.rim(psi)[0]
            if cut:
                # Opposite height signs bracket the independent sheet root
                # inside the required radius-error interval. This avoids an
                # unnecessary second complete nested inversion in the test.
                limit = 1e-10 * pole
                left = (
                    graph.watershed.cut_height(
                        geometry.field, max(0.0, rho - limit), psi
                    )
                    - x
                )
                right = graph.watershed.cut_height(geometry.field, rho + limit, psi) - x
                assert left * right <= 0
            else:
                exact = brent_root(
                    lambda r: (
                        geometry.field.potential(x, r * cos(psi), r * sin(psi))
                        - surface.potential.value
                    ),
                    rho * (1 - 1e-4),
                    rho * (1 + 1e-4),
                    absolute_tolerance=0.0,
                    relative_tolerance=8e-16,
                )
                assert abs(rho - exact) <= 1e-10 * pole
    arcs = surface.section_envelope(rocheometry.AxisNormalPlane("x", x))
    if qi < 1:
        assert len(arcs) == 2 and all(not p.closed for p in arcs)
        assert all(s in path.segments for p in arcs for s in p.segments)
    else:
        assert arcs == rocheometry.CompoundBezierPath((path,))


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
def test_offset_parts_tile_independent_slice(qi: float, factor: float) -> None:

    graph = Reference()._computers()
    first = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    second = rocheometry.ComponentGeometry(1 / qi, accuracy=_Wired(graph))
    level = first.surface(fill_out_factor=factor).potential
    z = (
        0.5
        * graph.watershed.rim(
            first.field, first.surface(fill_out_factor=factor)._depth, pi / 2
        )[2]
    )
    paths = tuple(
        g.surface(potential=level).section(rocheometry.AxisNormalPlane("z", z))
        for g in (first, second)
    )
    field = first.field

    # Project the independent envelope into the z-plane. The x endpoints are
    # bracketed against the orbital backs and the two point singularities.
    def back(g: rocheometry.ComponentGeometry) -> float:
        outer = abs(graph.outer(g.field).position[0])
        return -bounded_surface_root(
            lambda r: g.field.potential(-r, 0.0, z) - level.value,
            outer,
            scale=level.value,
        )

    left, right = back(first), 1 - back(second)

    def width(x: float) -> float:
        cap = brent_root(
            lambda y: (
                1
                - field.component_mass / hypot(x, y, z) ** 3
                - field.companion_mass / hypot(1 - x, y, z) ** 3
            ),
            0.0,
            2.0,
        )
        return bounded_surface_root(
            lambda y: field.potential(x, y, z) - level.value, cap, scale=level.value
        )

    nodes, weights = gauss_legendre(256)
    bounds = (left, 0.0, first.l1().position[0], 1.0, right)
    area = fsum(
        (b - a)
        * pi
        / 2
        * w
        * cos(pi * t / 2)
        * width((a + b) / 2 + (b - a) / 2 * sin(pi * t / 2))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        for t, w in zip(nodes, weights, strict=True)
    )
    assert sum(polygon_area(p) for p in paths) == pytest.approx(area, rel=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("reference", (False, True))
def test_cut_only_disc_has_no_envelope_runs(reference: bool) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    g = rocheometry.ComponentGeometry(0.05, accuracy=_Wired(graph))
    s = g.surface(fill_out_factor=0.5)
    front = max(graph.watershed.rim(g.field, s._depth, a)[0] for a in (0.0, pi / 2))
    plane = rocheometry.AxisNormalPlane("x", (front + g.l1().position[0]) / 2)
    path = single_path(s.section(plane))
    assert path.closed
    assert s.section_envelope(plane) == rocheometry.CompoundBezierPath()


@pytest.mark.slow
@pytest.mark.parametrize("qi", (1e-3, 0.05))
@pytest.mark.parametrize("factor", FACTORS)
def test_first_axial_cut_appears_continuously(qi: float, factor: float) -> None:
    graph = Reference()._computers()
    g = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    s = g.surface(fill_out_factor=factor)
    first = min(graph.watershed.rim(g.field, s._depth, a)[0] for a in (0.0, pi / 2))
    delta = 1e-7 * (g.l1().position[0] - first)
    areas = tuple(
        polygon_area(
            single_path(s.section(rocheometry.AxisNormalPlane("x", first + d)))
        )
        for d in (-delta, 0.0, delta)
    )
    assert max(areas) - min(areas) <= 1e-5 * areas[-1]


def test_section_no_longer_accepts_envelope_flag() -> None:
    from typing import Any

    surface = rocheometry.ComponentGeometry(1.0).lobe()
    with pytest.raises(TypeError, match="envelope_only"):
        single_path(cast(Any, surface).section(envelope_only=True))


@pytest.mark.parametrize("runs", (0, 1, 2))
def test_axial_envelope_compound_cardinality(runs: int) -> None:
    g = rocheometry.ComponentGeometry(0.05)
    s = g.surface(fill_out_factor=0.5)
    a, b = (g._computers.watershed.rim(g.field, s._depth, p)[0] for p in (0.0, pi / 2))
    x = {0: (max(a, b) + g.l1().position[0]) / 2, 1: min(a, b) - 0.01, 2: (a + b) / 2}[
        runs
    ]
    assert g.field.potential(x) > s.potential.value and x < g.l1().position[0]
    arcs = s.section_envelope(rocheometry.AxisNormalPlane("x", x))
    assert isinstance(arcs, rocheometry.CompoundBezierPath)
    assert len(arcs) == runs
    for arc in arcs:
        assert arc.closed == (runs == 1)
        if not arc.closed:
            for y, z in (arc.segments[0].p0, arc.segments[-1].p3):
                assert g.field.potential(x, y, z) == pytest.approx(
                    s.potential.value, abs=2e-14
                )


@pytest.mark.parametrize("qi", (0.001, 0.05, 20.0, 1000.0))
@pytest.mark.parametrize("factor", FACTORS)
def test_ring_and_empty_x_classification(qi: float, factor: float) -> None:
    g = rocheometry.ComponentGeometry(qi)
    s = g.surface(fill_out_factor=factor)
    x1 = g.l1().position[0]
    if qi < 1:
        plane = rocheometry.AxisNormalPlane("x", x1 + 1e-4)
        for reader in (s.section, s.section_envelope):
            with pytest.raises(rocheometry.EmptySectionError):
                reader(plane)
    else:
        largest = max(
            g._computers.watershed.rim(g.field, s._depth, a)[0] for a in (0.0, pi / 2)
        )
        plane = rocheometry.AxisNormalPlane("x", (x1 + largest) / 2)
        for reader in (s.section, s.section_envelope):
            with pytest.raises(
                NotImplementedError,
                match="ring-shaped section of the heavier component's part",
            ):
                reader(plane)


# Every distinct near-L2 SC-03 failure from the contract study. The y=0
# failure at the last qi has no z=0 counterpart in the original failure list.
_NEAR_L2_QI = (
    1e-8,
    1.1138129468526125e-8,
    1.2064393334540251e-8,
    1.2171682082219975e-8,
    1.2366146439773132e-8,
    1.6730227535937007e-8,
    1.9251800202760952e-8,
    2.8588509306973982e-8,
    4.267829196241071e-8,
)
_NEAR_L2_CASES = tuple((q, "y") for q in _NEAR_L2_QI) + tuple(
    (q, "z") for q in _NEAR_L2_QI[:-1]
)


@pytest.mark.parametrize("qi,axis", _NEAR_L2_CASES)
@pytest.mark.parametrize("accuracy", (Fitted(), Reference()))
def test_near_l2_zero_offset_sections_do_not_collapse(
    qi: float,
    axis: SectionAxis,
    accuracy: Fitted | Reference,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Near-L2 zero-offset sections construct within a refinement depth of 48."""
    from rocheometry._computers.section_path import axis_contact

    monkeypatch.setattr(axis_contact, "AXIS_CONTACT_MAX_REFINEMENT_DEPTH", 48)
    selected = rocheometry.ComponentGeometry(qi).surface(fill_out_factor=0.99)
    surface = rocheometry.ComponentGeometry(qi, accuracy=accuracy).surface(
        potential=selected.potential
    )
    plane = rocheometry.AxisNormalPlane(axis, 0.0)
    section = surface.section(plane)
    envelope = surface.section_envelope(plane)
    assert section.paths and envelope.paths
    assert all(path.closed for path in section.paths)
    assert all(not path.closed for path in envelope.paths)


@pytest.mark.parametrize("qi", (1e-8, 1e8))
@pytest.mark.parametrize("axis", ("y", "z"))
@pytest.mark.parametrize("accuracy", (Fitted(), Reference()))
def test_shallow_contact_refinement_has_representable_panels(
    qi: float, axis: SectionAxis, accuracy: Fitted | Reference
) -> None:
    """Measured depths 49--52 are genuine refinement, not collapsed floats."""
    chosen = rocheometry.ComponentGeometry(qi).surface(fill_out_factor=1e-6)
    surface = rocheometry.ComponentGeometry(qi, accuracy=accuracy).surface(
        potential=chosen.potential
    )
    plane = rocheometry.AxisNormalPlane(axis, 0.0)
    assert surface.section(plane).paths
    assert surface.section_envelope(plane).paths
