"""Plane charts are shared by sections, projections and binary views."""

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from math import atan2, cos, hypot, nextafter, pi, sin
from typing import cast, overload

import pytest

import rocheometry
from rocheometry._accuracy import _Accuracy
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._geometry import Point2D, Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import SectionAxis
from rocheometry._roche.field import RocheField
from rocheometry._types import BinaryFrame, ComponentId
from tests._helpers import _Wired
from tests.computers.silhouette.test_projected_outlines import projected

PLANES = (
    rocheometry.ORBITAL_PLANE,
    rocheometry.CentrelinePlane(pi / 6),
    rocheometry.CentrelinePlane(pi / 2),
    rocheometry.AxisNormalPlane("x", 0.17),
    rocheometry.AxisNormalPlane("y", -0.23),
    rocheometry.AxisNormalPlane("z", 0.31),
)

INCLINATION_ENTRIES = (
    pytest.param(rocheometry.CentrelinePlane, id="plane"),
    pytest.param(
        partial(RocheField(1.0).centreline_plane_hessian, 0.5, 0.0), id="hessian"
    ),
    pytest.param(partial(RocheField(1.0).critical_upper_tangent, 0.5), id="tangent"),
    pytest.param(
        partial(RocheField(1.0).critical_branch_curvature, 0.5), id="curvature"
    ),
)


@pytest.mark.parametrize("entry", INCLINATION_ENTRIES)
@pytest.mark.parametrize("inclination", (0.0, pi / 2, pi - 1e-12, pi))
def test_public_inclinations_accept_domain_and_fold_supplements(
    entry: Callable[[float], object], inclination: float
) -> None:
    folded = min(inclination, pi - inclination)
    assert entry(inclination) == entry(folded)


@pytest.mark.parametrize("entry", INCLINATION_ENTRIES)
@pytest.mark.parametrize(
    "inclination", (70, -0.1, 3.2, 2 * pi, nextafter(0.0, -1.0), nextafter(pi, 4.0))
)
def test_public_inclinations_reject_outside_domain(
    entry: Callable[[float], object], inclination: float
) -> None:
    with pytest.raises(ValueError, match=r"math\.radians\(\)"):
        entry(inclination)


@pytest.mark.parametrize("entry", INCLINATION_ENTRIES)
@pytest.mark.parametrize(
    "inclination", (False, True, float("inf"), -float("inf"), float("nan"))
)
def test_public_inclinations_still_reject_bools_and_nonfinite_values(
    entry: Callable[[float], object], inclination: float
) -> None:
    error = TypeError if isinstance(inclination, bool) else ValueError
    with pytest.raises(error, match="inclination"):
        entry(inclination)


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("point", ((0.0, 0.0), (0.7, -0.43), (-1.2, 0.31)))
def test_plane_project_and_lift_are_inverse(
    plane: rocheometry.SectionPlane, point: Point2D
) -> None:
    assert plane.project(plane.lift(point)) == pytest.approx(
        point, rel=1e-15, abs=1e-15
    )
    q = (0.31, -0.47, 0.83)
    foot = plane.lift(plane.project(q))
    assert plane.project(foot) == pytest.approx(plane.project(q), rel=1e-15, abs=1e-15)
    if isinstance(plane, rocheometry.AxisNormalPlane):
        assert foot[("x", "y", "z").index(plane.axis)] == plane.offset
    else:
        c, s = cos(plane.inclination), sin(plane.inclination)
        assert foot[1] * s + foot[2] * c == pytest.approx(0.0, abs=1e-15)
        assert (q[1] - foot[1]) * c - (q[2] - foot[2]) * s == pytest.approx(
            0.0, abs=1e-15
        )


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
@pytest.mark.parametrize("orientation", ("x", "y", "z", "centreline"))
def test_component_section_points_lift_onto_surface(
    reference: bool, kind: str, orientation: str
) -> None:
    geometry = rocheometry.ComponentGeometry(
        0.3, accuracy=rocheometry.Reference() if reference else rocheometry.Fitted()
    )
    surface = _surface(geometry, kind)
    plane = (
        rocheometry.CentrelinePlane(pi / 3)
        if orientation == "centreline"
        else rocheometry.AxisNormalPlane(cast(SectionAxis, orientation), 0.04)
    )
    bound = 1e-10 if reference else 3e-4 if orientation == "centreline" else 1e-5
    for path in surface.section_envelope(plane):
        for point in path.sample(137):
            _check_residual(surface, plane.lift(point), bound)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
@pytest.mark.parametrize("orientation", ("x", "y", "z", "centreline"))
@pytest.mark.parametrize("component_id", (1, 2))
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
def test_binary_section_points_lift_in_view_frame(
    reference: bool,
    kind: str,
    orientation: str,
    component_id: ComponentId,
    frame: BinaryFrame,
) -> None:
    binary = rocheometry.BinaryGeometry(
        0.3,
        separation=3.7,
        frame=frame,
        accuracy=rocheometry.Reference() if reference else rocheometry.Fitted(),
    )
    component = binary.component(component_id)
    surface = _surface(component, kind)
    plane = (
        rocheometry.CentrelinePlane(pi / 3)
        if orientation == "centreline"
        else rocheometry.AxisNormalPlane(
            cast(SectionAxis, orientation),
            (component.position[0] if orientation == "x" else 0)
            + 0.04 * binary.separation,
        )
    )
    bound = 1e-10 if reference else 3e-4 if orientation == "centreline" else 1e-5
    for path in surface.section_envelope(plane):
        for point in path.sample(137):
            _check_residual(
                surface.intrinsic_geometry,
                _local_position(component, plane.lift(point)),
                bound,
            )


@overload
def _surface(
    geometry: rocheometry.ComponentGeometry, kind: str
) -> rocheometry.ComponentSurfaceGeometry: ...


@overload
def _surface(
    geometry: rocheometry.BinaryComponentView, kind: str
) -> rocheometry.BinaryComponentSurfaceView: ...


def _surface(
    geometry: rocheometry.ComponentGeometry | rocheometry.BinaryComponentView, kind: str
) -> rocheometry.ComponentSurfaceGeometry | rocheometry.BinaryComponentSurfaceView:
    if kind == "detached":
        return geometry.surface(volume_radius_fill=0.8)
    if kind == "critical":
        return geometry.lobe()
    return geometry.surface(fill_out_factor=0.5)


def _local_position(
    component: rocheometry.BinaryComponentView, point: Point3D
) -> Point3D:
    sign = 1 if component.component_id == 1 else -1
    a = component.binary.separation
    return (
        (point[0] - component.position[0]) / (sign * a),
        point[1] / (sign * a),
        point[2] / a,
    )


def _check_residual(
    surface: rocheometry.ComponentSurfaceGeometry, point: Point3D, bound: float
) -> None:
    field = surface.component.field
    residual = abs(field.potential(*point) - surface.potential.value)
    scale = surface.named_radius("pole") * hypot(*field.gradient(*point))
    assert residual <= bound * scale + 2e-14 * abs(surface.potential.value)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
def test_component_two_y_slice_uses_view_offset(
    reference: bool, frame: BinaryFrame
) -> None:
    a, c = 3.7, 0.13
    binary = rocheometry.BinaryGeometry(
        0.3,
        separation=a,
        frame=frame,
        accuracy=rocheometry.Reference() if reference else rocheometry.Fitted(),
    )
    surface = binary.component(2).surface(volume_radius_fill=0.8)
    plane = rocheometry.AxisNormalPlane("y", c)
    old_local = surface.intrinsic_geometry.section(
        rocheometry.AxisNormalPlane("y", -c / a)
    )
    expected = old_local.transformed(
        rocheometry.AffineTransform2D(xx=-a, yy=a, tx=surface.component.position[0])
    )
    actual = surface.section(plane)
    for old, new in zip(expected.paths, actual.paths, strict=True):
        for p, q in zip(old.segments, new.segments, strict=True):
            for a2, b2 in zip(
                (p.p0, p.p1, p.p2, p.p3), (q.p0, q.p1, q.p2, q.p3), strict=True
            ):
                assert a2 == pytest.approx(b2, rel=1e-15, abs=1e-15)
    for path in actual:
        assert all(plane.lift(p)[1] == c for p in path.sample(37))


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("kind", ("detached", "critical"))
def test_face_on_silhouette_agrees_with_section_to_roundoff(
    reference: bool, kind: str
) -> None:
    accuracy: _Accuracy = rocheometry.Reference() if reference else rocheometry.Fitted()
    if not reference:
        # Equal schedules isolate the frame/plane identity, while the packaged
        # section count remains unchanged at 26 per half.
        graph = accuracy._computers()
        section = GaugeFixedSectionPathComputer(
            radius_computer=graph.radius,
            l1_computer=graph.l1,
            outer_computer=graph.outer,
            watershed_computer=graph.watershed,
            segments_per_half=44,
        )
        accuracy = _Wired(replace(graph, section=section))
    sources: list[rocheometry.ComponentGeometry | rocheometry.BinaryComponentView] = [
        rocheometry.ComponentGeometry(0.3, accuracy=accuracy)
    ]
    for frame in ("binary", "centre_of_mass"):
        binary = rocheometry.BinaryGeometry(
            0.3, separation=3.7, frame=cast(BinaryFrame, frame), accuracy=accuracy
        )
        sources.extend(binary.component(k) for k in (1, 2))
    for geometry in sources:
        surface = _surface(geometry, kind)
        for actual, expected in (
            (surface.silhouette(), surface.section()),
            (surface.silhouette_envelope(), surface.section_envelope()),
            (
                surface.silhouette(rocheometry.ORBITAL_PLANE),
                surface.section(rocheometry.ORBITAL_PLANE),
            ),
        ):
            for a, b in zip(actual.paths, expected.paths, strict=True):
                assert a.closed == b.closed
                for p, q in zip(a.segments, b.segments, strict=True):
                    for u, v in zip(
                        (p.p0, p.p1, p.p2, p.p3),
                        (q.p0, q.p1, q.p2, q.p3),
                        strict=True,
                    ):
                        assert hypot(u[0] - v[0], u[1] - v[1]) <= 1e-15 * hypot(*v)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_silhouettes_reject_axis_normal_planes(
    reference: bool, axis: SectionAxis
) -> None:
    accuracy = rocheometry.Reference() if reference else rocheometry.Fitted()
    surfaces = (
        rocheometry.ComponentGeometry(0.3, accuracy=accuracy).lobe(),
        rocheometry.BinaryGeometry(0.3, accuracy=accuracy).component(2).lobe(),
    )
    message = "silhouettes project along the line of sight, normal to a CentrelinePlane"
    for surface in surfaces:
        for reader in (surface.silhouette, surface.silhouette_envelope):
            with pytest.raises(TypeError, match=f"^{message}$"):
                reader(
                    cast(
                        rocheometry.CentrelinePlane,
                        rocheometry.AxisNormalPlane(axis, 0.0),
                    )
                )


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("inclination", (0.0, pi / 6, pi / 2))
@pytest.mark.parametrize("component_id", (1, 2))
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
def test_view_silhouette_contains_reference_quadrature_and_touches_limb(
    reference: bool,
    inclination: float,
    component_id: ComponentId,
    frame: BinaryFrame,
) -> None:
    accuracy = rocheometry.Reference() if reference else rocheometry.Fitted()
    binary = rocheometry.BinaryGeometry(
        0.3, separation=3.7, frame=frame, accuracy=accuracy
    )
    component = binary.component(component_id)
    surface = component.surface(volume_radius_fill=0.8)
    plane = rocheometry.CentrelinePlane(inclination)
    containment_excess, limb_error = _view_projection_errors(surface, plane)
    # The fitted outline's documented bound uses the pole radius. Preserve
    # its construction; this check validates the plane and frame mapping.
    tolerance = 1e-8 if reference else 3e-4
    assert containment_excess <= tolerance
    assert limb_error <= tolerance


def _view_projection_errors(
    surface: rocheometry.BinaryComponentSurfaceView, plane: rocheometry.CentrelinePlane
) -> tuple[float, float]:
    """Return worst containment excess and exact-limb error in pole radii."""
    component = surface.component
    binary = component.binary
    path = surface.silhouette(plane).paths[0]
    origin = plane.project(component.position)
    # Build an independent radial lookup from the returned view-space cubics.
    upper = sorted(
        (min(atan2(p[1] - origin[1], p[0] - origin[0]) for p in (s.p0, s.p3)), s)
        for s in path.segments
        if s.evaluate(0.5)[1] > origin[1]
    )
    starts = [a for a, _ in upper]

    def radius(angle: float) -> float:
        _, segment = upper[max(0, bisect_right(starts, angle) - 1)]
        c, s = cos(angle), sin(angle)

        def cross(t: float) -> float:
            x, y = segment.evaluate(t)
            return c * (y - origin[1]) - s * (x - origin[0])

        ends = (cross(0.0), cross(1.0))
        if min(map(abs, ends)) <= 1e-14 * pole:
            t = 0.0 if abs(ends[0]) <= abs(ends[1]) else 1.0
        else:
            t = brent_root(
                cross, 0.0, 1.0, absolute_tolerance=1e-15, relative_tolerance=1e-15
            )
        p = segment.evaluate(t)
        return hypot(p[0] - origin[0], p[1] - origin[1])

    pole = surface.named_radius("pole")
    reference_surface = (
        rocheometry.BinaryGeometry(
            binary.q,
            separation=binary.separation,
            frame=binary.frame,
            accuracy=rocheometry.Reference(),
        )
        .component(component.component_id)
        .surface(potential=surface.potential)
    )
    containment_excess = 0.0
    for point in reference_surface.quadrature().positions:
        x, t = plane.project(point)
        r = hypot(x - origin[0], t - origin[1])
        angle = atan2(abs(t - origin[1]), x - origin[0])
        containment_excess = max(containment_excess, (r - radius(angle)) / pole)

    # Exact per-angle maxima touch the outline; finite quadrature nodes need
    # only lie inside it, since they need not land on a tangent line of sight.
    local = projected(surface.intrinsic_geometry, plane.inclination)
    limb_error = 0.0
    for k in range(61):
        angle = (k + 0.5) * pi / 61
        r = local.radius(angle)
        point = local.maximum(r * cos(angle), r * sin(angle))
        if component.component_id == 2:
            point = (point[0], point[1], -point[2])
        x, t = plane.project(component._output_position(point))
        view_angle = atan2(abs(t - origin[1]), x - origin[0])
        limb_error = max(
            limb_error,
            abs(hypot(x - origin[0], t - origin[1]) - radius(view_angle)) / pole,
        )
    return containment_excess, limb_error
