"""Plane-aware sections are checked against independent radial intersections."""

from math import atan2, cos, hypot, pi, sin
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import (
    TracedSectionPathComputer,
    _section_radius,
)
from rocheometry._computers.surface_radius import BisectionSurfaceRadiusComputer
from rocheometry._planes import SectionAxis
from rocheometry._roche.field import RocheField
from rocheometry._types import ComponentId
from tests._helpers import WholeSection, _depth, _Wired, single_path


@pytest.fixture(scope="module")
def graph() -> ComputerSet:
    return Fitted()._computers()


@pytest.fixture(scope="module")
def tracer(graph: ComputerSet) -> TracedSectionPathComputer:
    return TracedSectionPathComputer(
        radius_computer=graph.radius,
        l1_computer=graph.l1,
        tolerance=1.0e-5,
    )


@pytest.mark.parametrize("qi", [1.0e-3, 0.3, 3.0, 1.0e3])
@pytest.mark.parametrize("fill", [0.8, 0.999, 0.999999, 1.0])
@pytest.mark.parametrize("inclination", [0.0, pi / 6, pi / 3, pi / 2])
def test_centreline_dense_radial_error(
    graph: ComputerSet,
    tracer: TracedSectionPathComputer,
    qi: float,
    fill: float,
    inclination: float,
) -> None:
    surface = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph)).surface(
        volume_radius_fill=fill
    )
    plane = rocheometry.CentrelinePlane(inclination)
    path = single_path(tracer(surface.component.field, surface._depth, plane))
    assert path.closed and path.segments[0].p0 == path.segments[-1].p3
    exact = BisectionSurfaceRadiusComputer(l1_computer=graph.l1)
    scale = surface.volume_radius()
    for segment in path.segments:
        for index in range(1, 16):
            x, u = segment.evaluate(index / 16)
            angle = atan2(u, x)
            radius = exact(
                surface.component.field,
                surface._depth,
                (
                    cos(angle),
                    sin(angle) * cos(inclination),
                    -sin(angle) * sin(inclination),
                ),
            )
            assert abs(hypot(x, u) - radius) / scale <= tracer.tolerance
    if fill == 1.0:
        first = path.segments[0]
        tangent = surface.component.field.critical_upper_tangent(
            first.p0[0], inclination
        )
        direction = (first.p1[0] - first.p0[0], first.p1[1] - first.p0[1])
        assert direction[0] * tangent[1] - direction[1] * tangent[0] == pytest.approx(
            0.0, abs=2e-16
        )


@pytest.mark.parametrize("qi", [1.0e-3, 0.3, 3.0, 1.0e3])
@pytest.mark.parametrize("fill", [0.8, 0.999, 0.999999, 1.0])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_section_dense_radial_error_and_containment(
    graph: ComputerSet,
    tracer: TracedSectionPathComputer,
    qi: float,
    fill: float,
    axis: SectionAxis,
) -> None:
    surface = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph)).surface(
        volume_radius_fill=fill
    )
    field = surface.component.field
    coordinate = ("x", "y", "z").index(axis)
    normal = cast(
        tuple[float, float, float],
        tuple(1.0 if j == coordinate else 0.0 for j in range(3)),
    )
    extent = surface.radius(normal)
    scale = surface.volume_radius()
    l1 = graph.l1(field).position[0]
    for fraction in (0.0, 0.25, 0.5, 0.9, 0.999):
        offset = fraction * extent
        path = single_path(
            tracer(
                field,
                graph.selection.depth(field, fill),
                rocheometry.AxisNormalPlane(axis, offset),
            )
        )
        origin = cast(
            tuple[float, float, float],
            tuple(offset if j == coordinate else 0.0 for j in range(3)),
        )
        for segment in path.segments:
            for index in range(1, 12):
                first, second = segment.evaluate(index / 12)
                angle = atan2(second, first)
                c, s = cos(angle), sin(angle)
                if axis == "x":
                    direction = (0.0, c, s)
                elif axis == "y":
                    direction = (c, 0.0, s)
                else:
                    direction = (c, s, 0.0)
                if offset == 0.0:
                    radius = surface.radius(direction)
                else:
                    radius = _section_radius(
                        field, surface._depth, origin, direction, l1, scale
                    )
                assert abs(hypot(first, second) - radius) / scale <= tracer.tolerance
                if axis == "z":
                    orbital_radius = surface.radius((c, s, 0.0))
                    assert (
                        hypot(first, second)
                        <= orbital_radius + tracer.tolerance * scale
                    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_section_rejects_boundary_exterior_and_companion_regions(
    axis: SectionAxis,
) -> None:
    surface = rocheometry.ComponentGeometry(0.3).lobe()
    coordinate = ("x", "y", "z").index(axis)
    direction = cast(
        tuple[float, float, float],
        tuple(1.0 if j == coordinate else 0.0 for j in range(3)),
    )
    extent = surface.radius(direction)
    for offset in (extent, 1.01 * extent, 2.0):
        with pytest.raises(rocheometry.EmptySectionError, match="strictly inside"):
            single_path(surface.section(rocheometry.AxisNormalPlane(axis, offset)))


def test_negative_offsets_have_identical_shapes() -> None:
    surface = rocheometry.ComponentGeometry(0.3).surface(volume_radius_fill=0.8)
    for axis in ("y", "z"):
        assert single_path(
            surface.section(rocheometry.AxisNormalPlane(axis, 0.1))
        ) == single_path(surface.section(rocheometry.AxisNormalPlane(axis, -0.1)))


@pytest.mark.parametrize(
    "qi, fill", [(0.001, 1.0), (0.3, 0.999999), (3.0, 0.8), (1000.0, 0.999)]
)
def test_reference_graph_traces_to_its_tighter_tolerance(
    qi: float, fill: float
) -> None:
    graph = Reference()._computers()
    surface = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph)).surface(
        volume_radius_fill=fill
    )
    for plane in (rocheometry.ORBITAL_PLANE, rocheometry.CentrelinePlane(pi / 3)):
        path = single_path(surface.section(plane))
        for segment in path.segments:
            for index in range(1, 12):
                x, u = segment.evaluate(index / 12)
                angle = atan2(u, x)
                radius = surface.radius(
                    (
                        cos(angle),
                        sin(angle) * cos(plane.inclination),
                        -sin(angle) * sin(plane.inclination),
                    )
                )
                assert abs(hypot(x, u) - radius) / surface.volume_radius() <= 1e-10


def test_default_router_and_reference_graph_share_dependencies(
    graph: ComputerSet,
) -> None:
    router = cast(PlaneRoutedSectionPathComputer, graph.section)  # type: ignore[attr-defined]
    assert isinstance(router.centreline, GaugeFixedSectionPathComputer)
    assert isinstance(router.section, TracedSectionPathComputer)
    assert router.section.radius_computer is graph.radius
    assert router.section.l1_computer is graph.l1
    assert Reference()._computers() is Reference()._computers()


@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_zero_z_section_uses_the_same_fitted_path(fill: float) -> None:
    binary = rocheometry.BinaryGeometry(0.3)
    plane = rocheometry.AxisNormalPlane("z", -0.0)
    for component_id in (1, 2):
        surface = binary.component(cast(ComponentId, component_id)).surface(
            volume_radius_fill=fill
        )
        assert single_path(surface.section(plane)) == single_path(
            surface.section(rocheometry.ORBITAL_PLANE)
        )
        intrinsic = surface.intrinsic_geometry
        field = intrinsic.component.field
        orbital = single_path(intrinsic.section())
        assert len(orbital.segments) == 52
        assert (
            single_path(binary._computers.section(field, intrinsic._depth, plane))
            == orbital
        )


@pytest.mark.parametrize("fill", [0.8, 1.0])
@pytest.mark.parametrize("axis", ["y", "z"])
def test_zero_sections_share_the_centreline_tracer(
    graph: ComputerSet,
    tracer: TracedSectionPathComputer,
    fill: float,
    axis: SectionAxis,
) -> None:
    field = RocheField(0.3)
    centreline = rocheometry.CentrelinePlane(pi / 2 if axis == "y" else 0.0)
    expected = single_path(
        tracer(field, graph.selection.depth(field, fill), centreline)
    )
    if axis == "y":
        expected = expected.transformed(
            rocheometry.AffineTransform2D(yy=-1.0)
        ).reversed()
    assert (
        single_path(
            tracer(
                field,
                graph.selection.depth(field, fill),
                rocheometry.AxisNormalPlane(axis, 0.0),
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    ("plane", "canonical", "orbital"),
    [
        (rocheometry.AxisNormalPlane("z", 0.0), rocheometry.ORBITAL_PLANE, True),
        (
            rocheometry.AxisNormalPlane("y", 0.0),
            rocheometry.CentrelinePlane(pi / 2),
            True,
        ),
        (
            rocheometry.AxisNormalPlane("x", 0.0),
            rocheometry.AxisNormalPlane("x", 0.0),
            False,
        ),
        (
            rocheometry.AxisNormalPlane("z", 1e-15),
            rocheometry.AxisNormalPlane("z", 1e-15),
            False,
        ),
        (
            rocheometry.AxisNormalPlane("y", 0.1),
            rocheometry.AxisNormalPlane("y", 0.1),
            False,
        ),
    ],
)
def test_router_canonicalizes_planes_and_preserves_section_coordinates(
    plane: rocheometry.SectionPlane, canonical: rocheometry.SectionPlane, orbital: bool
) -> None:
    field = RocheField(0.3)
    # Deliberately asymmetric: a Roche-symmetric path would hide a missing
    # coordinate reflection or an accidental change in traversal direction.
    path = rocheometry.CompoundBezierPath(
        (
            rocheometry.BezierPath(
                (rocheometry.CubicBezier((1, 0), (2, 3), (-4, -5), (1, 0)),)
            ),
        )
    )
    calls: list[tuple[bool, rocheometry.SectionPlane]] = []

    def orbital_computer(
        supplied: RocheField,
        potential: float,
        selected: rocheometry.SectionPlane,
        /,
    ) -> rocheometry.CompoundBezierPath:
        assert supplied is field and potential == _depth(
            field, rocheometry.Potential.jacobi(3.0)
        )
        calls.append((True, selected))
        return path

    def other_computer(
        supplied: RocheField,
        potential: float,
        selected: rocheometry.SectionPlane,
        /,
    ) -> rocheometry.CompoundBezierPath:
        assert supplied is field and potential == _depth(
            field, rocheometry.Potential.jacobi(3.0)
        )
        calls.append((False, selected))
        return path

    router = PlaneRoutedSectionPathComputer(
        centreline=WholeSection(orbital_computer), section=WholeSection(other_computer)
    )
    output = router(field, _depth(field, rocheometry.Potential.jacobi(3.0)), plane)
    assert calls == [(orbital, canonical)]
    if plane == rocheometry.AxisNormalPlane("y", 0.0):
        assert (
            output
            == path.transformed(rocheometry.AffineTransform2D(yy=-1.0)).reversed()
        )
        assert output != path
    else:
        assert output is path


def test_plane_normalization_and_hashability() -> None:
    assert rocheometry.CentrelinePlane(pi - 0.3).inclination == pytest.approx(0.3)
    assert rocheometry.CentrelinePlane(pi) == rocheometry.ORBITAL_PLANE
    assert rocheometry.CentrelinePlane(pi / 2).inclination == pi / 2
    assert (
        len(
            {
                rocheometry.AxisNormalPlane("z", 0.1),
                rocheometry.AxisNormalPlane("z", 0.1),
            }
        )
        == 1
    )
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="inclination"):
            rocheometry.CentrelinePlane(value)
    with pytest.raises(TypeError, match="inclination"):
        rocheometry.CentrelinePlane(True)
    with pytest.raises(ValueError, match="axis"):
        rocheometry.AxisNormalPlane(cast(SectionAxis, "w"), 0.1)


@pytest.mark.parametrize(
    "plane",
    [
        rocheometry.CentrelinePlane(pi / 6),
        rocheometry.AxisNormalPlane("x", 0.1),
        rocheometry.AxisNormalPlane("y", 0.1),
        rocheometry.AxisNormalPlane("z", 0.1),
        rocheometry.AxisNormalPlane("y", 0.0),
        rocheometry.AxisNormalPlane("z", 0.0),
    ],
)
def test_binary_section_transforms_respect_the_plane_chart(
    plane: rocheometry.SectionPlane,
) -> None:
    binary = rocheometry.BinaryGeometry(0.3, separation=2.5, frame="centre_of_mass")
    surface = binary.component(2).surface(volume_radius_fill=0.8)
    translation = 2.5 * (0.3 / (1 + 0.3))
    local_plane = plane
    if isinstance(plane, rocheometry.AxisNormalPlane):
        if plane.axis == "x":
            offset, _ = (
                surface.component._output_transform()
                .inverse()
                .apply((plane.offset, 0.0))
            )
        elif plane.axis == "y":
            offset = -plane.offset / 2.5
        else:
            offset = plane.offset / 2.5
        local_plane = rocheometry.AxisNormalPlane(plane.axis, offset)
    local = single_path(surface.intrinsic_geometry.section(local_plane))
    output = single_path(surface.section(plane))
    for original, transformed in zip(local.segments, output.segments, strict=True):
        for a, b in zip(
            (original.p0, original.p1, original.p2, original.p3),
            (transformed.p0, transformed.p1, transformed.p2, transformed.p3),
            strict=True,
        ):
            if isinstance(plane, rocheometry.CentrelinePlane) or plane.axis == "z":
                expected = (translation - 2.5 * a[0], -2.5 * a[1])
            elif plane.axis == "y":
                expected = (translation - 2.5 * a[0], 2.5 * a[1])
            else:
                expected = (-2.5 * a[0], 2.5 * a[1])
            assert b == pytest.approx(expected, abs=2e-15)


def test_refinement_limit_raises_instead_of_returning_an_unresolved_path(
    graph: ComputerSet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rocheometry._computers.section_path.traced as implementation

    monkeypatch.setattr(implementation, "SECTION_MAX_REFINEMENT_DEPTH", 0)
    tracer = TracedSectionPathComputer(
        radius_computer=graph.radius,
        l1_computer=graph.l1,
        tolerance=1e-15,
    )
    with pytest.raises(RuntimeError, match="depth limit"):
        single_path(
            tracer(
                RocheField(0.3),
                graph.selection.depth(RocheField(0.3), 0.999999),
                rocheometry.CentrelinePlane(pi / 6),
            )
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_tracer_rejects_invalid_tolerance(graph: ComputerSet, tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        TracedSectionPathComputer(
            radius_computer=graph.radius,
            l1_computer=graph.l1,
            tolerance=tolerance,
        )


def test_reference_geometry_pickle_reconnects_to_the_reference_graph() -> None:
    import pickle

    original = rocheometry.BinaryGeometry(0.3, accuracy=Reference())
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert (
        restored.geometry1._computers
        is restored.geometry2._computers
        is Reference()._computers()
    )


def test_reference_neck_section_meets_its_radial_tolerance() -> None:
    graph = Reference()._computers()
    surface = rocheometry.ComponentGeometry(0.001, accuracy=_Wired(graph)).lobe()
    field = surface.component.field
    offset = 0.999 * surface.named_radius("point")
    path = single_path(surface.section(rocheometry.AxisNormalPlane("x", offset)))
    for segment in path.segments:
        for index in range(1, 17):
            y, z = segment.evaluate(index / 17)
            angle = atan2(z, y)
            exact = _section_radius(
                field,
                surface._depth,
                (offset, 0.0, 0.0),
                (0.0, cos(angle), sin(angle)),
                surface.named_radius("point"),
                surface.volume_radius(),
            )
            assert abs(hypot(y, z) - exact) / surface.volume_radius() <= 1e-10


def test_z_sections_lie_inside_the_traced_orbital_path(
    tracer: TracedSectionPathComputer, graph: ComputerSet
) -> None:
    from rocheometry._numerics.roots import brent_root

    field = RocheField(0.3)
    surface = rocheometry.ComponentGeometry(0.3, accuracy=_Wired(graph)).surface(
        volume_radius_fill=0.999999
    )
    orbital = single_path(tracer(field, surface._depth, rocheometry.ORBITAL_PLANE))
    for fraction in (0.0, 0.25, 0.5, 0.9, 0.999):
        section = single_path(
            tracer(
                field,
                surface._depth,
                rocheometry.AxisNormalPlane(
                    "z", fraction * surface.named_radius("pole")
                ),
            )
        )
        for segment in section.segments[: len(section.segments) // 2]:
            for index in range(1, 9):
                x, y = segment.evaluate(index / 9)
                angle = atan2(y, x)
                for boundary in orbital.segments[: len(orbital.segments) // 2]:
                    if (
                        atan2(boundary.p0[1], boundary.p0[0])
                        <= angle
                        <= atan2(boundary.p3[1], boundary.p3[0])
                    ):
                        c, s = cos(angle), sin(angle)

                        def residual(t: float) -> float:
                            bx, by = boundary.evaluate(t)
                            return by * c - bx * s

                        t = brent_root(residual, 0.0, 1.0)
                        radial_bound = hypot(*boundary.evaluate(t))
                        assert (
                            hypot(x, y)
                            <= radial_bound
                            + 2 * tracer.tolerance * surface.volume_radius()
                        )
                        break
                else:
                    pytest.fail(
                        "orbital path does not contain the section's polar direction"
                    )


@pytest.mark.parametrize("accuracy", (Fitted(), Reference()))
@pytest.mark.parametrize(
    "qi,fill,axis,offset",
    (
        (1e4, 0.99, "x", 0.0),
        (1e6, 0.999999, "y", 0.79045549947269),
    ),
)
def test_axis_sections_bracket_the_stellar_crossing(
    accuracy: Fitted | Reference,
    qi: float,
    fill: float,
    axis: SectionAxis,
    offset: float,
) -> None:
    """SC-01/02 must not step over the outside interval into centrifugal space."""
    chosen = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill)
    surface = rocheometry.ComponentGeometry(qi, accuracy=accuracy).surface(
        potential=chosen.potential
    )
    plane = rocheometry.AxisNormalPlane(axis, offset)
    section = surface.section(plane)
    assert section.paths and all(p.closed for p in section.paths)
    assert surface.section_envelope(plane) == section
