"""Disconnected watershed collars are ordinary compound planar regions."""

from collections import deque
from math import atan2, cos, fsum, hypot, pi, sin, sqrt
from typing import Literal

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._numerics.roots import brent_root
from tests._helpers import _depth, _Wired


def signed_area(path: rocheometry.BezierPath) -> float:
    points = tuple(s.evaluate(k / 256) for s in path.segments for k in range(256))
    return (
        fsum(
            p[0] * q[1] - q[0] * p[1]
            for p, q in zip(points, points[1:] + points[:1], strict=True)
        )
        / 2
    )


def total_area(path: rocheometry.CompoundBezierPath) -> float:
    return sum(signed_area(p) for p in path)


@pytest.mark.parametrize("reference", (False, True))
def test_reported_separated_collar(reference: bool) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    g = rocheometry.ComponentGeometry(0.001, accuracy=_Wired(graph))
    surface = g.surface(fill_out_factor=0.5)
    z = 0.999 * graph.watershed.rim(g.field, surface._depth, pi / 2)[2]
    plane = rocheometry.AxisNormalPlane("z", z)
    result = surface.section(plane)
    envelope = surface.section_envelope(plane)
    assert isinstance(result, rocheometry.CompoundBezierPath) and len(result) == 2
    star, collar = result.paths
    assert star.segments[0].p0 == pytest.approx((0.06629094756929682, 0), abs=1e-9)
    assert collar.segments[0].p0 == pytest.approx((0.06713910161249002, 0), abs=1e-9)
    assert collar.bounds[0] == pytest.approx(0.06697510360609804, abs=1e-9)
    assert star.bounds[2] < collar.bounds[0]
    for piece in result:
        assert piece.closed and signed_area(piece) > 0
        assert piece.segments[0].p0[1] == 0
        assert piece.segments[0].p0[0] == piece.bounds[2]
        assert piece.segments[0].p1[1] > 0
        assert piece.segments[-1].p3 == piece.segments[0].p0
    assert len(envelope) == 2
    assert envelope.paths[0] == star
    assert not envelope.paths[1].closed
    assert all(s in collar.segments for s in envelope.paths[1].segments)
    assert surface.section(rocheometry.AxisNormalPlane("z", -z)) == result


@pytest.mark.parametrize("factor", (-0.5, 0.0, 0.5))
@pytest.mark.parametrize(
    "plane",
    (
        rocheometry.ORBITAL_PLANE,
        rocheometry.CentrelinePlane(pi / 3),
        rocheometry.AxisNormalPlane("x", 0.0),
        rocheometry.AxisNormalPlane("y", 0.01),
        rocheometry.AxisNormalPlane("z", 0.01),
    ),
)
def test_ordinary_public_sections_are_single_piece_compounds(
    factor: float, plane: rocheometry.SectionPlane
) -> None:
    for g in (
        rocheometry.ComponentGeometry(1.0),
        rocheometry.BinaryGeometry(1.0).component(1),
    ):
        result = g.surface(fill_out_factor=factor).section(plane)
        assert isinstance(result, rocheometry.CompoundBezierPath)
        assert len(result) == 1 and result.paths[0].closed


@pytest.mark.parametrize("q, component_id", ((0.001, 1), (1000.0, 2)))
def test_binary_transforms_every_separated_piece(
    q: float, component_id: Literal[1, 2]
) -> None:
    binary = rocheometry.BinaryGeometry(q, separation=3.0, frame="centre_of_mass")
    surface = binary.component(component_id).surface(fill_out_factor=0.5)
    local = surface.intrinsic_geometry
    g = local.component
    z = 0.999 * g._computers.watershed.rim(g.field, local._depth, pi / 2)[2]
    plane = rocheometry.AxisNormalPlane("z", binary.separation * z)
    local_plane = rocheometry.AxisNormalPlane("z", plane.offset / binary.separation)
    assert len(surface.section(plane)) == 2
    transform = surface.component._output_transform()
    assert surface.section(plane) == local.section(local_plane).transformed(transform)
    assert surface.section_envelope(plane) == local.section_envelope(
        local_plane
    ).transformed(transform)


def separation_level(
    g: rocheometry.ComponentGeometry, u: rocheometry.Potential, z: float
) -> float:
    """Independent axis minimum, with no section construction."""
    front = g._computers.watershed.cut_height(g.field, z, pi / 2)
    x = brent_root(lambda x: g.field.gradient(x, 0, z)[0], 0, front)
    return g.field.potential(x, 0, z) - u.value


@pytest.mark.slow
@pytest.mark.parametrize("reference", (False, True))
def test_area_is_continuous_when_collar_separates(reference: bool) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    g = rocheometry.ComponentGeometry(0.001, accuracy=_Wired(graph))
    surface = g.surface(fill_out_factor=0.5)
    rz = graph.watershed.rim(g.field, surface._depth, pi / 2)[2]
    z0 = brent_root(
        lambda z: separation_level(g, surface.potential, z), 0.995 * rz, 0.999 * rz
    )
    differences = []
    for epsilon in (1e-6, 1e-7):
        paths = tuple(
            surface.section(rocheometry.AxisNormalPlane("z", z0 + d * epsilon * rz))
            for d in (-1, 1)
        )
        assert tuple(len(p) for p in paths) == (1, 2)
        areas = tuple(total_area(p) for p in paths)
        differences.append(abs(areas[0] - areas[1]) / max(areas))
    assert differences[1] < differences[0]
    assert differences[1] < 1e-5


@pytest.mark.slow
def test_separated_parts_tile_independent_envelope_slice() -> None:
    graph = Reference()._computers()
    g = rocheometry.ComponentGeometry(0.001, accuracy=_Wired(graph))
    other = rocheometry.ComponentGeometry(1000.0, accuracy=_Wired(graph))
    s = g.surface(fill_out_factor=0.5)
    z = 0.999 * graph.watershed.rim(g.field, s._depth, pi / 2)[2]
    parts = tuple(
        c.surface(potential=s.potential).section(rocheometry.AxisNormalPlane("z", z))
        for c in (g, other)
    )
    assert len(parts[0]) == 2
    field = g.field

    def residual(x: float) -> float:
        return field.potential(x, 0, z) - s.potential.value

    minimum = brent_root(lambda x: field.gradient(x, 0, z)[0], 0, g.l1().position[0])
    left = brent_root(residual, -abs(graph.outer(field).position[0]), 0)
    a = brent_root(residual, 0, minimum)
    b = brent_root(residual, minimum, 1)
    right = brent_root(residual, 1, 1 + abs(graph.outer(other.field).position[0]))

    def width(x: float) -> float:
        cap = brent_root(
            lambda y: (
                1
                - field.component_mass / hypot(x, y, z) ** 3
                - field.companion_mass / hypot(x - 1, y, z) ** 3
            ),
            0,
            2,
        )
        return brent_root(
            lambda y: field.potential(x, y, z) - s.potential.value, 0, cap
        )

    nodes, weights = gauss_legendre(256)
    intervals = (
        (left, a),
        (b, g.l1().position[0]),
        (g.l1().position[0], 1),
        (1, right),
    )
    exact = fsum(
        (hi - lo)
        * pi
        / 2
        * w
        * cos(pi * t / 2)
        * width((lo + hi) / 2 + (hi - lo) / 2 * sin(pi * t / 2))
        for lo, hi in intervals
        for t, w in zip(nodes, weights, strict=True)
    )
    assert sum(total_area(p) for p in parts) == pytest.approx(exact, rel=1e-8)


def grid_piece_count(
    g: rocheometry.ComponentGeometry,
    level: rocheometry.Potential,
    plane: rocheometry.AxisNormalPlane,
) -> int:
    """Flood a dense plane grid, evaluating field and height independently."""
    nx, ny = 2049, 1025
    outer = abs(g._computers.outer(g.field).position[0])
    offset = plane.offset
    normal = 1 if plane.axis == "y" else 2
    axis_angle = 0.0 if normal == 1 else pi / 2
    rim = g._computers.watershed.rim(g.field, _depth(g.field, level), axis_angle)
    cut = g._computers.watershed.cut_height(g.field, offset, axis_angle)
    end = 0.0
    if offset < rim[normal]:
        alpha = brent_root(
            lambda p: (
                g._computers.watershed.rim(g.field, _depth(g.field, level), p)[normal]
                - offset
            ),
            0,
            pi / 2,
        )
        end = g._computers.watershed.rim(g.field, _depth(g.field, level), alpha)[
            3 - normal
        ]
    span = sqrt(outer**2 - offset * offset)
    xs = tuple(-span + (cut + span) * (1 - (1 - i / (nx - 1)) ** 4) for i in range(nx))
    ys = tuple(
        span
        * ((j - (ny - 1) / 2) / ((ny - 1) / 2)) ** 2
        * (1 if j >= (ny - 1) / 2 else -1)
        for j in range(ny)
    )
    occupied = bytearray(nx * ny)
    for j, y in enumerate(ys):
        psi = atan2(abs(y), offset) if normal == 1 else atan2(offset, abs(y))
        height = (
            g._computers.watershed.cut_height(g.field, hypot(y, offset), psi)
            if abs(y) < end
            else rim[0]
        )
        for i, x in enumerate(xs):
            if (
                x < height
                and x * x + y * y < span * span
                and g.field.potential(
                    x, offset if normal == 1 else y, y if normal == 1 else offset
                )
                >= level.value
            ):
                occupied[j * nx + i] = 1
    count = 0
    for root in range(nx * ny):
        if not occupied[root]:
            continue
        count += 1
        occupied[root] = 0
        queue = deque((root,))
        crosses_axis = False
        while queue:
            k = queue.popleft()
            crosses_axis |= k // nx == (ny - 1) // 2
            for n in (
                k - nx,
                k + nx,
                k - 1 if k % nx else -1,
                k + 1 if k % nx < nx - 1 else -1,
            ):
                if 0 <= n < len(occupied) and occupied[n]:
                    occupied[n] = 0
                    queue.append(n)
        assert crosses_axis
    return count


@pytest.mark.slow
@pytest.mark.parametrize("qi", (0.001, 0.05, 1000.0))
@pytest.mark.parametrize("fraction", (0.6, 0.999))
@pytest.mark.parametrize("axis", ("y", "z"))
def test_dense_grid_piece_count_and_single_axis_minimum(
    qi: float, fraction: float, axis: Literal["y", "z"]
) -> None:
    g = rocheometry.ComponentGeometry(qi)
    s = g.surface(fill_out_factor=0.5)
    normal = 1 if axis == "y" else 2
    angle = 0.0 if normal == 1 else pi / 2
    offset = fraction * g._computers.watershed.rim(g.field, s._depth, angle)[normal]
    cut = g._computers.watershed.cut_height(g.field, offset, angle)
    derivatives = tuple(
        g.field.gradient(
            cut * i / 2048, offset if normal == 1 else 0, offset if normal == 2 else 0
        )[0]
        for i in range(2049)
    )
    assert (
        sum(a < 0 < b for a, b in zip(derivatives[:-1], derivatives[1:], strict=True))
        <= 1
    )
    plane = rocheometry.AxisNormalPlane(axis, offset)
    outline = s.section(plane)
    assert grid_piece_count(g, s.potential, plane) == len(outline)
