"""Accuracy, topology and smoothness of the fixed angular section gauge."""

from math import cos, exp, hypot, isfinite, log, pi, sin, sqrt
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._roche.field import RocheField
from tests._helpers import gauge_neck_is_uncapped, single_path

INCLINATIONS = (0.0, pi / 6, pi / 3, pi / 2)


def check_section(qi: float, fill: float, inclination: float) -> float:
    surface = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill)
    path = single_path(surface.section(rocheometry.CentrelinePlane(inclination)))
    assert path.closed and len(path.segments) == 52
    assert path.segments[0].p0 == path.segments[-1].p3
    scale = surface.volume_radius()
    worst = 0.0
    for segment in path.segments:
        for index in range(12):
            x, y = segment.evaluate((index + 0.5) / 12)
            r = hypot(x, y)
            exact = surface.radius(
                (x / r, y / r * cos(inclination), -y / r * sin(inclination))
            )
            worst = max(worst, abs(r - exact) / scale)
    assert worst <= 3e-4, (qi, fill, inclination, worst)
    geometry = surface.component
    l1 = geometry.l1()
    base_angles = tuple(
        pi * (u - sin(2 * pi * u) / (2 * pi)) for u in (j / 24 for j in range(25))
    )
    nu = _neck_width(geometry.field, surface._depth, l1)
    a, b = sqrt(10.0 * nu * base_angles[1]), base_angles[1] / 2
    angles = (base_angles[0], a * (b / hypot(a, b)), *base_angles[1:])
    # Exact radial knots and symmetry are independent of the cubic error budget.
    for k, segment in enumerate(path.segments[1:26]):
        x, y = segment.p0
        if k < 2 and fill == 1.0:
            assert (x, y) == (surface.component.l1().position[0], 0.0)
        else:
            # Test the prescribed ray, rather than renormalizing the rounded
            # knot coordinates: tiny direction changes perturb near-L1 roots
            # at very small qi. Keep the tolerance purely relative.
            angle = angles[k]
            c, s = (1.0, 0.0) if k == 0 else (cos(angle), sin(angle))
            transverse = (cos(inclination), -sin(inclination))
            if inclination == pi / 2:
                transverse = (0.0, -1.0)
            geometry = surface.component
            radius = geometry._computers.radius(
                geometry.field,
                surface._depth,
                (c, s * transverse[0], s * transverse[1]),
            )
            assert hypot(x, y) == pytest.approx(radius, rel=1e-12, abs=0.0)
            assert (x, y) == (radius * c, radius * s)
        mirror = path.segments[-2 - k]
        for upper, lower in zip(
            (segment.p0, segment.p1, segment.p2, segment.p3),
            (mirror.p3, mirror.p2, mirror.p1, mirror.p0),
            strict=True,
        ):
            assert lower == (upper[0], -upper[1])
    if fill == 1.0:
        l1_point = (l1.position[0], 0.0)
        for segment in (
            path.segments[0],
            path.segments[1],
            path.segments[-2],
            path.segments[-1],
        ):
            assert segment.p0 == segment.p1 == segment.p2 == segment.p3 == l1_point
        assert all(segment.p0 != segment.p3 for segment in path.segments[2:-2])
    else:
        assert all(segment.p0 != segment.p3 for segment in path.segments[1:-1])
    back = path.segments[25].p3
    assert back[1] == 0.0
    assert -back[0] == pytest.approx(
        surface.radius((-1.0, 0.0, 0.0)), rel=1e-12, abs=0.0
    )
    return worst


@pytest.mark.parametrize(
    "qi, fill, inclination",
    [
        (1e-8, 0.999999, 0.0),
        (1.0, 1.0, pi / 2),
        (1e8, 0.3, pi / 3),
        (0.01, 0.9, pi / 6),
    ],
)
def test_fast_accuracy_subset(qi: float, fill: float, inclination: float) -> None:
    check_section(qi, fill, inclination)


@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_control_points_vary_smoothly_in_q(fill: float) -> None:
    points = []
    scales = []
    for index in range(201):
        qi = exp(log(0.5) + index / 200 * log(4.0))
        surface = rocheometry.ComponentGeometry(qi).surface(volume_radius_fill=fill)
        points.append(
            tuple(
                p
                for segment in single_path(surface.section()).segments
                for p in (segment.p0, segment.p1, segment.p2, segment.p3)
            )
        )
        scales.append(surface.volume_radius())
    worst = max(
        abs(a[d] - 2 * b[d] + c[d]) / scales[index]
        for index in range(1, 200)
        for a, b, c in zip(
            points[index - 1], points[index], points[index + 1], strict=True
        )
        for d in (0, 1)
    )
    assert worst <= 1e-5, worst


def test_routing_and_zero_y_section_accuracy() -> None:
    graph = Fitted()._computers()
    router = cast(
        PlaneRoutedSectionPathComputer,
        graph.section,
    )
    assert isinstance(router.centreline, GaugeFixedSectionPathComputer)
    assert isinstance(router.section, TracedSectionPathComputer)
    surface = rocheometry.ComponentGeometry(0.3).surface(volume_radius_fill=0.8)
    field = surface.component.field
    for inclination in INCLINATIONS:
        plane = rocheometry.CentrelinePlane(inclination)
        assert single_path(surface.section(plane)) == single_path(
            router.centreline(field, surface._depth, plane)
        )
    section = rocheometry.AxisNormalPlane("z", 0.1)
    assert single_path(surface.section(section)) == single_path(
        router.section(field, surface._depth, section)
    )
    plane = rocheometry.AxisNormalPlane("y", 0.0)
    path = single_path(surface.section(plane))
    assert path == single_path(router.centreline(field, surface._depth, plane))
    traced = single_path(router.section(field, surface._depth, plane))
    from crosschecks.distance.bezier import _PathDistance

    distance = _PathDistance(traced)
    assert (
        max(distance(point) for point in path.sample(577)) / surface.volume_radius()
        <= 3e-4
    )


@pytest.mark.parametrize("n", [3, 8, 26])
def test_custom_topology(n: int) -> None:
    graph = Fitted()._computers()
    computer = GaugeFixedSectionPathComputer(
        radius_computer=graph.radius, l1_computer=graph.l1, segments_per_half=n
    )
    field = RocheField(1.0)
    assert (
        len(single_path(computer(field, 0.0, rocheometry.ORBITAL_PLANE)).segments)
        == 2 * n
    )


@pytest.mark.parametrize(
    "n, error",
    [
        (1, ValueError),
        (2, ValueError),
        (0, ValueError),
        (-2, ValueError),
        (True, TypeError),
        (2.0, TypeError),
        (3.0, TypeError),
    ],
)
def test_invalid_topology(n: object, error: type[Exception]) -> None:
    graph = Fitted()._computers()
    with pytest.raises(error, match="segments_per_half"):
        GaugeFixedSectionPathComputer(
            radius_computer=graph.radius,
            l1_computer=graph.l1,
            segments_per_half=cast(int, n),
        )


@pytest.mark.parametrize(
    "plane",
    [
        rocheometry.AxisNormalPlane("z", 0.1),
        rocheometry.AxisNormalPlane("y", 0.1),
        rocheometry.AxisNormalPlane("x", 0.0),
    ],
)
def test_rejects_parallel_slices(plane: rocheometry.SectionPlane) -> None:
    graph = Fitted()._computers()
    computer = GaugeFixedSectionPathComputer(
        radius_computer=graph.radius, l1_computer=graph.l1
    )
    field = RocheField(1.0)
    with pytest.raises(ValueError, match="centreline"):
        single_path(computer(field, 0.0, plane))


@pytest.mark.parametrize("qi", [1e-4, 0.1, 1.0, 10.0, 1e4])
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_control_points_approach_contact_continuously(
    qi: float, inclination: float
) -> None:
    """Check the quarter-power rate as detached controls approach the cone."""
    geometry = rocheometry.ComponentGeometry(qi)
    lobe = geometry.lobe()
    plane = rocheometry.CentrelinePlane(inclination)
    critical = single_path(lobe.section(plane))
    changes = []
    uncapped = []
    # The finest pair probes the rate below the neck angle's smooth upper cap.
    for gap in (1e-8, 1e-10, 1e-12):
        surface = geometry.surface(volume_radius_fill=1 - gap)
        uncapped.append(gauge_neck_is_uncapped(surface, plane))
        detached = single_path(surface.section(plane))
        assert len(detached.segments) == len(critical.segments) == 52
        changes.append(
            max(
                hypot(p[0] - q[0], p[1] - q[1])
                for before, after in zip(
                    detached.segments, critical.segments, strict=True
                )
                for p, q in zip(
                    (before.p0, before.p1, before.p2, before.p3),
                    (after.p0, after.p1, after.p2, after.p3),
                    strict=True,
                )
            )
            / lobe.named_radius("pole")
        )
    assert changes[0] > changes[1] > changes[2], changes
    assert changes[2] <= 1e-3, changes
    if all(uncapped[1:]):
        assert changes[2] <= 0.4 * changes[1], changes


@pytest.mark.parametrize("qi", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_critical_arc_length_sampling_skips_degenerate_segments(
    qi: float, inclination: float
) -> None:
    path = single_path(
        rocheometry.ComponentGeometry(qi)
        .lobe()
        .section(rocheometry.CentrelinePlane(inclination))
    )
    points = path.sample(201, spacing="arc_length")
    assert len(points) == 201
    assert points[0] == points[-1] == path.segments[0].p0
    assert all(isfinite(coordinate) for point in points for coordinate in point)
    # A closed path deliberately repeats its starting point at the end.
    assert len(set(points[:-1])) == len(points) - 1
    assert all(a != b for a, b in zip(points, points[1:]))
    assert all(
        isfinite(coordinate)
        for point in path.sample(201, spacing="parameter")
        for coordinate in point
    )
