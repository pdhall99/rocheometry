"""The projected-shoulder gauge keeps fixed topology and analytic knot tangents."""

from math import atan2, cos, hypot, pi, sin, ulp

import pytest

import rocheometry
from rocheometry._computers.section_path.gauge_fixed import _gauge_angles
from rocheometry._computers.silhouette.paths import GaugeFixedSilhouettePathComputer
from tests._helpers import select
from tests.computers.silhouette.test_projected_outlines import projected


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e6, 1e7, 1e8))
@pytest.mark.parametrize("inclination", (0.0, pi / 6, pi / 3, pi / 2))
@pytest.mark.parametrize("start,nu", ((0.0, 0.0), (0.0, 0.1), (0.02, 0.001)))
def test_shoulder_schedule_is_increasing_with_the_same_endpoint_slots(
    qi: float, inclination: float, start: float, nu: float
) -> None:
    assert (
        GaugeFixedSilhouettePathComputer.__dataclass_fields__[
            "segments_per_half"
        ].default
        == 44
    )
    angles = _gauge_angles(44, start, nu)
    assert len(angles) == 44
    assert angles[0] == start
    assert angles[-1] == pi
    if nu == 0:
        assert angles[0] == angles[1]
    else:
        assert angles[0] < angles[1]
    assert all(a < b for a, b in zip(angles[1:], angles[2:]))


@pytest.mark.parametrize("qi", (1e6, 1e7, 1e8))
@pytest.mark.parametrize("kind,level", (("fill", 1.0), ("F", 1e-6), ("F", 0.99)))
def test_heavy_silhouette_retains_exact_knot_tangents(
    qi: float, kind: str, level: float
) -> None:
    surface = select(rocheometry.ComponentGeometry(qi), kind, level)
    plane = rocheometry.CentrelinePlane(pi / 3)
    path = surface.silhouette_envelope(plane).paths[0]
    f = projected(surface, plane.inclination)
    assert len(path.segments) == 88
    for segment in path.segments[:44]:
        if segment.p0 == segment.p3:
            continue
        for point, arm in (
            (
                segment.p0,
                (segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1]),
            ),
            (
                segment.p3,
                (segment.p3[0] - segment.p2[0], segment.p3[1] - segment.p2[1]),
            ),
        ):
            angle = atan2(point[1], point[0])
            tangent = (
                f.cone_tangent()
                if point == f.l1.position[:2] and f.critical
                else f.tangent(point, cos(angle), sin(angle))
            )
            # Account for the independent LOS solve and subtraction of a
            # very short arm from coordinates near one.
            assert abs(arm[0] * tangent[1] - arm[1] * tangent[0]) <= max(
                2e-12 * hypot(*arm), 4 * ulp(hypot(*point))
            )
            assert arm[0] * tangent[0] + arm[1] * tangent[1] > 0


@pytest.mark.parametrize("qi", (1e5, 1e6, 1e7, 1e8))
@pytest.mark.parametrize("kind,level", (("fill", 1.0), ("F", 0.01), ("F", 0.99)))
def test_heavy_gauge_control_points_are_smooth_in_q(
    qi: float, kind: str, level: float
) -> None:
    paths = []
    for ratio in (qi * (1 - 1e-6), qi, qi * (1 + 1e-6)):
        surface = select(rocheometry.ComponentGeometry(ratio), kind, level)
        path = surface.silhouette_envelope(rocheometry.CentrelinePlane(pi / 3)).paths[0]
        assert len(path.segments) == 88
        paths.append([p for s in path.segments for p in (s.p0, s.p1, s.p2, s.p3)])
    pole = select(rocheometry.ComponentGeometry(qi), kind, level).named_radius("pole")
    assert (
        max(
            hypot(a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1])
            for a, b, c in zip(*paths, strict=True)
        )
        <= 1e-8 * pole
    )
