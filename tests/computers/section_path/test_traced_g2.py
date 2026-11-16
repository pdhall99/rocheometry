"""Curvature matching on reference centreline sections and offset sections."""

from math import cos, hypot, pi, sin

import pytest

import rocheometry
from rocheometry import Reference
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._geometry import Point2D
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired, single_path


@pytest.mark.parametrize(
    "qi, inclination, expected",
    [
        (0.1, 0.0, 1.044681),
        (0.1, 0.6, 0.998564),
        (0.1, pi / 2, 0.906541),
        (10.0, 0.0, -1.044681),
        (1.0, 0.0, 0.0),
        (1.0, 0.6, 0.0),
        (1.0, pi / 2, 0.0),
    ],
)
def test_critical_branch_curvature_limit(
    qi: float, inclination: float, expected: float
) -> None:
    graph = Reference()._computers()
    field = RocheField(qi)
    l1 = graph.l1(field)
    curvature = field.critical_branch_curvature(l1.position[0], inclination)
    assert curvature == pytest.approx(expected, abs=1e-6)
    errors = []
    for angle in (2e-3, 1e-3):
        radius = graph.radius(
            field,
            0.0,
            (cos(angle), sin(angle) * cos(inclination), -sin(angle) * sin(inclination)),
        )
        x, u = radius * cos(angle), radius * sin(angle)
        gradient = field.gradient(x, u * cos(inclination), -u * sin(inclination))
        gx, gu = (
            gradient[0],
            gradient[1] * cos(inclination) - gradient[2] * sin(inclination),
        )
        a, b, c = field.centreline_plane_hessian(x, u, inclination)
        near = -(a * gu * gu - 2 * b * gx * gu + c * gx * gx) / hypot(gx, gu) ** 3
        errors.append(abs(near - curvature))
    assert errors[1] < 0.02
    assert errors[1] / errors[0] == pytest.approx(0.5, abs=0.02)


def _cross(a: Point2D, b: Point2D) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _curvatures(segment: rocheometry.CubicBezier) -> tuple[float, float]:
    a = (segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1])
    b = (segment.p2[0] - segment.p1[0], segment.p2[1] - segment.p1[1])
    c = (segment.p3[0] - segment.p2[0], segment.p3[1] - segment.p2[1])
    return (
        2 / 3 * _cross(a, b) / hypot(*a) ** 3,
        2 / 3 * _cross(b, c) / hypot(*c) ** 3,
    )


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("fill", [1.0, 0.9])
@pytest.mark.parametrize(
    "plane",
    [
        rocheometry.CentrelinePlane(0),
        rocheometry.CentrelinePlane(0.6),
        rocheometry.CentrelinePlane(pi / 2),
        rocheometry.AxisNormalPlane("x", 0.1),
        rocheometry.AxisNormalPlane("z", 0.05),
    ],
)
def test_reference_grid_curvature_continuity_and_segment_budget(
    qi: float, fill: float, plane: rocheometry.SectionPlane
) -> None:
    surface = rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=fill)
    path = single_path(surface.section(plane))
    assert len(path.segments) <= 400
    curvatures = [_curvatures(segment) for segment in path.segments]
    for i, (_, end) in enumerate(curvatures):
        if (
            i == len(curvatures) - 1
            and fill == 1.0
            and isinstance(plane, rocheometry.CentrelinePlane)
        ):
            continue  # The upper and lower branches meet at the L1 corner.
        start = curvatures[(i + 1) % len(curvatures)][0]
        assert abs(end - start) <= 1e-8 * max(
            abs(end), abs(start), 1 / surface.volume_radius()
        )


@pytest.mark.parametrize("tolerance", [0.1, 1e-15])
def test_inadmissible_arms_split_until_hermite_fallback_at_depth_limit(
    monkeypatch: pytest.MonkeyPatch, tolerance: float
) -> None:
    import rocheometry._computers.section_path.traced as implementation

    intervals: list[float] = []

    def inadmissible(
        left: implementation._Knot,
        right: implementation._Knot,
        t0: Point2D,
        t1: Point2D,
        h0: float,
    ) -> None:
        intervals.append(right.angle - left.angle)
        return None

    monkeypatch.setattr(implementation, "_g2_arms", inadmissible)
    monkeypatch.setattr(implementation, "SECTION_MAX_REFINEMENT_DEPTH", 2)
    graph = Reference()._computers()
    tracer = TracedSectionPathComputer(
        radius_computer=graph.radius,
        l1_computer=graph.l1,
        tolerance=tolerance,
    )
    if tolerance == 1e-15:
        with pytest.raises(RuntimeError, match="depth limit"):
            single_path(
                tracer(
                    RocheField(0.1),
                    0.0,
                    rocheometry.ORBITAL_PLANE,
                )
            )
    else:
        path = single_path(
            tracer(
                RocheField(0.1),
                0.0,
                rocheometry.ORBITAL_PLANE,
            )
        )
        assert len(path.segments) == 32
        assert intervals.count(pi / 4) == 4
        assert intervals.count(pi / 8) == 8
        assert len(intervals) == 28
