"""Cubic nearest points include every stationary candidate and endpoint."""

from math import hypot, sqrt
from random import Random

import pytest

from crosschecks.distance.bezier import _PathDistance, _SegmentDistance
from rocheometry import AffineTransform2D, BezierPath, CubicBezier
from rocheometry._numerics.roots import brent_root


def test_line_distance_clamps_at_endpoints() -> None:
    curve = CubicBezier((0.0, 0.0), (1 / 3, 0.0), (2 / 3, 0.0), (1.0, 0.0))
    distance = _PathDistance(BezierPath((curve,), closed=False))
    assert distance((0.4, 2.0)) == pytest.approx(2.0)
    assert distance((-1.0, 1.0)) == pytest.approx(sqrt(2))
    assert distance((2.0, 1.0)) == pytest.approx(sqrt(2))


@pytest.mark.parametrize("scale", [1e-15, 1.0, 1e15])
def test_curved_distance_is_scale_invariant(scale: float) -> None:
    # P(t) = (t, t^2); the closest parameter to (0, 1) is sqrt(1/2).
    parabola = CubicBezier(
        (0.0, 0.0), (1 / 3, 0.0), (2 / 3, 1 / 3), (1.0, 1.0)
    ).transformed(AffineTransform2D.scale(scale))
    distance = _PathDistance(BezierPath((parabola,), closed=False))
    assert distance((0.0, scale)) / scale == pytest.approx(sqrt(0.75), rel=2e-14)
    assert distance((0.5 * scale, 0.25 * scale)) / scale <= 2e-14


def test_degenerate_segment_and_stationary_reversal() -> None:
    constant = CubicBezier((0.5, 0.0), (0.5, 0.0), (0.5, 0.0), (0.5, 0.0))
    assert _SegmentDistance.from_segment(constant).distance((0.5, 1.0)) == 1.0
    reversal = CubicBezier((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0))
    assert _SegmentDistance.from_segment(reversal).distance(
        (0.75, 1.0)
    ) == pytest.approx(1.0, abs=2e-14)
    assert _SegmentDistance.from_segment(reversal).distance((0.5, 0.0)) <= 2e-14


def test_path_pruning_does_not_drop_an_interior_nearest_point() -> None:
    segments = (
        CubicBezier((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        CubicBezier((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (3.0, 1.0)),
    )
    distance = _PathDistance(BezierPath(segments, closed=False))
    for segment in segments:
        assert distance(segment.evaluate(0.373)) <= 3e-14


def test_multiple_stationary_points_match_an_independent_bernstein_sweep() -> None:
    random = Random(4107)
    for _ in range(32):
        points = tuple((random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(4))
        curve = CubicBezier(*points)
        point = (random.uniform(-2, 2), random.uniform(-2, 2))

        def derivative(t: float) -> float:
            u = 1 - t
            tangent = tuple(
                3 * (u * u * (b - a) + 2 * u * t * (c - b) + t * t * (d - c))
                for a, b, c, d in zip(*points, strict=True)
            )
            p = curve.evaluate(t)
            return (p[0] - point[0]) * tangent[0] + (p[1] - point[1]) * tangent[1]

        candidates = [0.0, 1.0]
        left, f_left = 0.0, derivative(0.0)
        for index in range(1, 1001):
            right = index / 1000
            f_right = derivative(right)
            if f_left * f_right < 0.0:
                candidates.append(brent_root(derivative, left, right))
            left, f_left = right, f_right
        expected = min(
            hypot(p[0] - point[0], p[1] - point[1])
            for p in (curve.evaluate(t) for t in candidates)
        )
        actual = _SegmentDistance.from_segment(curve).distance(point)
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-13)


def test_contract_outline_distance_includes_reverse_and_exact_corners() -> None:
    from crosschecks.distance.curves import two_way_distance
    from rocheometry import CompoundBezierPath

    def line(end: float) -> CompoundBezierPath:
        segment = CubicBezier(
            (0.0, 0.0), (end / 3, 0.0), (2 * end / 3, 0.0), (end, 0.0)
        )
        return CompoundBezierPath((BezierPath((segment,), closed=False),))

    result = two_way_distance(line(0.1), line(1.0), count=51, corners=[(1.0, 0.0)])
    assert result["forward"] <= 1e-15
    assert result["reverse"] == pytest.approx(0.9, abs=1e-15)
    assert result["corner"] == pytest.approx(0.9, abs=1e-15)
    assert result["value"] == max(
        result["forward"], result["reverse"], result["corner"]
    )
