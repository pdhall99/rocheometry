"""Nearest-point distances to cubic paths without polyline flattening."""

from dataclasses import dataclass
from math import fsum, hypot
from sys import float_info

from rocheometry._bezier import BezierPath, CubicBezier
from rocheometry._geometry import Bounds2D, Point2D
from rocheometry._numerics.polynomial import power_value
from rocheometry._numerics.roots import brent_root, same_sign


@dataclass(frozen=True, slots=True)
class _SegmentDistance:
    segment: CubicBezier
    bounds: Bounds2D
    coefficients: tuple[Point2D, Point2D, Point2D]

    @classmethod
    def from_segment(cls, segment: CubicBezier) -> "_SegmentDistance":
        points = (segment.p0, segment.p1, segment.p2, segment.p3)
        differences = tuple(
            (b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:], strict=False)
        )
        d0, d1, d2 = differences
        return cls(
            segment,
            (
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points),
            ),
            (
                (3 * d0[0], 3 * d0[1]),
                (3 * (d1[0] - d0[0]), 3 * (d1[1] - d0[1])),
                (d2[0] - 2 * d1[0] + d0[0], d2[1] - 2 * d1[1] + d0[1]),
            ),
        )

    def lower_bound(self, point: Point2D) -> float:
        xmin, ymin, xmax, ymax = self.bounds
        return hypot(
            max(xmin - point[0], 0.0, point[0] - xmax),
            max(ymin - point[1], 0.0, point[1] - ymax),
        )

    def distance(self, point: Point2D) -> float:
        # A closest point is an endpoint or a zero of (P(t)-point) dot P'(t),
        # a polynomial of degree at most five. Scaling keeps its coefficient
        # arithmetic independent of the curve's units.
        relative = (
            (self.segment.p0[0] - point[0], self.segment.p0[1] - point[1]),
            *self.coefficients,
        )
        scale = max(abs(value) for coefficient in relative for value in coefficient)
        if scale == 0.0:
            return 0.0
        vectors = tuple((x / scale, y / scale) for x, y in relative)
        coefficients = tuple(
            fsum(
                j * (vectors[i][0] * vectors[j][0] + vectors[i][1] * vectors[j][1])
                for i in range(4)
                for j in range(1, 4)
                if i + j - 1 == degree
            )
            for degree in range(6)
        )
        parameters = (0.0, 1.0, *_unit_interval_roots(coefficients))
        return min(
            hypot(p[0] - point[0], p[1] - point[1])
            for p in (self.segment.evaluate(t) for t in parameters)
        )


class _PathDistance:
    def __init__(self, path: BezierPath) -> None:
        self.segments = tuple(
            _SegmentDistance.from_segment(segment) for segment in path.segments
        )

    def __call__(self, point: Point2D) -> float:
        candidates = sorted(
            (segment.lower_bound(point), index)
            for index, segment in enumerate(self.segments)
        )
        best = float("inf")
        for lower_bound, index in candidates:
            if lower_bound >= best:
                break
            best = min(best, self.segments[index].distance(point))
        return best


def _unit_interval_roots(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    while len(coefficients) > 1 and coefficients[-1] == 0.0:
        coefficients = coefficients[:-1]
    scale = max(abs(value) for value in coefficients)
    if scale == 0.0 or len(coefficients) == 1:
        return ()
    coefficients = tuple(value / scale for value in coefficients)
    if len(coefficients) == 2:
        root = -coefficients[0] / coefficients[1]
        return (root,) if 0.0 <= root <= 1.0 else ()
    derivative = tuple(
        index * value for index, value in enumerate(coefficients) if index
    )
    critical = _unit_interval_roots(derivative)
    boundaries = (0.0, *(root for root in critical if 0.0 < root < 1.0), 1.0)
    values = tuple(power_value(t, coefficients) for t in boundaries)
    tolerance = 32 * float_info.epsilon * fsum(abs(value) for value in coefficients)
    roots = [
        t
        for t, value in zip(boundaries, values, strict=True)
        if abs(value) <= tolerance
    ]
    for left, right, f_left, f_right in zip(
        boundaries, boundaries[1:], values, values[1:], strict=False
    ):
        if f_left != 0.0 and f_right != 0.0 and not same_sign(f_left, f_right):
            roots.append(
                brent_root(
                    lambda t: power_value(t, coefficients),
                    left,
                    right,
                    absolute_tolerance=8 * float_info.epsilon,
                    relative_tolerance=0.0,
                )
            )
    return tuple(sorted(set(roots)))
