"""Containment of contact nodes in the two open envelope arcs together."""

from math import cos, dist, pi, sin

from crosschecks.distance.bezier import _PathDistance

import rocheometry
from rocheometry._bezier import _axis_extrema
from rocheometry._geometry import Point2D
from rocheometry._numerics.roots import brent_root


def _bridge(a: Point2D, b: Point2D) -> rocheometry.CubicBezier:
    return rocheometry.CubicBezier(
        a,
        ((2 * a[0] + b[0]) / 3, (2 * a[1] + b[1]) / 3),
        ((a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3),
        b,
    )


class ContactProjection:
    """Use cubic row intersections for winding, and exact nearest distances.

    A single envelope arc cannot enclose points. Join the two components' arcs
    rather than inventing a silhouette of the watershed cut. Round-off endpoint gaps are bridged only for this validation's winding test.
    """

    def __init__(
        self, first: rocheometry.BezierPath, second: rocheometry.BezierPath
    ) -> None:
        if dist(first.segments[-1].p3, second.segments[0].p0) > dist(
            first.segments[-1].p3, second.segments[-1].p3
        ):
            second = second.reversed()
        segments = (
            *first.segments,
            _bridge(first.segments[-1].p3, second.segments[0].p0),
            *second.segments,
            _bridge(second.segments[-1].p3, first.segments[0].p0),
        )
        self.distances = (_PathDistance(first), _PathDistance(second))
        self.pieces = []
        for segment in segments:
            ends = (
                0.0,
                *_axis_extrema(
                    *(p[1] for p in (segment.p0, segment.p1, segment.p2, segment.p3))
                ),
                1.0,
            )
            for a, b in zip(ends[:-1], ends[1:], strict=True):
                ya, yb = segment.evaluate(a)[1], segment.evaluate(b)[1]
                if ya != yb:
                    self.pieces.append((min(ya, yb), max(ya, yb), segment, a, b))
        self.cache: dict[Point2D, float] = {}

    def excess(self, point: Point2D) -> float:
        point = (point[0], abs(point[1]))
        if point not in self.cache:
            x, y = point
            hits = []
            for lo, hi, segment, a, b in self.pieces:
                if lo <= y < hi:
                    t = brent_root(
                        lambda t: segment.evaluate(t)[1] - y,
                        a,
                        b,
                        absolute_tolerance=1e-15,
                        relative_tolerance=1e-15,
                    )
                    hits.append(segment.evaluate(t)[0])
            inside = sum(hit > x for hit in hits) % 2
            self.cache[point] = 0.0 if inside else min(d(point) for d in self.distances)
        return self.cache[point]


def crossing_at_depth(
    geometry: rocheometry.ComponentGeometry,
    depth: float,
    plane: rocheometry.CentrelinePlane,
) -> Point2D:
    """Locate the LOS-stationary rim at the stored surface depth."""
    field = geometry.field
    sine, cosine = sin(plane.inclination), cos(plane.inclination)

    def point(psi: float) -> tuple[float, float, float]:
        return geometry._computers.watershed.rim_at_depth(field, depth, psi)

    def residual(psi: float) -> float:
        _, gy, gz = field.gradient(*point(psi))
        return gy * sine + gz * cosine

    psi = (
        -pi / 2
        if plane.inclination == pi / 2
        else brent_root(
            residual,
            -pi / 2,
            0.0,
            absolute_tolerance=0.0,
            relative_tolerance=1e-15,
        )
    )
    return plane.project(point(psi))
