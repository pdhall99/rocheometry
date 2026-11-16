"""Renderer-independent cubic Bézier path primitives."""

from __future__ import annotations

import reprlib
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from math import copysign, fsum, hypot, isfinite, sqrt
from sys import float_info
from typing import Literal, TypeAlias, get_args

from rocheometry._geometry import AffineTransform2D, Bounds2D, Point2D
from rocheometry._numerics.quadrature import gauss_legendre
from rocheometry._numerics.roots import brent_root
from rocheometry._validation import (
    normalize_point2d,
    require_bool,
    require_choice,
    require_instance,
    require_int,
)

SampleSpacing: TypeAlias = Literal["parameter", "arc_length"]
"""The sampling coordinate along a piecewise Bézier path."""


@dataclass(frozen=True, slots=True)
class CubicBezier:
    """An immutable cubic Bézier segment with finite planar control points.

    All points must be two-coordinate tuples; their real numeric coordinates
    are converted to floats. All four points use the same caller-chosen units.

    Args:
        p0: Start point.
        p1: Control point defining the starting tangent.
        p2: Control point defining the ending tangent.
        p3: End point.

    Attributes:
        p0: Normalized start point.
        p1: Normalized first control point.
        p2: Normalized second control point.
        p3: Normalized end point.
    """

    p0: Point2D
    p1: Point2D
    p2: Point2D
    p3: Point2D

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2", "p3"):
            object.__setattr__(
                self,
                name,
                normalize_point2d(getattr(self, name), name=name),
            )

    def evaluate(self, t: float) -> Point2D:
        """Evaluate the segment at a parameter in `[0, 1]`.

        Args:
            t: Bézier parameter in `[0, 1]`, including both endpoints.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t: expected a value in [0, 1], got {reprlib.repr(t)}")
        u = 1.0 - t
        b0 = u * u * u
        b1 = 3.0 * u * u * t
        b2 = 3.0 * u * t * t
        b3 = t * t * t
        return (
            b0 * self.p0[0] + b1 * self.p1[0] + b2 * self.p2[0] + b3 * self.p3[0],
            b0 * self.p0[1] + b1 * self.p1[1] + b2 * self.p2[1] + b3 * self.p3[1],
        )

    def transformed(self, transform: AffineTransform2D) -> CubicBezier:
        """Return this segment transformed by one affine map."""
        return CubicBezier(
            p0=transform.apply(self.p0),
            p1=transform.apply(self.p1),
            p2=transform.apply(self.p2),
            p3=transform.apply(self.p3),
        )

    def reversed(self) -> CubicBezier:
        """Return the same geometric segment with reversed parametrization."""
        return CubicBezier(
            p0=self.p3,
            p1=self.p2,
            p2=self.p1,
            p3=self.p0,
        )

    def split(
        self,
        t: float = 0.5,
    ) -> tuple[CubicBezier, CubicBezier]:
        """Split this segment at one parameter value using de Casteljau.

        Args:
            t: Split parameter in `[0, 1]`. Endpoint splits produce one degenerate
                segment and one segment with the original geometry.

        Returns:
            The `(before, after)` segments, ordered along the original curve.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t: expected a value in [0, 1], got {reprlib.repr(t)}")

        p01 = _interpolate(self.p0, self.p1, t)
        p12 = _interpolate(self.p1, self.p2, t)
        p23 = _interpolate(self.p2, self.p3, t)
        p012 = _interpolate(p01, p12, t)
        p123 = _interpolate(p12, p23, t)
        split_point = _interpolate(p012, p123, t)
        return (
            CubicBezier(self.p0, p01, p012, split_point),
            CubicBezier(split_point, p123, p23, self.p3),
        )


@dataclass(frozen=True, slots=True)
class BezierPath:
    """An immutable, connected sequence of planar cubic Bézier segments.

    Adjacent endpoints must coincide exactly. A closed path must also have
    identical first and last endpoints; this flag does not insert a segment.

    Attributes:
        segments: Nonempty tuple of connected cubic segments in traversal order.
        closed: Whether the final endpoint coincides with the first endpoint.
    """

    segments: tuple[CubicBezier, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        require_instance(self.segments, tuple, name="segments")
        if not self.segments:
            raise ValueError(
                "segments: expected at least one cubic Bézier segment, got ()"
            )
        for index, segment in enumerate(self.segments):
            require_instance(
                segment,
                CubicBezier,
                name=f"segments[{index}]",
            )
        object.__setattr__(self, "closed", require_bool(self.closed, name="closed"))
        for left, right in zip(self.segments, self.segments[1:], strict=False):
            if left.p3 != right.p0:
                raise ValueError("adjacent Bézier segments must join exactly")
        if self.closed and self.segments[-1].p3 != self.segments[0].p0:
            raise ValueError("closed Bézier paths must close exactly")

    def __repr__(self) -> str:
        """Summarize the path: its segment count, whether it closes, its bounds."""
        shape = "closed" if self.closed else "open"
        return (
            f"<BezierPath: {len(self.segments)} cubic segments, {shape}, "
            f"bounds={_short_bounds(self.bounds)}>"
        )

    @property
    def bounds(self) -> Bounds2D:
        """The axis-aligned `(xmin, ymin, xmax, ymax)` bounds in the path's units.

        Bounds include interior cubic extrema rather than only control points or
        sampled points.
        """
        xs: list[float] = []
        ys: list[float] = []
        for segment in self.segments:
            x_parameters = {0.0, 1.0}
            x_parameters.update(
                _axis_extrema(
                    segment.p0[0],
                    segment.p1[0],
                    segment.p2[0],
                    segment.p3[0],
                )
            )
            for t in x_parameters:
                xs.append(segment.evaluate(t)[0])

            y_parameters = {0.0, 1.0}
            y_parameters.update(
                _axis_extrema(
                    segment.p0[1],
                    segment.p1[1],
                    segment.p2[1],
                    segment.p3[1],
                )
            )
            for t in y_parameters:
                ys.append(segment.evaluate(t)[1])
        return (min(xs), min(ys), max(xs), max(ys))

    def sample(
        self, n: int = 200, *, spacing: SampleSpacing = "parameter"
    ) -> tuple[Point2D, ...]:
        """Sample `n` points over the ordered piecewise path.

        Sampling may be uniform in piecewise Bézier parameter or numerical arc
        length. For a closed path the first and final points are identical.

        Args:
            n: Total number of samples including both path endpoints. Must be
                an integer of at least `2`; booleans are rejected.
            spacing: `parameter` preserves equal samples per segment; `arc_length`
                uses equal distances along the curve, including across joins.
        """
        n = require_int(n, name="n")
        if n < 2:
            raise ValueError(
                "n: expected an integer greater than or equal to 2, "
                f"got {reprlib.repr(n)}"
            )

        require_choice(
            require_instance(spacing, str, name="spacing"),
            get_args(SampleSpacing),
            name="spacing",
        )
        if spacing == "arc_length":
            return _arc_length_samples(self.segments, n)

        segment_count = len(self.segments)
        samples: list[Point2D] = []
        for index in range(n):
            if index == n - 1:
                segment_index = segment_count - 1
                t = 1.0
            else:
                position = segment_count * index / (n - 1)
                segment_index = min(int(position), segment_count - 1)
                t = position - segment_index
            samples.append(self.segments[segment_index].evaluate(t))
        return tuple(samples)

    def transformed(self, transform: AffineTransform2D) -> BezierPath:
        """Return this path transformed by one affine map."""
        return BezierPath(
            segments=tuple(segment.transformed(transform) for segment in self.segments),
            closed=self.closed,
        )

    def reversed(self) -> BezierPath:
        """Return the same geometric path with reversed parametrization."""
        return BezierPath(
            segments=tuple(segment.reversed() for segment in reversed(self.segments)),
            closed=self.closed,
        )


@dataclass(frozen=True, slots=True)
class CompoundBezierPath:
    """Zero or more paths, transformed, bounded and exported together.

    Paths need not connect, and each retains its own closed flag. Sampling
    belongs to the individual paths because there is no traversal across gaps.

    Attributes:
        paths: Tuple of Bézier paths in traversal order; may be empty.
    """

    paths: tuple[BezierPath, ...] = ()

    def __post_init__(self) -> None:
        require_instance(self.paths, tuple, name="paths")
        for index, path in enumerate(self.paths):
            require_instance(path, BezierPath, name=f"paths[{index}]")

    def __len__(self) -> int:
        """Return the number of paths."""
        return len(self.paths)

    def __iter__(self) -> Iterator[BezierPath]:
        """Iterate over the paths in traversal order."""
        return iter(self.paths)

    def __repr__(self) -> str:
        """Summarize the compound: its paths, segments and bounds."""
        if not self.paths:
            return "<CompoundBezierPath: empty>"
        segments = sum(len(path.segments) for path in self.paths)
        noun = "path" if len(self.paths) == 1 else "paths"
        return (
            f"<CompoundBezierPath: {len(self.paths)} {noun}, {segments} cubic segments, "
            f"bounds={_short_bounds(self.bounds)}>"
        )

    @property
    def bounds(self) -> Bounds2D:
        """Union of the paths' exact bounds; undefined for an empty compound."""
        if not self.paths:
            raise ValueError("an empty compound path has no bounds")
        bounds = tuple(path.bounds for path in self.paths)
        return (
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        )

    def transformed(self, transform: AffineTransform2D) -> CompoundBezierPath:
        """Return every path transformed by the same affine map."""
        return CompoundBezierPath(tuple(path.transformed(transform) for path in self))

    def reversed(self) -> CompoundBezierPath:
        """Reverse the path order and each path's direction."""
        return CompoundBezierPath(
            tuple(path.reversed() for path in reversed(self.paths))
        )


def _axis_extrema(
    p0: float,
    p1: float,
    p2: float,
    p3: float,
) -> tuple[float, ...]:
    # Normalize first differences before forming the derivative coefficients.
    # This preserves translations and handles both tiny and large curves.
    d0, d1, d2 = p1 - p0, p2 - p1, p3 - p2
    scale = max(abs(d0), abs(d1), abs(d2))
    if scale == 0.0:
        return ()
    if isfinite(scale):
        d0, d1, d2 = d0 / scale, d1 / scale, d2 / scale
    else:
        # Opposite finite coordinates can have an overflowing difference.
        scale = max(abs(p0), abs(p1), abs(p2), abs(p3))
        p0, p1, p2, p3 = p0 / scale, p1 / scale, p2 / scale, p3 / scale
        d0, d1, d2 = p1 - p0, p2 - p1, p3 - p2

    # The derivative divided by a common factor is A*t^2 + B*t + C.
    a, b, c = d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0
    scale = max(abs(a), abs(b), abs(c))
    a, b, c = a / scale, b / scale, c / scale
    epsilon = 32.0 * float_info.epsilon

    roots: list[float] = []
    if abs(a) <= epsilon:
        if abs(b) > epsilon:
            root = -c / b
            if 0.0 < root < 1.0:
                roots.append(root)
        return tuple(roots)

    discriminant = b * b - 4.0 * a * c
    if discriminant < -epsilon:
        return ()
    discriminant = max(0.0, discriminant)
    root_discriminant = sqrt(discriminant)
    # Compute one root without subtracting nearly equal values, then use the
    # product of the roots for the other.
    q = -0.5 * (b + copysign(root_discriminant, b))
    candidates = (q / a, c / q) if q != 0.0 else (-b / (2.0 * a),)
    for root in candidates:
        if 0.0 < root < 1.0:
            roots.append(root)
    return tuple(roots)


def _interpolate(start: Point2D, end: Point2D, t: float) -> Point2D:
    return (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
    )


def _arc_length_samples(
    segments: tuple[CubicBezier, ...], n: int
) -> tuple[Point2D, ...]:
    nodes, weights = gauss_legendre(16)

    def length(segment: CubicBezier, left: float, right: float) -> float:
        half = 0.5 * (right - left)
        middle = 0.5 * (right + left)

        def speed(t: float) -> float:
            u = 1.0 - t
            derivative = tuple(
                3.0 * (u * u * (b - a) + 2.0 * u * t * (c - b) + t * t * (d - c))
                for a, b, c, d in zip(
                    segment.p0, segment.p1, segment.p2, segment.p3, strict=True
                )
            )
            return hypot(*derivative)

        return half * fsum(
            w * speed(middle + half * x) for x, w in zip(nodes, weights, strict=True)
        )

    intervals: list[tuple[CubicBezier, float, float, float]] = []

    def partition(
        segment: CubicBezier,
        left: float,
        right: float,
        whole: float,
        tolerance: float,
        depth: int,
    ) -> None:
        middle = 0.5 * (left + right)
        before, after = length(segment, left, middle), length(segment, middle, right)
        refined = before + after
        if abs(refined - whole) <= tolerance:
            if before > 0.0:
                intervals.append((segment, left, middle, before))
            if after > 0.0:
                intervals.append((segment, middle, right, after))
            return
        if depth == 24:
            raise RuntimeError("arc-length integration did not converge")
        partition(segment, left, middle, before, 0.5 * tolerance, depth + 1)
        partition(segment, middle, right, after, 0.5 * tolerance, depth + 1)

    for segment in segments:
        control_length = fsum(
            hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(
                (segment.p0, segment.p1, segment.p2),
                (segment.p1, segment.p2, segment.p3),
                strict=True,
            )
        )
        if not isfinite(control_length):
            raise ValueError("arc length is not finite")
        partition(
            segment, 0.0, 1.0, length(segment, 0.0, 1.0), 5.0e-12 * control_length, 0
        )
    if not intervals:
        return (segments[0].p0,) * n
    prefix = [0.0]
    for _, _, _, distance in intervals:
        prefix.append(prefix[-1] + distance)
    samples = [segments[0].p0]
    for index in range(1, n - 1):
        target = prefix[-1] * index / (n - 1)
        interval_index = min(bisect_right(prefix, target) - 1, len(intervals) - 1)
        segment, left, right, distance = intervals[interval_index]
        fraction = (target - prefix[interval_index]) / distance

        def residual(t: float) -> float:
            if t == 0.0:
                return -fraction
            if t == 1.0:
                return 1.0 - fraction
            return (
                length(segment, left, left + (right - left) * t) / distance - fraction
            )

        t = brent_root(residual, 0.0, 1.0)
        samples.append(segment.evaluate(left + (right - left) * t))
    return tuple(samples) + (segments[-1].p3,)


def _short_bounds(bounds: Bounds2D) -> str:
    return "(" + ", ".join(f"{value:.6g}" for value in bounds) + ")"
