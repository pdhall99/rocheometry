"""Batched nearest points on cubics, by safeguarded multi-start Newton.

This measurement helper depends on numpy; production Rocheometry does not.
Every reference segment whose control-hull box can beat an existing upper
bound is searched. Nine starts per segment resolve competing stationary
points; endpoints are always retained. An independent polynomial-root
implementation is used in tests and in an audit sample from each study.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import rocheometry
from crosschecks.distance.bezier import _SegmentDistance

Array = npt.NDArray[np.float64]


def controls(compound: rocheometry.CompoundBezierPath) -> Array:
    return np.array(
        [[s.p0, s.p1, s.p2, s.p3] for path in compound.paths for s in path.segments],
        dtype=float,
    ).reshape((-1, 4, 2))


def sample(compound: rocheometry.CompoundBezierPath, count: int = 51) -> Array:
    c = controls(compound)
    t = np.linspace(0, 1, count)
    u = 1 - t
    basis = np.array([u**3, 3 * u * u * t, 3 * u * t * t, t**3])
    return (
        np.einsum("ict,cn->int", c, basis).reshape((-1, 2))
        if len(c)
        else np.empty((0, 2))
    )


def two_way_distance(
    fitted: rocheometry.CompoundBezierPath,
    reference: rocheometry.CompoundBezierPath,
    *,
    count: int = 101,
    corners: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """Sample both directions and include exact reference junction positions.

    Each sampled maximum is certified by independent stationary-polynomial
    isolation. Junctions use that same exact nearest-cubic calculation, rather
    than relying on a sample landing near a corner. All values are lengths;
    callers divide by the measured volume radius once.
    """
    if not fitted.paths and not reference.paths:
        return dict(value=0.0, forward=0.0, reverse=0.0, corner=0.0, checks=0)
    forward_points, reverse_points = sample(fitted, count), sample(reference, count)
    forward, index, checks = NearestCurve(reference).maximum(forward_points)
    nearest_fitted = NearestCurve(fitted)
    reverse, reverse_index, reverse_checks = nearest_fitted.maximum(reverse_points)
    corner, corner_checks = 0.0, 0
    if len(corners):
        corner, _, corner_checks = nearest_fitted.maximum(np.asarray(corners))
    return dict(
        value=max(forward, reverse, corner),
        forward=forward,
        reverse=reverse,
        corner=corner,
        forward_point=forward_points[index].tolist(),
        reverse_point=reverse_points[reverse_index].tolist(),
        checks=checks + reverse_checks + corner_checks,
        samples_per_segment=count,
    )


def evaluate(c: Array, t: Array) -> tuple[Array, Array, Array]:
    a = -c[0] + 3 * c[1] - 3 * c[2] + c[3]
    b = 3 * c[0] - 6 * c[1] + 3 * c[2]
    d = 3 * (c[1] - c[0])
    p = ((a * t[..., None] + b) * t[..., None] + d) * t[..., None] + c[0]
    v = (3 * a * t[..., None] + 2 * b) * t[..., None] + d
    w = 6 * a * t[..., None] + 2 * b
    return p, v, w


class NearestCurve:
    def __init__(self, compound: rocheometry.CompoundBezierPath) -> None:
        self.controls = controls(compound)
        self.segments = [s for path in compound.paths for s in path.segments]
        self.lower = (
            self.controls.min(axis=1) if len(self.controls) else np.empty((0, 2))
        )
        self.upper = (
            self.controls.max(axis=1) if len(self.controls) else np.empty((0, 2))
        )
        self.anchors = (
            np.concatenate(
                [evaluate(c, np.linspace(0, 1, 9))[0] for c in self.controls]
            )
            if len(self.controls)
            else np.empty((0, 2))
        )

    def distances(self, points: npt.ArrayLike) -> Array:
        points = np.asarray(points, dtype=float)
        if not len(self.controls):
            return np.full(len(points), np.inf)
        # A real point on a cubic gives a conservative distance upper bound.
        best = np.empty(len(points))
        for i in range(0, len(points), 256):
            z = points[i : i + 256, None, :] - self.anchors[None, :, :]
            best[i : i + 256] = np.min(np.sum(z * z, axis=2), axis=1)
        for k, c in enumerate(self.controls):
            delta = np.maximum(
                np.maximum(self.lower[k] - points, points - self.upper[k]), 0.0
            )
            mask = np.sum(delta * delta, axis=1) <= best * (1 + 1e-14) + 1e-30
            indices = np.nonzero(mask)[0]
            if not len(indices):
                continue
            q = points[indices]
            t = np.broadcast_to(np.linspace(0, 1, 9), (len(q), 9)).copy()
            for _ in range(16):
                p, v, w = evaluate(c, t)
                r = p - q[:, None, :]
                f = np.sum(r * v, axis=2)
                df = np.sum(v * v + r * w, axis=2)
                step = np.divide(f, df, out=np.zeros_like(f), where=df != 0)
                # Limit a Newton step to avoid jumping across all basins.
                t = np.clip(t - np.clip(step, -0.25, 0.25), 0.0, 1.0)
            p, _, _ = evaluate(c, t)
            squared = np.sum((p - q[:, None, :]) ** 2, axis=2)
            squared = np.minimum(
                np.min(squared, axis=1),
                np.minimum(
                    np.sum((q - c[0]) ** 2, axis=1), np.sum((q - c[3]) ** 2, axis=1)
                ),
            )
            best[indices] = np.minimum(best[indices], squared)
        return np.sqrt(best)

    def audit(self, points: Array, distances: Array, count: int = 5) -> float:
        if not len(points):
            return 0.0
        ids = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int)
        errors = []
        for j in ids:
            point = (float(points[j][0]), float(points[j][1]))
            best = float(distances[j])
            for segment in self.segments:
                strategy = _SegmentDistance.from_segment(segment)
                if strategy.lower_bound(point) <= best + 1e-14:
                    best = min(best, strategy.distance(point))
            errors.append(abs(float(distances[j]) - best))
        return max(errors, default=0.0)

    def maximum(
        self, points: Array | Sequence[tuple[float, float]]
    ) -> tuple[float, int, int]:
        """Certify the largest sampled distance with independent root isolation.

        Newton distances are upper bounds (actual points on actual cubics).
        Rechecking their largest candidate until it cannot beat the certified
        maximum prevents a missed Newton basin from becoming a false failure.
        """
        distances = self.distances(points)
        tolerance = 2e-14 * max(1e-100, float(np.max(np.abs(self.controls))))
        best = -1.0
        worst = 0
        checks = 0
        while len(distances):
            j = int(np.argmax(distances))
            if distances[j] <= best + tolerance:
                break
            point = (float(points[j][0]), float(points[j][1]))
            exact = float(distances[j])
            for segment in self.segments:
                strategy = _SegmentDistance.from_segment(segment)
                if strategy.lower_bound(point) <= exact + tolerance:
                    exact = min(exact, strategy.distance(point))
            distances[j] = exact
            checks += 1
            if exact > best:
                best = exact
                worst = j
            if checks > len(points):
                raise RuntimeError("nearest-curve maximum did not certify")
        return best, worst, checks
