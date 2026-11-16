"""Geometric validation for approximate component-enclosing sections.

A test helper: the library never validates its own sections at runtime.
"""

from dataclasses import dataclass
from math import hypot
from sys import float_info

from crosschecks.distance.bezier import _PathDistance

from rocheometry._bezier import BezierPath
from rocheometry._computers.section_path.reference import (
    SectionReference,
)
from rocheometry._geometry import Point2D
from rocheometry._planes import ORBITAL_PLANE
from rocheometry._validation import (
    _normalize_finite_positive_float,
    require_instance,
    require_int,
)

# Resolution budget: curve validation distribution samples.
CURVE_VALIDATION_DISTRIBUTION_SAMPLES = 2049
# Resolution budget: curve intersection coefficient tolerance.
CURVE_INTERSECTION_COEFFICIENT_TOLERANCE = 64.0 * float_info.epsilon


@dataclass(frozen=True, slots=True)
class DirectedCurveError:
    """Largest sampled nearest-curve distance in one direction.

    Attributes:
        distance: Largest sampled nearest-curve distance in the input length units.
        source_point: Source-curve point where the maximum sampled distance occurs.
    """

    distance: float
    source_point: Point2D


@dataclass(frozen=True, slots=True)
class CurveErrorReport:
    """Symmetric geometric error summary for one approximate section.

    Attributes:
        reference_to_fit: Maximum sampled distance from reference to fit.
        fit_to_reference: Maximum sampled distance from fit to reference.
        scale: Positive normalization length, normally the reference volume radius.
        median_distance: Median of the sampled nearest-curve distance distribution.
        p95_distance: 95th percentile of that distance distribution.
        p99_distance: 99th percentile of that distance distribution.
        reference_tolerance: Traced reference radial tolerance in length units;
            this does not bound the sampled Hausdorff estimate.
        sample_spacing: Estimated maximum mean arc-length sample spacing in
            length units, computed from the two sampled paths.
    """

    reference_to_fit: DirectedCurveError
    fit_to_reference: DirectedCurveError
    scale: float
    median_distance: float
    p95_distance: float
    p99_distance: float
    reference_tolerance: float
    sample_spacing: float

    @property
    def hausdorff_distance(self) -> float:
        """The symmetric sampled Hausdorff-like distance."""
        return max(self.reference_to_fit.distance, self.fit_to_reference.distance)

    @property
    def normalized_hausdorff(self) -> float:
        """The maximum distance in units of the configured scale."""
        return self.hausdorff_distance / self.scale

    @property
    def normalized_median(self) -> float:
        """The median nearest-curve distance in scale units."""
        return self.median_distance / self.scale

    @property
    def normalized_p95(self) -> float:
        """The 95th-percentile distance in scale units."""
        return self.p95_distance / self.scale

    @property
    def normalized_p99(self) -> float:
        """The 99th-percentile distance in scale units."""
        return self.p99_distance / self.scale


def evaluate_component_section_fit_error(
    reference: SectionReference,
    path: BezierPath,
    *,
    distribution_samples: int = CURVE_VALIDATION_DISTRIBUTION_SAMPLES,
    scale: float | None = None,
) -> CurveErrorReport:
    """Compare cubics directly with the authoritative traced reference path.

    Closed paths are compared over the full section; open paths are compared
    with the reference upper half. Both directions use uniform arc-length
    samples, plus knots and segment midpoints to resolve narrow neck features.
    Target distances minimize over cubics, without further field solves.

    Args:
        reference: Orbital-plane geometry and traced reference path.
        path: Fitted closed section or connected open upper-half path in the
            reference's separation-unit chart.
        distribution_samples: Samples per direction, including endpoints;
            integer at least `2`.
        scale: Positive normalization length, or `None` to use volume radius.

    Returns:
        Bidirectional sampled Hausdorff-like distances and their distribution.
        These are numerical estimates, not analytic maximum-error certificates.
    """
    if reference.plane != ORBITAL_PLANE:
        raise ValueError("component section fit validation requires an orbital plane")
    require_instance(path, BezierPath, name="path")
    scale = reference.volume_radius if scale is None else scale
    scale = _normalize_finite_positive_float(scale, name="scale")
    distribution_samples = require_int(
        distribution_samples, name="distribution_samples"
    )
    if distribution_samples < 2:
        raise ValueError("distribution_samples must be an integer of at least 2")
    fit = path
    reference_path = reference.path if path.closed else reference.upper_path
    distance_to_reference = _PathDistance(reference_path)
    distance_to_fit = _PathDistance(fit)
    reference_samples = reference_path.sample(
        distribution_samples, spacing="arc_length"
    )
    fit_samples = fit.sample(distribution_samples, spacing="arc_length")
    reference_distances = tuple(distance_to_fit(point) for point in reference_samples)
    fit_distances = tuple(distance_to_reference(point) for point in fit_samples)
    reference_error = _directed_path_error(
        reference_path, distance_to_fit, reference_samples, reference_distances
    )
    fit_error = _directed_path_error(
        fit, distance_to_reference, fit_samples, fit_distances
    )
    distribution = sorted((*reference_distances, *fit_distances))
    return CurveErrorReport(
        reference_to_fit=reference_error,
        fit_to_reference=fit_error,
        scale=scale,
        median_distance=_percentile(distribution, 0.5),
        p95_distance=_percentile(distribution, 0.95),
        p99_distance=_percentile(distribution, 0.99),
        reference_tolerance=reference.path_tolerance * reference.volume_radius,
        sample_spacing=max(
            _polyline_length(reference_samples), _polyline_length(fit_samples)
        )
        / (distribution_samples - 1),
    )


def _directed_path_error(
    path: BezierPath,
    distance: _PathDistance,
    samples: tuple[Point2D, ...],
    distances: tuple[float, ...],
) -> DirectedCurveError:
    # Include knots and segment midpoints so small adaptively traced neck
    # features are checked even between the uniform arc-length samples.
    probes = tuple(
        point
        for segment in path.segments
        for point in (segment.p0, segment.evaluate(0.5))
    ) + (path.segments[-1].p3,)
    vertex_error = _largest_sample_error(
        probes, tuple(distance(point) for point in probes)
    )
    sample_error = _largest_sample_error(samples, distances)
    return max((vertex_error, sample_error), key=lambda error: error.distance)


def _largest_sample_error(
    points: tuple[Point2D, ...],
    distances: tuple[float, ...],
) -> DirectedCurveError:
    index = max(range(len(distances)), key=distances.__getitem__)
    return DirectedCurveError(distances[index], points[index])


def _polyline_length(points: tuple[Point2D, ...]) -> float:
    return sum(
        _distance(start, end) for start, end in zip(points, points[1:], strict=False)
    )


def _polyline_self_intersects(points: tuple[Point2D, ...]) -> bool:
    segment_count = len(points) - 1
    for first in range(segment_count):
        a0 = points[first]
        a1 = points[first + 1]
        for second in range(first + 1, segment_count):
            if second == first + 1:
                continue
            if first == 0 and second == segment_count - 1:
                continue
            b0 = points[second]
            b1 = points[second + 1]
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


def _segments_intersect(
    a0: Point2D,
    a1: Point2D,
    b0: Point2D,
    b1: Point2D,
) -> bool:
    scale = max(
        1.0,
        *(abs(value) for point in (a0, a1, b0, b1) for value in point),
    )
    tolerance = CURVE_INTERSECTION_COEFFICIENT_TOLERANCE * scale * scale
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)

    if _opposite_signs(o1, o2, tolerance) and _opposite_signs(
        o3,
        o4,
        tolerance,
    ):
        return True
    return (
        (abs(o1) <= tolerance and _point_on_segment(b0, a0, a1, tolerance))
        or (abs(o2) <= tolerance and _point_on_segment(b1, a0, a1, tolerance))
        or (abs(o3) <= tolerance and _point_on_segment(a0, b0, b1, tolerance))
        or (abs(o4) <= tolerance and _point_on_segment(a1, b0, b1, tolerance))
    )


def _orientation(origin: Point2D, first: Point2D, second: Point2D) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _opposite_signs(first: float, second: float, tolerance: float) -> bool:
    return (first > tolerance and second < -tolerance) or (
        first < -tolerance and second > tolerance
    )


def _point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
    tolerance: float,
) -> bool:
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def _distance(first: Point2D, second: Point2D) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])
