"""Piecewise smooth reference limbs with exact two-maximum junctions."""

from bisect import bisect_left
from dataclasses import dataclass
from math import atan, cos, hypot, inf, nextafter, pi, sin, sqrt, ulp
from sys import float_info

from rocheometry._bezier import BezierPath, CubicBezier
from rocheometry._computers.ownership import certify_part
from rocheometry._computers.saddle import _neck_width
from rocheometry._computers.section_path.gauge_fixed import _neck_angle
from rocheometry._computers.section_path.traced import (
    SECTION_MAX_REFINEMENT_DEPTH,
    _Knot,
    _trace_polar_field,
    _trig,
)
from rocheometry._computers.silhouette.chord import _curvature_term_bounds
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point2D, Point3D

# Insensitive control: limb branch bound ulps.
LIMB_BRANCH_BOUND_ULPS = 32
# Insensitive control: limb junction angle root tolerance.
LIMB_JUNCTION_ANGLE_ROOT_TOLERANCE = 8e-16
# Insensitive control: limb junction polish acceptance factor.
LIMB_JUNCTION_POLISH_ACCEPTANCE_FACTOR = 1.0


@dataclass(frozen=True, slots=True)
class _Junction:
    angle: float
    point: Point2D
    before: Point3D
    after: Point3D


@dataclass(frozen=True, slots=True)
class _Sample:
    maxima: tuple[Point3D, ...]
    margins: tuple[float, ...]


def _opposite(a: float, b: float) -> bool:
    """Compare signs without underflowing their product."""
    return a < 0 < b or b < 0 < a


class _RestartTrace(Exception):
    pass


def _corridor_concave(
    field: ProjectedField, left: float, right: float, lower: float, upper: float
) -> bool:
    """Certify concavity on every sky ray and radius in the interval.

    The central mass's m/r term proves a positive inner bound for a limb
    radius; the bounding sphere supplies the outer bound. The distance
    rectangles enclose the whole annular sky sector, not just its endpoints.
    """
    u_upper = nextafter(nextafter(field.l1.potential.value, inf) - field.depth, inf)
    emin, emax = min(1.0, field._e), max(1.0, field._e)
    m, companion = field.field.component_mass, field.field.companion_mass
    allowance = LIMB_BRANCH_BOUND_ULPS * float_info.epsilon
    total_lower = nextafter(1 + field.field._mass_error, 0.0)
    total_upper = nextafter(1 + field.field._mass_error, inf)
    square = 1 + field.outer**2
    # Convexity of the companion's inverse distance at fixed radius and
    # completion of the centrifugal square give this background lower bound.
    background = companion / sqrt(square) + companion**2 / (2 * total_upper) * (
        1 - (square**-1.5 - 1) ** 2
    )
    background -= allowance * (abs(background) + companion + companion**2 / total_lower)
    inner = nextafter(m / (u_upper - max(0.0, background)) / sqrt(emax), 0.0)
    inner *= 1 - allowance
    front = (
        field.start_point[0] if field.start_point is not None else field.l1.position[0]
    )
    radial_upper = (
        min(field.outer, front / cos(right)) if cos(right) > 0 else field.outer
    )
    left_upper = min(field.outer, front / cos(left)) if cos(left) > 0 else field.outer
    closest = min(left_upper, max(inner, cos(left)))
    b_lower = emin * hypot(closest * cos(left) - 1, closest * sin(left)) ** 2
    a_lower, a_upper = emin * inner**2, emax * radial_upper**2
    b_upper = emax * (1 + field.outer) ** 2
    a_lower *= 1 - allowance
    b_lower *= 1 - allowance
    a_upper *= 1 + allowance
    b_upper *= 1 + allowance
    widening = allowance * max(abs(lower), abs(upper))
    lower, upper = lower - widening, upper + widening
    z_lower = 0.0 if lower <= 0 <= upper else field._e * min(lower**2, upper**2)
    z_upper = field._e * max(lower**2, upper**2)
    z_lower *= 1 - allowance
    z_upper *= 1 + allowance
    first = _curvature_term_bounds(
        a_lower, a_upper, z_lower, z_upper, field.outer**2 * (1 + allowance)
    )[1]
    second = _curvature_term_bounds(b_lower, b_upper, z_lower, z_upper)[1]
    value = field._s2 * (1 + field.field._mass_error) + field._e * (
        field.field.component_mass * first + field.field.companion_mass * second
    )
    scale = (
        field._s2
        + abs(field._e * field.field.component_mass * first)
        + abs(field._e * field.field.companion_mass * second)
    )
    return value + allowance * scale < 0


def _branch_difference(
    field: ProjectedField, angle: float
) -> tuple[float, tuple[float, Point3D, float, Point3D]]:
    """Difference of branch roots, retaining their sub-ULP remainders."""
    first, p = field.branch_radius(angle, 0)
    second, q = field.branch_radius(angle, -1)
    c, s = _trig(angle)
    gp, gq = field.field.gradient(*p), field.field.gradient(*q)
    dp = gp[0] * c + (gp[1] * field.c - gp[2] * field.s) * s
    dq = gq[0] * c + (gq[1] * field.c - gq[2] * field.s) * s
    # Retain the sub-ULP root remainders. Factoring the difference of
    # branch potentials avoids subtracting two shared anchor offsets.
    difference = (
        (first - second)
        + field.field.potential_difference(q, p) / dq
        + field._residual(p) * (1 / dq - 1 / dp)
    )

    return difference, (first, p, second, q)


def _junction(
    field: ProjectedField, left: float, right: float, *, polish: bool = False
) -> _Junction:
    """Cross two independently bracketed branch limb radii in one dimension."""
    from rocheometry._numerics.roots import brent_root

    values: dict[float, tuple[float, Point3D, float, Point3D]] = {}
    differences: dict[float, float] = {}

    def difference(angle: float) -> float:
        if angle in differences:
            return differences[angle]
        value, data = _branch_difference(field, angle)
        values[angle] = data
        differences[angle] = value
        return value

    angle = brent_root(
        difference,
        left,
        right,
        absolute_tolerance=0.0,
        relative_tolerance=LIMB_JUNCTION_ANGLE_ROOT_TOLERANCE,
    )
    if angle not in values:
        difference(angle)
    first, p, second, q = values[angle]
    radius = (first + second) / 2
    c, s = _trig(angle)
    point = radius * c, radius * s
    # Re-isolate at the one shared sky point: both one-sided derivatives
    # belong to precisely the endpoint used by both adjacent Bezier pieces.
    maxima = field.branch_maxima(*point)
    if len(maxima) < 2:
        raise RuntimeError("junction maxima merged before the crossing")
    points = maxima[0], maxima[-1]
    floor = field._value_roundoff(*point)
    slopes = []
    for maximum in points:
        certify_part(
            field.field, field.l1, field.watershed, maximum, contact=field.contact
        )
        if abs(field._residual(maximum)) > floor:
            raise RuntimeError("branch crossing is not on the global limb")
    if abs(field.value(*point)) > floor:
        raise RuntimeError("branch crossing is not a global junction")
    for maximum in points:
        gx, gy, gz = field.field.gradient(*maximum)
        gt = gy * field.c - gz * field.s
        radial = c * gx + s * gt
        if radial >= 0:
            raise RuntimeError("limb junction is not star-shaped")
        slopes.append(-radius * (-s * gx + c * gt) / radial)
    fl, fr = differences[left], differences[right]
    if fl == fr == 0:
        raise ValueError("junction crossing lacks a sign-changing bracket")
    before_index = 0 if fl > 0 or (fl == 0 and fr < 0) else 1
    after_index = 0 if fr > 0 or (fr == 0 and fl < 0) else 1
    # At a simple crossing the analytic derivative fixes its orientation,
    # even when last-bit height noise reverses signs in an ULP-wide bracket.
    # An exactly zero derivative keeps the bracket's higher-order crossing.
    # There is deliberately no tangent-jump cutoff.
    if slopes[0] != slopes[1]:
        before_index = 0 if slopes[0] < slopes[1] else 1
        after_index = 1 - before_index
    before, after = points[before_index], points[after_index]
    result = _Junction(angle, point, before, after)
    if polish:
        from rocheometry._computers.silhouette.junction_polish import _newton_polish

        try:
            candidate = _newton_polish(field, left, right, seed_angle=angle)
        except RuntimeError:
            pass
        else:
            if left <= candidate.angle <= right and max(
                abs(field._residual(p)) for p in (candidate.before, candidate.after)
            ) <= LIMB_JUNCTION_POLISH_ACCEPTANCE_FACTOR * max(
                abs(field._residual(p)) for p in (before, after)
            ):
                ordered = sorted(
                    (candidate.before, candidate.after),
                    key=lambda p: p[1] * field.s + p[2] * field.c,
                )
                result = _Junction(
                    candidate.angle,
                    candidate.point,
                    ordered[before_index],
                    ordered[after_index],
                )
    return result


def _score(
    field: ProjectedField, angle: float, maxima: tuple[Point3D, ...]
) -> tuple[float, float]:
    """Height difference and its derivative along the global limb."""
    r = field.radius(angle)
    c, s = _trig(angle)
    p, q = maxima[0], maxima[-1]
    value = field.field.potential_difference(p, q)
    gradients = [field.field.gradient(*v) for v in (p, q)]
    sky = [(g[0], g[1] * field.c - g[2] * field.s) for g in gradients]
    gx, gt = sky[0 if value >= 0 else 1]
    dr = -r * (-s * gx + c * gt) / (c * gx + s * gt)
    dx, dt = dr * c - r * s, dr * s + r * c
    derivative = (sky[0][0] - sky[1][0]) * dx + (sky[0][1] - sky[1][1]) * dt
    return value, derivative


def _margin(field: ProjectedField, angle: float, primary: int) -> tuple[float, float]:
    """Secondary height and its derivative on the primary branch's limb."""
    r, p = field.branch_radius(angle, primary)
    c, s = _trig(angle)
    maxima = field.branch_maxima(r * c, r * s)
    q = maxima[-1 if primary == 0 else 0]
    gp, gq = field.field.gradient(*p), field.field.gradient(*q)
    px, pt = gp[0], gp[1] * field.c - gp[2] * field.s
    qx, qt = gq[0], gq[1] * field.c - gq[2] * field.s
    radial = px * c + pt * s
    other_radial = qx * c + qt * s
    dr = -r * (-s * px + c * pt) / radial
    # Correct the residual left at the binary64 primary radius, retaining
    # the shared level offset only once in the difference.
    value = field.field.potential_difference(q, p) + field._residual(p) * (
        1 - other_radial / radial
    )
    derivative = qx * (dr * c - r * s) + qt * (dr * s + r * c)
    return value, derivative


def _partition_junction(
    field: ProjectedField, left: float, right: float, p: Point3D, q: Point3D
) -> _Junction | None:
    """Locate a branch transfer even if its coexistence interval is very short.

    Split the LOS between the endpoint winners. Each constrained half has a
    maximum even outside the coexistence interval. Their height crossing is
    either one stationary point moving through the separator (smooth), or
    two distinct maxima exchanging visibility (a true junction).
    """
    from rocheometry._numerics.roots import brent_root

    wp = p[1] * field.s + p[2] * field.c
    wq = q[1] * field.s + q[2] * field.c
    if wp == wq:
        return None
    lower, upper = sorted((wp, wq))
    if _corridor_concave(field, left, right, lower, upper):
        return None
    separator = (wp + wq) / 2
    valid: dict[float, float] = {}
    winners: dict[float, tuple[Point3D, Point3D]] = {}
    stationary: dict[float, tuple[Point3D, ...]] = {}

    def difference(angle: float) -> float:
        r = field.radius(angle)
        c, s = _trig(angle)
        x, t = r * c, r * s
        chord = field._chord(x, t)
        maxima = field.branch_maxima(x, t)
        stationary[angle] = maxima
        if len(maxima) == 2:
            valid[angle] = _score(field, angle, maxima)[0]
        candidates = [(v[1] * field.s + v[2] * field.c, v) for v in maxima]
        candidates.extend(
            (w, field._point(x, t, w)) for w in (-chord.limit, chord.limit)
        )
        if -chord.limit < separator < chord.limit:
            candidates.append((separator, field._point(x, t, separator)))
        before = [v for w, v in candidates if w <= separator]
        after = [v for w, v in candidates if w >= separator]
        if not before or not after:
            return -pi / 2 if not before else pi / 2

        def best(points: list[Point3D]) -> Point3D:
            chosen = points[0]
            for candidate in points[1:]:
                if field.field.potential_difference(candidate, chosen) > 0:
                    chosen = candidate
            return chosen

        a, b = best(before), best(after)
        winners[angle] = (a, b)
        return atan(field.field.potential_difference(a, b))

    fl, fr = difference(left), difference(right)
    if not _opposite(fl, fr):
        return None
    angle = brent_root(
        difference,
        left,
        right,
        absolute_tolerance=0.0,
        relative_tolerance=LIMB_JUNCTION_ANGLE_ROOT_TOLERANCE,
    )
    difference(angle)
    if any(p not in stationary[angle] for p in winners[angle]):
        return None
    if winners[angle][0] == winners[angle][1]:
        return None
    r = field.radius(angle)
    c, s = _trig(angle)
    if len(field.branch_maxima(r * c, r * s)) < 2:
        return None
    # Locate a sign-changing pair *inside* the coexistence region. No angular
    # width is prescribed: start at adjacent floats and expand geometrically.
    distance = max(angle - nextafter(angle, left), nextafter(angle, right) - angle)
    while True:
        lo, hi = max(left, angle - distance), min(right, angle + distance)
        difference(lo)
        difference(hi)
        ordered = sorted(valid)
        for a, b in zip(ordered, ordered[1:]):
            if _opposite(valid[a], valid[b]):
                ha, _ = _branch_difference(field, a)
                hb, _ = _branch_difference(field, b)
                if _opposite(ha, hb):
                    return _junction(field, a, b)
        if lo == left and hi == right:
            return None
        distance *= 2


def _hidden_junctions(
    field: ProjectedField,
    left: float,
    right: float,
    first: tuple[Point3D, ...],
    last: tuple[Point3D, ...],
) -> list[_Junction]:
    """Search a hidden branch's margin maximum, then both visibility crossings."""
    from rocheometry._numerics.roots import brent_root

    if len(first) == len(last) == 2:
        a, da = _score(field, left, first)
        b, db = _score(field, right, last)
        if _opposite(a, b):
            return [_junction(field, left, right)]
        orientation = -1 if a > 0 else 1
        if orientation * da > 0 > orientation * db:
            primary = 0 if a > 0 else -1

            def derivative(angle: float) -> float:
                return _margin(field, angle, primary)[1]

            peak = brent_root(
                derivative,
                left,
                right,
                absolute_tolerance=0.0,
                relative_tolerance=LIMB_JUNCTION_ANGLE_ROOT_TOLERANCE,
            )
            if _margin(field, peak, primary)[0] > 0:
                return [_junction(field, left, peak), _junction(field, peak, right)]
        return []
    p = max(first, key=field._residual)
    q = max(last, key=field._residual)
    junction = _partition_junction(field, left, right, p, q)
    return [junction] if junction is not None else []


def _aligned_arm(point: Point2D, control: Point2D, tangent: Point2D) -> Point2D:
    """Retain the coarse displacement and round the finer component once.

    Subtracting two absolute control coordinates can erase a short arm's
    direction. Its coarsest representable parameter step fixes the length;
    derive the other coordinate from that actual displacement and tangent.
    """
    active = [k for k in (0, 1) if tangent[k] and control[k] != point[k]]
    if not active:
        return control
    axis = max(active, key=lambda k: ulp(point[k]) / abs(tangent[k]))
    other = 1 - axis
    result = list(control)
    result[other] = point[other] + (control[axis] - point[axis]) * (
        tangent[other] / tangent[axis]
    )
    return result[0], result[1]


def _junction_arms(
    segments: list[CubicBezier], junctions: list[_Junction], field: ProjectedField
) -> list[CubicBezier]:
    tangents = {
        j.point: (
            field.branch_knot(j.angle, j.point, j.before).derivative,
            field.branch_knot(j.angle, j.point, j.after).derivative,
        )
        for j in junctions
    }
    result = []
    for segment in segments:
        p1, p2 = segment.p1, segment.p2
        if segment.p0 in tangents:
            p1 = _aligned_arm(segment.p0, p1, tangents[segment.p0][1])
        if segment.p3 in tangents:
            p2 = _aligned_arm(segment.p3, p2, tangents[segment.p3][0])
        result.append(CubicBezier(segment.p0, p1, p2, segment.p3))
    return result


def trace_limb(field: ProjectedField, tolerance: float) -> BezierPath:
    """Trace each maximizing branch separately, preserving tangent corners."""
    samples: dict[float, _Sample] = {}
    angles: list[float] = []
    junctions: list[_Junction] = []

    def radius(angle: float) -> float:
        r = field.radius(angle)
        if field.s == 0 or field.c == 0 or angle in (0.0, pi) or angle in samples:
            return r
        c, s = _trig(angle)
        maxima = field.branch_maxima(r * c, r * s)
        index = bisect_left(angles, angle)
        angles.insert(index, angle)
        samples[angle] = _Sample(maxima, tuple(field._residual(p) for p in maxima))
        for a, b in (
            (angles[index - 1], angle) if index else (angle, angle),
            (angle, angles[index + 1]) if index + 1 < len(angles) else (angle, angle),
        ):
            if a == b or any(a <= j.angle <= b for j in junctions):
                continue
            found = _hidden_junctions(field, a, b, samples[a].maxima, samples[b].maxima)
            if found:
                junctions.extend(found)
                junctions.sort(key=lambda j: j.angle)
                raise _RestartTrace
        return r

    while True:
        try:
            upper = []
            ends = [field.start, *(j.angle for j in junctions), pi]
            for index, (start, end) in enumerate(zip(ends[:-1], ends[1:], strict=True)):
                boundary = (
                    junctions[index - 1]
                    if index
                    else junctions[0]
                    if junctions
                    else None
                )
                branch_index = None
                if boundary is not None:
                    selected = boundary.after if index else boundary.before
                    other = boundary.before if index else boundary.after
                    branch_index = (
                        0
                        if selected[1] * field.s + selected[2] * field.c
                        < other[1] * field.s + other[2] * field.c
                        else -1
                    )

                def knot(angle: float) -> _Knot:
                    if index and angle == start:
                        j = junctions[index - 1]
                        return field.branch_knot(angle, j.point, j.after)
                    if index < len(junctions) and angle == end:
                        j = junctions[index]
                        return field.branch_knot(angle, j.point, j.before)
                    radius(angle)
                    if branch_index is not None:
                        r = field.radius(angle)
                        c, s = _trig(angle)
                        point = (
                            field.start_point
                            if angle == field.start and field.start_point is not None
                            else (r * c, r * s)
                        )
                        maxima = field.branch_maxima(*point)
                        if len(maxima) == 2:
                            return field.branch_knot(angle, point, maxima[branch_index])
                    return field.knot(angle)

                piece = _trace_polar_field(
                    radius,
                    knot,
                    start=start,
                    end=end,
                    scale=field.scale,
                    tolerance=tolerance,
                    upper_only=True,
                    # A split critical branch keeps the original critical cap.
                    depth_limit=56 if field.contact else SECTION_MAX_REFINEMENT_DEPTH,
                    front_panel=(
                        _neck_angle(
                            _neck_width(field.field, field.depth, field.l1),
                            (end - start) / 4,
                        )
                        if start == 0 and field.depth < 0
                        else None
                    ),
                )
                upper.extend(piece.segments)
            if junctions:
                upper = _junction_arms(upper, junctions, field)
            from rocheometry._geometry import AffineTransform2D

            reflection = AffineTransform2D(yy=-1.0)
            lower = tuple(
                segment.reversed().transformed(reflection)
                for segment in reversed(upper)
            )
            return BezierPath(tuple(upper) + lower, closed=not field.contact)
        except _RestartTrace:
            continue
