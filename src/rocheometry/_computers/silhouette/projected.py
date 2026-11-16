"""The line-of-sight envelope field for quadrature silhouettes."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as state
from math import atan2, fsum, hypot, inf, nextafter, pi, sqrt, ulp
from sys import float_info

from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.neck import (
    _DD,
    _add,
    _neck_field,
)
from rocheometry._computers.ownership import FIELD_CERTIFICATE_BOUND_ULPS, certify_part
from rocheometry._computers.section_path.traced import _Knot, _trig
from rocheometry._computers.silhouette.chord import _Chord
from rocheometry._computers.surface_radius import SurfaceRadiusComputer
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import Point2D, Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._planes import CentrelinePlane, _transverse_direction
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint

# Insensitive control: los residual value ulps.
LOS_RESIDUAL_VALUE_ULPS = 8
# Insensitive control: los maximum radius root relative tolerance.
LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE = 2e-15
# Insensitive control: los rim crossing root tolerance.
LOS_RIM_CROSSING_ROOT_TOLERANCE = 1e-15
# Insensitive control: los start angle match tolerance.
LOS_START_ANGLE_MATCH_TOLERANCE = 8e-16
# Insensitive control: los stationarity relative tolerance.
LOS_STATIONARITY_RELATIVE_TOLERANCE = 2e-13


class _EmptyLineOfSightError(ValueError):
    """The sky line has no point in the component's bounding sphere."""


@dataclass
class ProjectedField:
    """One level's polar sky field, with per-trace exact-point sharing.

    All stationary points are isolated using interval bounds on the whole
    bounding chord; anchored differences select the global maximum.
    Accepted limb maxima must be interior and have negative LOS curvature.
    Contact tracing starts where this stationary limb intersects the watershed.
    """

    field: RocheField
    depth: float
    plane: CentrelinePlane
    radius_computer: SurfaceRadiusComputer
    l1: CriticalPoint
    outer_computer: CriticalPointComputer
    watershed: WatershedComputer
    _points: dict[float, tuple[float, Point3D]] = state(default_factory=dict)
    _maxima: dict[Point2D, Point3D] = state(default_factory=dict)
    _stationary: dict[Point2D, tuple[Point3D, ...]] = state(default_factory=dict)
    _neck_points: dict[Point2D, tuple[_DD, _DD, _DD]] = state(default_factory=dict)

    def __post_init__(self) -> None:
        self.second = _transverse_direction(self.plane)
        self.c, self.s = self.second[1], -self.second[2]
        self.line_of_sight = (0.0, self.s, self.c)
        self.critical = self.depth == 0
        self.contact = self.depth > 0
        self.outer_point = self.outer_computer(self.field)
        self.outer = abs(self.outer_point.position[0])
        self._e = self.c * self.c + self.s * self.s
        self._s2 = self.s * self.s
        self._outer_squared = self.outer**2
        self._certified = (
            self.field.component_mass / self.outer** 3
            > 1
            + FIELD_CERTIFICATE_BOUND_ULPS * float_info.epsilon
            + abs(self.field._mass_error)
        )
        self._front_offset = self.depth
        self._own_offset = self.field._depth_offset(
            self.depth, self.l1.position[0], self.outer_point.position[0]
        )
        self._back_anchor = self.contact and abs(self._own_offset) < abs(self.depth)
        self._back_offset = (
            self._own_offset if self._back_anchor else self._front_offset
        )
        self._canonical_field = type(self.field) is RocheField
        self.scale = self.radius_computer(self.field, self.depth, (0.0, 0.0, 1.0))
        self.start_point = self._crossing() if self.contact else None
        self.start = (
            atan2(self.start_point[1], self.start_point[0]) if self.start_point else 0.0
        )
        self.start_radius = (
            self._polish_start() if self.start_point is not None else None
        )

    def _point(self, x: float, t: float, w: float) -> Point3D:
        return (x, t * self.c + w * self.s, -t * self.s + w * self.c)

    def _los(self, point: Point3D) -> tuple[float, float]:
        _, gy, gz = self.field.gradient(*point)
        h = self.field.hessian(*point)
        return (
            gy * self.s + gz * self.c,
            self.s * (h[1][1] * self.s + h[1][2] * self.c)
            + self.c * (h[2][1] * self.s + h[2][2] * self.c),
        )

    def _chord(self, x: float, t: float) -> _Chord:
        e = self._e
        square = (self._outer_squared - x * x) / e - t * t
        # Enclose subtraction and coefficient rounding, especially when the
        # sky point nearly touches the sphere. A returned limb winner must
        # still lie strictly inside the original, unexpanded sphere.
        error = (
            FIELD_CERTIFICATE_BOUND_ULPS
            * float_info.epsilon
            * (self._outer_squared + x * x + e * t * t)
            / e
        )
        limit = nextafter(sqrt(square + error), inf) if square + error > 0 else 0.0
        return _Chord(
            x * x + e * t * t,
            (x - 1) ** 2 + e * t * t,
            self.field.component_mass,
            self.field.companion_mass,
            e,
            self._s2,
            t * self.c * self.s,
            limit,
            self.field._mass_error,
        )

    def _residual(self, p: Point3D) -> float:
        if self._back_anchor and p[0] < 0:
            return (
                self.field.potential_difference(p, self.outer_point.position)
                + self._back_offset
            )
        if self._canonical_field:
            # Positions are finite lifted chord points and the anchor is the
            # computed interior L1. Retain the helper's exact arithmetic,
            # avoiding repeated public argument validation and offset lookup.
            return (
                self.field._axis_potential_difference(p, self.l1.position[0])
                + self._front_offset
            )
        return self.field.potential_difference(p, self.l1.position) + self._front_offset

    def maxima(self, x: float, t: float) -> tuple[Point3D, ...]:
        """Global-maximum candidates in LOS order, including constrained ends.

        A branch can be omitted only when its entire interval is certified
        below both the level and an attained potential lower bound.
        """
        # Equal numeric keys, including +/-0 and numpy scalar inputs, must
        # produce the same arithmetic and signed-zero bits on a cache miss.
        x, t = float(x), float(t)
        x, t = (0.0 if x == 0.0 else x), (0.0 if t == 0.0 else t)
        key = (x, t)
        if key in self._stationary:
            return self._stationary[key]
        if t < 0:
            result = tuple((p[0], -p[1], -p[2]) for p in reversed(self.maxima(x, -t)))
            self._stationary[key] = result
            return result
        chord = self._chord(x, t)
        if chord.limit == 0.0:
            raise _EmptyLineOfSightError(
                "line of sight does not meet the component bound"
            )
        certified = self._certified
        if certified and (t == 0.0 or self.s == 0.0 or self.c == 0.0):
            # m / outer**3 > 1 by the axial equilibrium equation. Hence
            # D/w < 0 throughout the sphere in either principal view.
            result = (self._point(x, t, 0.0),)
        else:
            # The full chord is accounted for analytically when T=m/r0^3+
            # mu/r1^3 > 1 throughout the sphere: D>0 on w<0, and D<0 on
            # w>=t*s/c. Otherwise the interval solver examines the full chord.
            upper = min(chord.limit, t * self.s / self.c) if self.c else chord.limit
            anchor = self.outer_point if self._back_anchor and x < 0 else self.l1
            offset = (
                self._back_offset if self._back_anchor and x < 0 else self._front_offset
            )
            # An attained lower bound is needed for safe pruning. This coarse
            # absolute-potential allowance is used only for pruning, never for
            # root accuracy or comparing the saddle-relative maximum heights.
            rounding = None

            def attained_lower(w: float) -> float:
                nonlocal rounding
                if rounding is None:
                    rounding = (
                        FIELD_CERTIFICATE_BOUND_ULPS
                        * float_info.epsilon
                        * (
                            chord.m / sqrt(chord.a)
                            + chord.mu / sqrt(chord.b)
                            + 0.5
                            * (
                                (x - chord.mu) ** 2
                                + (abs(t * self.c) + chord.limit * self.s) ** 2
                            )
                            + (abs(self.l1.potential.value) + abs(self.depth))
                            + abs(anchor.potential.value)
                        )
                    )
                return self._residual(self._point(x, t, w)) - rounding

            roots = chord.stationary(
                0.0 if certified else None,
                upper if certified else None,
                attained_lower=attained_lower,
                level_upper=lambda lo, hi: chord.level_upper(
                    x, t, self.c, self.s, anchor.position[0], offset, lo, hi
                ),
            )
            candidates = [self._point(x, t, w) for w in roots if chord.curvature(w) < 0]
            if not certified and chord.derivative(-chord.limit) <= 0:
                candidates.insert(0, self._point(x, t, -chord.limit))
            if chord.derivative(chord.limit) >= 0:
                candidates.append(self._point(x, t, chord.limit))
            if not candidates:
                if chord.limit != 0:
                    raise RuntimeError("no maximum on the bounded line of sight")
                candidates = [self._point(x, t, 0.0)]
            result = tuple(candidates)
        self._stationary[key] = result
        return result

    def _bounding_maximum(self, x: float, t: float) -> Point3D:
        """Return the global maximum, independently of caches and warm starts.

        Stationary points are isolated over the whole bounding chord with
        interval bounds on D and D'. Ties use LOS order reproducibly; reference
        tracing retains both one-sided derivatives at a junction.
        """
        key = (x, t)
        if key not in self._maxima:
            self._maxima[key] = max(self.maxima(x, t), key=self._residual)
        return self._maxima[key]

    def branch_maxima(self, x: float, t: float) -> tuple[Point3D, ...]:
        """Return every interior maximum, without global-winner pruning.

        Reference junction continuation also needs a currently hidden branch.
        Isolating the whole chord without a level/lower-bound pruning callback
        retains that branch independently of the ordinary global evaluator.
        """
        chord = self._chord(x, t)
        left, right = (
            (0.0, min(chord.limit, t * self.s / self.c))
            if self._certified and t >= 0 and self.c
            else (None, None)
        )
        return tuple(
            self._point(x, t, w)
            for w in chord.stationary(left, right)
            if chord.curvature(w) < 0
        )

    def branch_radius(self, angle: float, branch: int) -> tuple[float, Point3D]:
        """Solve a single isolated maximum branch's level on a sky ray.

        Branches are ordered along the line of sight; junction continuation
        uses the first and last maxima while both remain distinct.
        """
        c, s = _trig(angle)
        initial = self.radius(angle)
        points: dict[float, Point3D] = {}

        def residual(factor: float) -> float:
            r = initial * factor
            maxima = self.branch_maxima(r * c, r * s)
            if len(maxima) < 2:
                raise RuntimeError("junction continuation lost a maximum branch")
            p = maxima[branch]
            points[factor] = p
            return self._residual(p)

        value = residual(1.0)
        p = points[1.0]
        gx, gy, gz = self.field.gradient(*p)
        derivative = initial * (gx * c + (gy * self.c - gz * self.s) * s)
        if derivative >= 0:
            raise RuntimeError("junction branch is not star-shaped")
        step = max(abs(value / derivative), ulp(1.0))
        lo, hi = 1.0 - step, 1.0 + step
        while residual(lo) < 0:
            step *= 2
            lo = 1.0 - step
        while residual(hi) > 0:
            step *= 2
            hi = 1.0 + step
        factor = brent_root(
            residual,
            lo,
            hi,
            absolute_tolerance=0.0,
            relative_tolerance=LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE,
        )
        point = points.get(factor)
        if point is None:
            residual(factor)
            point = points[factor]
        return initial * factor, point

    def maximum(self, x: float, t: float) -> Point3D:
        """Return a stateless global winner and certify its part membership.

        The bounding-chord problem relaxes the watershed constraint. A winner
        certified on the star's side is therefore also the constrained global
        maximum; no interior constrained endpoint can exceed it.
        """
        point = self._bounding_maximum(x, t)
        certify_part(self.field, self.l1, self.watershed, point, contact=self.contact)
        return point

    def value(self, x: float, t: float) -> float:
        """Evaluate the relaxed global-chord field with exact-anchor offsets.

        Its zeros are accepted only after maximum() certifies part membership.
        A negative relaxed value also excludes every admissible chord point.
        """
        if x == t == 0.0:
            return float("inf")
        try:
            return self._residual(self._bounding_maximum(x, t))
        except _EmptyLineOfSightError:
            return -inf

    def _crossing(self) -> Point2D:
        def point(psi: float) -> Point3D:
            return self.watershed.rim(self.field, self.depth, psi)

        if self.s == 0.0:
            psi = 0.0
        elif self.c == 0.0:
            psi = -pi / 2
        else:

            def residual(psi: float) -> float:
                _, gy, gz = self.field.gradient(*point(psi))
                return gy * self.s + gz * self.c

            psi = brent_root(
                residual,
                -pi / 2,
                0.0,
                absolute_tolerance=0.0,
                relative_tolerance=LOS_RIM_CROSSING_ROOT_TOLERANCE,
            )
        p = point(psi)
        return (p[0], p[1] * self.c - p[2] * self.s)

    def _value_roundoff(self, x: float, t: float) -> float:
        """Bound arithmetic in the factored difference, offset and sky lift.

        Bound the magnitudes inside each factored numerator, including terms
        that cancel before fsum, rather than using the absolute potential.
        The gradient term covers rounding the lifted maximizing position.
        """
        p = self._bounding_maximum(x, t)
        a = self.l1.position[0]
        if self.contact and x < 0 and abs(self._own_offset) < abs(self.depth):
            a = self.outer_point.position[0]
        px, py, pz = p
        dx, transverse = px - a, py * py + pz * pz
        r0, r1 = hypot(*p), hypot(px - 1, py, pz)
        m, c = self.field.component_mass, self.field.companion_mass
        scale = (
            m * (abs(dx * (px + a)) + transverse) / (r0 * abs(a) * (r0 + abs(a)))
            + c
            * (abs(dx * (px + a - 2)) + transverse)
            / (r1 * abs(1 - a) * (r1 + abs(1 - a)))
            + 0.5 * (abs(dx * (px + a - 2 * c)) + py * py)
            + abs(self.field._depth_offset(self.depth, self.l1.position[0], a))
            + hypot(*self.field.gradient(*p)) * hypot(*p)
        )
        return LOS_RESIDUAL_VALUE_ULPS * float_info.epsilon * scale

    def _polish_start(self) -> float:
        """Certify the rim crossing with G, or solve on its own sky ray.

        Already-resolved crossings are retained exactly, avoiding a new root
        rounding and preserving the traced and fixed-gauge controls there.
        """
        assert self.start_point is not None
        radius = hypot(*self.start_point)
        if abs(self.value(*self.start_point)) <= self._value_roundoff(
            *self.start_point
        ):
            return radius
        radius = self._limb_radius(
            self.start,
            radius,
            self.outer * (1 - LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE),
        )
        c, s = _trig(self.start)
        self.start_point = (radius * c, radius * s)
        return radius

    def _limb_radius(self, angle: float, initial: float, cap: float) -> float:
        """Use the same bracket and G solve for a crossing and a regular ray."""
        c, s = _trig(angle)
        inner = initial * 0.9
        outer = min(cap, initial * 1.05)
        while self.value(outer * c, outer * s) > 0:
            if outer == cap:
                if self.value(outer * c, outer * s) <= self._value_roundoff(
                    outer * c, outer * s
                ):
                    return cap
                raise RuntimeError(
                    "projected limb is not bracketed inside the component"
                )
            inner, outer = outer, min(cap, outer * 1.1)
        return initial * brent_root(
            lambda factor: self.value(initial * factor * c, initial * factor * s),
            inner / initial,
            outer / initial,
            absolute_tolerance=0.0,
            relative_tolerance=LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE,
        )

    def radius(self, angle: float) -> float:
        """Intersect a sky polar ray with the component's stationary limb."""
        if angle in self._points:
            return self._points[angle][0]
        c, s = _trig(angle)
        if self._displacement_neck_ray(c, s):
            return self._neck_radius(angle)
        if (
            self.start_point is not None
            and abs(angle - self.start) <= LOS_START_ANGLE_MATCH_TOLERANCE
        ):
            assert self.start_radius is not None
            r = self.start_radius
            p = self.maximum(*self.start_point)
        elif angle == 0.0 and self.critical:
            r, p = self.l1.position[0], self.l1.position
        elif self.c == 0.0 or self.s == 0.0:
            r = self.radius_computer(
                self.field, self.depth, (c, s * self.second[1], s * self.second[2])
            )
            p = self._point(r * c, r * s, 0.0)
        else:
            initial = self.radius_computer(
                self.field, self.depth, (c, s * self.second[1], s * self.second[2])
            )
            cap = self.outer * (1 - LOS_MAXIMUM_RADIUS_ROOT_RELATIVE_TOLERANCE)
            if c > 0:
                front = self.start_point[0] if self.start_point else self.l1.position[0]
                cap = min(cap, front / c)
            r = self._limb_radius(angle, initial, cap)
            p = self.maximum(r * c, r * s)
        dv, vv = self._los(p)
        if (
            vv >= 0
            or hypot(*p) >= self.outer
            or abs(dv)
            > LOS_STATIONARITY_RELATIVE_TOLERANCE * hypot(*self.field.gradient(*p))
        ):
            raise RuntimeError("projected limb maximum is not interior")
        self._points[angle] = (r, p)
        return r

    def _displacement_neck_ray(self, c: float, s: float) -> bool:
        """Certify the local field for a contact limb, including LOS displacement.

        A limb can lose its axial tangent before a point-mass distance rounds
        to its axial value. Bound both the axial neck and transverse LOS scale
        instead. The same sixth-order expansion certificate used by radial
        rays supplies the machine-epsilon bound; there is no fitted window.
        """
        if not self.contact or not self._canonical_field or c <= 0 or s == 0:
            return False
        x1 = self.l1.position[0]
        h = self.l1.hessian
        hvv = h[1][1] * self.s**2 + h[2][2] * self.c**2
        hvt = (h[1][1] - h[2][2]) * self.s * self.c
        transverse = x1 * abs(s / c) * hypot(1.0, hvt / hvv)
        axial = sqrt(self.depth) * sqrt(2 / h[0][0])
        return (hypot(axial, transverse) / min(x1, 1 - x1)) ** 4 <= float_info.epsilon

    def _neck_radius(self, angle: float) -> float:
        """Keep the sky root and stationary point relative to L1 until use."""
        model = _neck_field(self.field, self.l1.position[0])
        c, s = _trig(angle)
        direction = (c, s * self.second[1], s * self.second[2])
        h0 = self.l1.hessian
        hvv = h0[1][1] * self.s**2 + h0[2][2] * self.c**2
        hvt = (h0[1][1] - h0[2][2]) * self.s * self.c

        def at(offset: float) -> tuple[_DD, _DD, _DD]:
            base = model.on_ray(offset, direction)
            t = fsum(base[1]) * self.c - fsum(base[2]) * self.s
            w = -hvt * t / hvv
            while True:
                p = (
                    base[0],
                    _add(base[1], (w * self.s, 0.0)),
                    _add(base[2], (w * self.c, 0.0)),
                )
                _, gy, gz = model.gradient(p)
                derivative = gy * self.s + gz * self.c
                gamma3 = 3 * float_info.epsilon / (1 - 3 * float_info.epsilon)
                if abs(derivative) <= gamma3 * (abs(gy * self.s) + abs(gz * self.c)):
                    return p
                correction = derivative / hvv
                updated = w - correction
                if updated == w:
                    return p
                # Newton's stationary residual can straddle adjacent w values.
                if abs(correction) <= ulp(w):
                    return p
                w = updated

        scale = hypot(
            sqrt(self.depth) * sqrt(2 / h0[0][0]), self.l1.position[0] * s / c
        )
        p0 = at(0.0)
        upper = model.sheet_offset(fsum(p0[1]), fsum(p0[2]))

        def value(q: float) -> float:
            return model.value(at(q * scale), self.depth)

        hi = upper / scale
        if value(hi) >= 0:
            offset = upper
        else:
            lo = hi - 1
            while value(lo) <= 0:
                lo = hi - 2 * (hi - lo)
            offset = scale * brent_root(
                value, lo, hi, absolute_tolerance=ulp(1.0), relative_tolerance=0.0
            )
        displacement = at(offset)
        r = fsum((model.x1, offset)) / c
        sky = (r * c, r * s)
        if angle == self.start and self.start_point is not None:
            sky = self.start_point
            assert self.start_radius is not None
            r = self.start_radius
        absolute = (
            fsum((model.x1, *displacement[0])),
            fsum(displacement[1]),
            fsum(displacement[2]),
        )
        # Keep the existing whole-chord isolation and global winner proof.
        # Its widened coordinate bounds cover the retained low displacement.
        winner = self._bounding_maximum(*sky)
        if (
            abs(
                winner[1] * self.s
                + winner[2] * self.c
                - (absolute[1] * self.s + absolute[2] * self.c)
            )
            > self._chord(*sky).limit
            * FIELD_CERTIFICATE_BOUND_ULPS
            * float_info.epsilon
        ):
            raise RuntimeError(
                "displacement neck is not the certified global LOS branch"
            )
        self._neck_points[sky] = displacement
        self._points[angle] = (r, absolute)
        return r

    def cone_tangent(self) -> Point2D:
        """Return the upper tangent of the Hessian's projected critical cone."""
        if self.c == 0.0 or self.s == 0.0:
            return self.field.critical_upper_tangent(
                self.l1.position[0], self.plane.inclination
            )
        h = self.l1.hessian
        vv = h[1][1] * self.s**2 + h[2][2] * self.c**2
        tt = h[1][1] * h[2][2] / vv
        slope = sqrt(h[0][0] / -tt)
        norm = hypot(1.0, slope)
        return (-1 / norm, slope / norm)

    def cone_curvature(self) -> float:
        """Use the potential's cubic term along the lifted cone generator."""
        if self.c == 0.0 or self.s == 0.0:
            return self.field.critical_branch_curvature(
                self.l1.position[0], self.plane.inclination
            )
        tx, tt = self.cone_tangent()
        h = self.l1.hessian
        vv = h[1][1] * self.s**2 + h[2][2] * self.c**2
        tv = (h[1][1] - h[2][2]) * self.s * self.c
        w = -tv * tt / vv
        ey, ez = tt * self.c + w * self.s, -tt * self.s + w * self.c
        x = self.l1.position[0]
        u_xxx, u_xyy = self.field.axial_third_derivatives(x)
        cubic = u_xxx * tx**3 + 3 * u_xyy * tx * (ey * ey + ez * ez)
        effective = h[1][1] * h[2][2] / vv
        denominator = tx * (-tt) * h[0][0] + tt * tx * effective
        return -cubic / (3 * denominator)

    def differential(
        self, point: Point2D, maximum: Point3D | None = None
    ) -> tuple[Point2D, tuple[float, float, float]]:
        """Return the envelope-theorem gradient and Schur-complement Hessian."""
        p = self.maximum(*point) if maximum is None else maximum
        displacement = self._neck_points.get(point)
        g = (
            self.field.gradient(*p)
            if displacement is None
            else _neck_field(self.field, self.l1.position[0]).gradient(displacement)
        )
        h = self.field.hessian(*p)
        hvx = h[0][1] * self.s + h[0][2] * self.c
        hvt = self.c * (h[1][1] * self.s + h[1][2] * self.c) - self.s * (
            h[2][1] * self.s + h[2][2] * self.c
        )
        hvv = self.s * (h[1][1] * self.s + h[1][2] * self.c) + self.c * (
            h[2][1] * self.s + h[2][2] * self.c
        )
        if hvv >= 0:
            raise RuntimeError("line-of-sight stationary point is not a maximum")
        gsky = (g[0], g[1] * self.c - g[2] * self.s)
        return gsky, (
            h[0][0] - hvx * hvx / hvv,
            h[0][1] * self.c - h[0][2] * self.s - hvx * hvt / hvv,
            self.c**2 * h[1][1]
            - 2 * self.c * self.s * h[1][2]
            + self.s**2 * h[2][2]
            - hvt * hvt / hvv,
        )

    def tangent(self, point: Point2D, c: float, s: float) -> Point2D:
        if self.critical and point == self.l1.position[:2]:
            return self.cone_tangent()
        (gx, gt), _ = self.differential(point)
        norm = hypot(gx, gt)
        tangent = (gt / norm, -gx / norm)
        return (
            (-tangent[0], -tangent[1])
            if -s * tangent[0] + c * tangent[1] < 0
            else tangent
        )

    def knot(self, angle: float) -> _Knot:
        r = self.radius(angle)
        c, s = _trig(angle)
        point = (
            self.start_point
            if angle == self.start and self.start_point
            else (r * c, r * s)
        )
        return self.branch_knot(angle, point, radius=r)

    def branch_knot(
        self,
        angle: float,
        point: Point2D,
        maximum: Point3D | None = None,
        *,
        radius: float | None = None,
    ) -> _Knot:
        """Evaluate a knot on a specified one-sided maximizing branch."""
        r = hypot(*point) if radius is None else radius
        c, s = _trig(angle)
        if angle == 0.0 and self.critical:
            tx, tt = self.cone_tangent()
            slope = r * tx / tt
            curvature = self.cone_curvature()
        else:
            (gx, gt), (hxx, hxt, htt) = self.differential(point, maximum)
            slope = (
                0.0
                if angle in (0.0, pi)
                else -r * (-s * gx + c * gt) / (c * gx + s * gt)
            )
            curvature = (
                -(hxx * gt * gt - 2 * hxt * gx * gt + htt * gx * gx)
                / hypot(gx, gt) ** 3
            )
        return _Knot(
            angle,
            point,
            (slope * c - r * s, slope * s + r * c),
            curvature,
            point in self._neck_points,
        )
