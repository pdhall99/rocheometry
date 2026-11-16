"""Certified stationary-point isolation on a bounded line of sight.

Bounds include every monotone piece of each point-mass term. No sampling
mesh, starting guess or previous query is used to decide which roots exist.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import inf, nextafter, sqrt
from sys import float_info

from rocheometry._errors import DegenerateLineOfSightError
from rocheometry._numerics.roots import brent_root

# Insensitive control: los stationary interval bound ulps.
LOS_STATIONARY_INTERVAL_BOUND_ULPS = 32
# Insensitive control: los stationary root relative tolerance.
LOS_STATIONARY_ROOT_RELATIVE_TOLERANCE = 1e-15

# Resolution budget: los interval epsilon.
LOS_INTERVAL_EPSILON = float_info.epsilon


def _widen(lower: float, upper: float, scale: float) -> tuple[float, float]:
    # Fewer than 16 rounded arithmetic operations per term; leave additional
    # room for libm rounding and finish with an outward nextafter.
    allowance = LOS_STATIONARY_INTERVAL_BOUND_ULPS * LOS_INTERVAL_EPSILON * scale
    return nextafter(lower - allowance, -inf), nextafter(upper + allowance, inf)


def _curvature_term_bounds(
    a_lower: float,
    a_upper: float,
    z_lower: float,
    z_upper: float,
    norm_upper: float = inf,
) -> tuple[float, float]:
    """Bound (2z-a)/(a+z)^(5/2) on a positive distance rectangle.

    No interior stationary point exists. Edge extrema occur at a=4z or
    z=3a/2, in addition to the four corners.
    """
    nodes = [(a, z) for a in (a_lower, a_upper) for z in (z_lower, z_upper)]
    nodes.extend((4 * z, z) for z in (z_lower, z_upper) if a_lower < 4 * z < a_upper)
    nodes.extend(
        (a, 1.5 * a) for a in (a_lower, a_upper) if z_lower < 1.5 * a < z_upper
    )
    # The bounding sphere also gives a+z <= norm_upper. On that diagonal
    # the denominator is constant and the numerator is linear, so only its
    # intersections with the rectangle's edges can be extrema.
    nodes.extend(
        (a, norm_upper - a)
        for a in (a_lower, a_upper)
        if z_lower <= norm_upper - a <= z_upper
    )
    nodes.extend(
        (norm_upper - z, z)
        for z in (z_lower, z_upper)
        if a_lower <= norm_upper - z <= a_upper
    )
    nodes = [(a, z) for a, z in nodes if a + z <= nextafter(norm_upper, inf)]
    if not nodes:
        return -inf, inf
    values = [(2 * z - a) / (a + z) ** 2.5 for a, z in nodes]
    scale = max((2 * z + a) / (a + z) ** 2.5 for a, z in nodes)
    return _widen(min(values), max(values), scale)


@dataclass(frozen=True, slots=True)
class _Chord:
    a: float
    b: float
    m: float
    mu: float
    e: float
    s2: float
    linear: float
    limit: float
    mass_error: float = 0.0

    def derivative(self, w: float) -> float:
        z = self.e * w * w
        value = self.linear + w * (
            self.s2
            - self.e * (self.m / (self.a + z) ** 1.5 + self.mu / (self.b + z) ** 1.5)
        )
        return (
            value + self.mass_error * (self.linear + w * self.s2)
            if self.mass_error
            else value
        )

    def curvature(self, w: float) -> float:
        z = self.e * w * w
        value = self.s2 + self.e * (
            self.m * (2 * z - self.a) / (self.a + z) ** 2.5
            + self.mu * (2 * z - self.b) / (self.b + z) ** 2.5
        )
        return value + self.mass_error * self.s2 if self.mass_error else value

    def _term_bounds(
        self, a: float, left: float, right: float, order: int
    ) -> tuple[float, float, float]:
        # f has extrema at +/-sqrt(a/(2e)); f' at +/-sqrt(3a/(2e)).
        critical = sqrt((1 if order == 0 else 3) * a / (2 * self.e))
        if not (
            left < -critical < right or left < 0.0 < right or left < critical < right
        ):
            # Same endpoint expressions and min/max as the general path;
            # avoid constructing tuples, generators and lists on a single
            # monotone piece with the same cancellation scale.
            zl, zr = self.e * left * left, self.e * right * right
            if order == 0:
                vl = left / (a + zl) ** 1.5
                vr = right / (a + zr) ** 1.5
                return min(vl, vr), max(vl, vr), max(abs(vl), abs(vr))
            dl, dr = (a + zl) ** 2.5, (a + zr) ** 2.5
            vl, vr = (a - 2 * zl) / dl, (a - 2 * zr) / dr
            return min(vl, vr), max(vl, vr), max((a + 2 * zl) / dl, (a + 2 * zr) / dr)
        nodes = (
            left,
            right,
            *(v for v in (-critical, 0.0, critical) if left < v < right),
        )
        if order == 0:
            values = tuple(w / (a + self.e * w * w) ** 1.5 for w in nodes)
            scale = max(abs(v) for v in values)
        else:
            values_list = []
            scale = 0.0
            for w in nodes:
                z = self.e * w * w
                denominator = (a + z) ** 2.5
                values_list.append((a - 2 * z) / denominator)
                # The numerator can cancel. Its unfactored magnitude, not
                # |f'|, bounds rounding at the zero of f'. This scale decreases
                # with |w|, so its largest value is among these same nodes.
                scale = max(scale, (a + 2 * z) / denominator)
            values = tuple(values_list)
        return min(values), max(values), scale

    def bounds(self, left: float, right: float, order: int) -> tuple[float, float]:
        if order == 0:
            lower, upper = self.linear + self.s2 * left, self.linear + self.s2 * right
            scale = abs(self.linear) + self.s2 * max(abs(left), abs(right))
        else:
            lower = upper = scale = self.s2
        if self.mass_error:
            if order == 0:
                lower += self.mass_error * (self.linear + self.s2 * left)
                upper += self.mass_error * (self.linear + self.s2 * right)
            else:
                lower += self.mass_error * self.s2
                upper += self.mass_error * self.s2
            scale += abs(self.mass_error) * (
                abs(self.linear) + self.s2 * max(1.0, abs(left), abs(right))
            )
        for a, mass in ((self.a, self.m), (self.b, self.mu)):
            if mass == 0.0:
                continue
            lo, hi, term_scale = self._term_bounds(a, left, right, order)
            coefficient = self.e * mass
            lower -= coefficient * hi
            upper -= coefficient * lo
            scale += coefficient * term_scale
        return _widen(lower, upper, scale)

    def _uncertain_zero(self, w: float, value: float) -> bool:
        # The mass terms have the same sign; recover their total magnitude
        # without another inverse-distance evaluation. Never discard a zero
        # that can have rounded onto the boundary of a monotone interval.
        scale = (
            abs(self.linear) + abs(w) * self.s2 + abs(self.linear + w * self.s2 - value)
        )
        return (
            abs(value)
            <= LOS_STATIONARY_INTERVAL_BOUND_ULPS * LOS_INTERVAL_EPSILON * scale
        )

    def root(self, left: float, right: float) -> float:
        """Polish only an already-certified single-root bracket."""
        fl, fr = self.derivative(left), self.derivative(right)
        w = left - fl * (right - left) / (fr - fl)
        for _ in range(64):
            value = self.derivative(w)
            if value == 0:
                return w
            if (value > 0) == (fl > 0):
                left, fl = w, value
            else:
                right, fr = w, value
            candidate = w - value / self.curvature(w)
            if candidate == w or nextafter(candidate, w) == w:
                return min((w, candidate), key=lambda v: abs(self.derivative(v)))
            if not left < candidate < right:
                candidate = left + (right - left) / 2
            if candidate in (left, right):
                return min((left, right), key=lambda v: abs(self.derivative(v)))
            w = candidate
        scale = max(abs(left), abs(right))
        return scale * brent_root(
            lambda v: self.derivative(v * scale),
            left / scale,
            right / scale,
            absolute_tolerance=0.0,
            relative_tolerance=LOS_STATIONARY_ROOT_RELATIVE_TOLERANCE,
        )

    def level_upper(
        self,
        x: float,
        t: float,
        c: float,
        s: float,
        anchor: float,
        offset: float,
        left: float,
        right: float,
    ) -> float:
        """Bound the anchored U-u from above, rounding outward.

        Each inverse distance is largest at the smallest |w|. The quadratic
        is convex and therefore largest at an endpoint. Factored numerators
        retain the saddle-relative scale; the allowance also covers sky lift.
        """
        w = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
        transverse = self.e * (t * t + w * w)
        r0, r1 = sqrt(x * x + transverse), sqrt((x - 1) ** 2 + transverse)
        dx = x - anchor
        n0 = -dx * (x + anchor) - transverse
        n1 = -dx * (x + anchor - 2) - transverse
        q0 = dx * (x + anchor - 2 * self.mu)
        q1 = max((t * c + left * s) ** 2, (t * c + right * s) ** 2)
        first = self.m * n0 / (r0 * abs(anchor) * (r0 + abs(anchor)))
        second = self.mu * n1 / (r1 * abs(1 - anchor) * (r1 + abs(1 - anchor)))
        upper = first + second + 0.5 * (q0 + q1) + offset
        if self.mass_error:
            upper += self.mass_error / 2 * (dx * (x + anchor) + q1)
        scale = (
            self.m / r0
            + self.mu / r1
            + abs(first)
            + abs(second)
            + abs(q0)
            + q1
            + abs(offset)
        )
        return nextafter(
            upper + LOS_STATIONARY_INTERVAL_BOUND_ULPS * LOS_INTERVAL_EPSILON * scale,
            inf,
        )

    def stationary(
        self,
        left: float | None = None,
        right: float | None = None,
        *,
        attained_lower: Callable[[float], float] | None = None,
        level_upper: Callable[[float, float], float] | None = None,
    ) -> tuple[float, ...]:
        """Isolate every stationary branch not excluded by a level bound.

        A D bound excluding zero proves absence; a D' bound excluding zero
        proves uniqueness. Boundary values within their rounding enclosure
        are retained, never discarded by a rounded sign. Such candidates are
        merged only when monotonicity proves they represent at most one root.
        All non-monotone intervals are bisected, including close max/min pairs.
        """
        initial = (
            -self.limit if left is None else left,
            self.limit if right is None else right,
        )
        initial_curvature = self.bounds(*initial, 1)
        if initial_curvature[1] < 0:
            # Strict concavity certifies a unique maximum on the entire
            # retained chord. These are exactly the baseline isolating ends;
            # neither a hint nor a tighter bracket changes root rounding.
            fl, fr = self.derivative(initial[0]), self.derivative(initial[1])
            if fl > 0 > fr:
                return (self.root(*initial),)
        pending = [initial]
        roots = []
        best: float | None = None
        evaluated = 0
        while pending:
            left, right = pending.pop()
            lower, upper = self.bounds(left, right, 0)
            if lower > 0 or upper < 0:
                continue
            dl, du = (
                initial_curvature
                if (left, right) == initial
                else self.bounds(left, right, 1)
            )
            if dl <= 0 <= du:
                middle = left + (right - left) / 2
                value = self.derivative(middle)
                scale = (
                    abs(self.linear)
                    + abs(middle) * self.s2
                    + abs(self.linear + middle * self.s2 - value)
                )
                delta = max(abs(dl), abs(du)) * max(middle - left, right - middle)
                ml, mu = _widen(value - delta, value + delta, scale + delta)
                if max(lower, ml) > 0 or min(upper, mu) < 0:
                    continue
            fl, fr = self.derivative(left), self.derivative(right)
            if dl > 0 or du < 0:
                if fl == 0:
                    roots.append(left)
                if fr == 0:
                    roots.append(right)
                if (fl > 0) != (fr > 0) and fl != 0 and fr != 0:
                    roots.append(self.root(left, right))
                else:
                    if fl != 0 and self._uncertain_zero(left, fl):
                        roots.append(left)
                    if fr != 0 and self._uncertain_zero(right, fr):
                        roots.append(right)
                continue
            # A negative upper bound excludes the limb. Additionally require
            # it to be below an attained value, preserving the actual global
            # maximum even when the whole chord lies outside the silhouette.
            if level_upper:
                upper_level = level_upper(left, right)
                if upper_level < 0.0 and attained_lower:
                    # The original pruning rule cannot fire for U-u >= 0.
                    # Evaluate precisely its same lower bound only when a
                    # negative upper bound actually needs the comparison.
                    if best is None:
                        best = max(
                            attained_lower(-self.limit), attained_lower(self.limit)
                        )
                    for w in roots[evaluated:]:
                        best = max(best, attained_lower(w))
                    evaluated = len(roots)
                    if upper_level < min(0.0, best):
                        continue
            middle = left + (right - left) / 2
            if middle in (left, right):
                raise DegenerateLineOfSightError(
                    f"degenerate line-of-sight stationary pair in [{left!r}, {right!r}]"
                )
            pending.append((middle, right))
            pending.append((left, middle))
        isolated: list[float] = []
        for w in sorted(set(roots)):
            if isolated:
                lo, hi = self.bounds(isolated[-1], w, 1)
                if lo > 0 or hi < 0:
                    # A monotone D cannot have two distinct zeros. Boundary
                    # candidates here represent one root at the arithmetic
                    # floor, not two maximizing branches.
                    isolated[-1] = min(
                        (isolated[-1], w), key=lambda v: (abs(self.derivative(v)), v)
                    )
                    continue
            isolated.append(w)
        return tuple(isolated)
