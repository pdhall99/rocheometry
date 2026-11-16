"""Adaptive gravity lines with dense output for the L1 watershed."""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from dataclasses import field as dataclass_field
from functools import partial
from math import atan2, cos, exp, fsum, hypot, log, nextafter, pi, sin, sqrt
from typing import ClassVar, cast

from rocheometry._computers.composition import Memoized
from rocheometry._computers.contact import _geometry_depth, coverage_depth
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.neck import (
    _neck_field,
    _neck_sheet_point,
    _unresolved_neck_position,
)
from rocheometry._computers.watershed.family import (
    _BuildCutSection,
    _BuildFamily,
    _BuildSection,
    _CutSection,
    _FamilyRim,
    _LineFamily,
    _plane,
    _Section,
)
from rocheometry._geometry import Point3D
from rocheometry._numerics.roots import _adjacent_float_root, brent_root
from rocheometry._roche.field import RocheField
from rocheometry._validation import _normalize_finite_positive_float, require_int

# Fixed coverage and initialization controls, tested by release perturbations.
# Work guard or cache capacity: watershed rim depth coverage factor.
WATERSHED_RIM_DEPTH_COVERAGE_FACTOR = 1.05
# Work guard or cache capacity: watershed family depth coverage factor.
WATERSHED_FAMILY_DEPTH_COVERAGE_FACTOR = 2.0
# Insensitive control: watershed seed depth factor.
WATERSHED_SEED_DEPTH_FACTOR = 1e-12
# Insensitive control: watershed initial log step.
WATERSHED_INITIAL_LOG_STEP = 0.1
# Hairer, Nørsett & Wanner, Solving ODEs I, II.4; Hairer's lecture notes,
# III.4, (4.6): https://unige.ch/~hairer/poly/poly.pdf (printed p. 66).
# Work guard or cache capacity: watershed rk safety factor.
WATERSHED_RK_SAFETY_FACTOR = 0.9
# Insensitive control: watershed rk min step factor.
WATERSHED_RK_MIN_STEP_FACTOR = 0.2
# Insensitive control: watershed rk max step factor.
WATERSHED_RK_MAX_STEP_FACTOR = 5.0
# Insensitive control: watershed rim radius root tolerance.
WATERSHED_RIM_RADIUS_ROOT_TOLERANCE = 8e-15
# Insensitive control: watershed trig cleanup tolerance.
WATERSHED_TRIG_CLEANUP_TOLERANCE = 1e-15
# Insensitive control: watershed azimuth match tolerance.
WATERSHED_AZIMUTH_MATCH_TOLERANCE = 1e-14
# Insensitive control: watershed depth root relative tolerance.
WATERSHED_DEPTH_ROOT_RELATIVE_TOLERANCE = 2e-14
# Insensitive control: watershed rim stationarity tolerance.
WATERSHED_RIM_STATIONARITY_TOLERANCE = 2e-13

# Dormand-Prince 5(4), with its quartic continuous extension.
# Derived: dormand prince stage coefficients.
DORMAND_PRINCE_STAGE_COEFFICIENTS = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
# Derived: dormand prince stage times.
DORMAND_PRINCE_STAGE_TIMES = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
# Derived: dormand prince error coefficients.
DORMAND_PRINCE_ERROR_COEFFICIENTS = (
    71 / 57600,
    0.0,
    -71 / 16695,
    71 / 1920,
    -17253 / 339200,
    22 / 525,
    -1 / 40,
)
# Derived: dormand prince dense coefficients.
DORMAND_PRINCE_DENSE_COEFFICIENTS = (
    (
        1.0,
        -8048581381 / 2820520608,
        8663915743 / 2820520608,
        -12715105075 / 11282082432,
    ),
    (0.0, 0.0, 0.0, 0.0),
    (
        0.0,
        131558114200 / 32700410799,
        -68118460800 / 10900136933,
        87487479700 / 32700410799,
    ),
    (0.0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072),
    (
        0.0,
        127303824393 / 49829197408,
        -318862633887 / 49829197408,
        701980252875 / 199316789632,
    ),
    (0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844),
    (0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423),
)


def _velocity(field: RocheField, p: Point3D, depth: float) -> Point3D:
    g = field.gradient(*p)
    factor = -depth / fsum(v * v for v in g)
    return (factor * g[0], factor * g[1], factor * g[2])


def _sheet_seed(field: RocheField, x1: float, depth: float, alpha: float) -> Point3D:
    """Use the invariant sheet's quadratic height near the L1 saddle.

    For X=x1+A*y*y+B*z*z, tangency of gravity gives
    A=U_xyy/(2*(2*U_yy-U_xx)) and likewise for B. The flat seed is
    insufficient when a requested rim is shallower than the integration seed:
    its radial tangency can lie on the wrong side of the true watershed.
    """
    h = field.hessian(x1)
    c, t = cos(alpha), sin(alpha)
    if alpha == 0:
        c, t = 1.0, 0.0
    elif alpha == pi / 2:
        c, t = 0.0, 1.0
    r = sqrt(2 * depth / (-h[1][1] * c * c - h[2][2] * t * t))
    y, z = r * c, r * t
    _, u_xyy = field.axial_third_derivatives(x1)
    x = x1 + u_xyy / 2 * (
        y * y / (2 * h[1][1] - h[0][0]) + z * z / (2 * h[2][2] - h[0][0])
    )
    return (x, y, z)


@dataclass(frozen=True, slots=True)
class _DenseSteps:
    """Flat rows of width, initial coordinates and twelve quartic coefficients."""

    data: array[float]

    def __len__(self) -> int:
        return len(self.data) // 16

    def value(self, index: int, start: float, s: float) -> Point3D:
        offset = 16 * index
        q = self.data
        width = q[offset]
        t = (s - start) / width
        return cast(
            Point3D,
            tuple(
                q[offset + 1 + j]
                + width
                * t
                * (
                    q[offset + 4 + j]
                    + t
                    * (
                        q[offset + 7 + j]
                        + t * (q[offset + 10 + j] + t * q[offset + 13 + j])
                    )
                )
                for j in range(3)
            ),
        )


@dataclass(frozen=True, slots=True)
class WatershedLine:
    """One downhill gravity line, interpolated in log potential depth.

    Attributes:
        field: Canonical component field used to integrate the line.
        alpha: Starting angle in the transverse plane, in radians.
        x_l1: L1's axial coordinate in the integration frame.
        u_l1: Canonical potential of L1.
        log_depths: Accepted integrator nodes, from the quadratic seed outward.
        steps: Flat double-array quartic extensions of the accepted steps.
        mirrored: Whether returned x coordinates use the reciprocal frame.
    """

    field: RocheField
    alpha: float
    x_l1: float
    u_l1: float
    log_depths: array[float]
    steps: _DenseSteps
    mirrored: bool = False

    def at_log_depth(self, s: float) -> Point3D:
        """Interpolate and polish at depth exp(s), preserving the sheet."""
        depth = exp(s)
        if s < self.log_depths[0]:
            p = _sheet_seed(self.field, self.x_l1, depth, self.alpha)
        else:
            index = min(
                len(self.steps) - 1, max(0, bisect_right(self.log_depths, s) - 1)
            )
            q = self.steps.value(index, self.log_depths[index], s)
            p = (self.x_l1 * (1 + q[0]), self.x_l1 * q[1], self.x_l1 * q[2])
        for _ in range(2):
            g = self.field.gradient(*p)
            correction = -(
                self.field.potential_difference(p, (self.x_l1, 0.0, 0.0)) + depth
            ) / fsum(v * v for v in g)
            p = (
                p[0] + correction * g[0],
                p[1] + correction * g[1],
                p[2] + correction * g[2],
            )
        return (1 - p[0], p[1], p[2]) if self.mirrored else p

    def point(self, depth: float) -> Point3D:
        """Return the point at depth, with L1 as the exact zero endpoint."""
        if depth <= 0:
            return (1 - self.x_l1 if self.mirrored else self.x_l1, 0.0, 0.0)
        return self.at_log_depth(log(depth))

    def at_radius(self, rho: float) -> Point3D:
        """Invert the monotone transverse distance on this line's dense output."""
        if rho <= 0:
            return (1 - self.x_l1 if self.mirrored else self.x_l1, 0.0, 0.0)
        count = len(self.steps)
        data = self.steps.data
        last = 16 * (count - 1)
        if rho >= self.x_l1 * hypot(data[last + 2], data[last + 3]):
            end = self.at_log_depth(self.log_depths[-1])
            if rho > hypot(end[1], end[2]):
                raise ValueError("transverse radius lies beyond the traced watershed")
        lower_index, upper_index = 0, count
        while lower_index < upper_index:
            middle = (lower_index + upper_index) // 2
            radius = self.x_l1 * hypot(data[16 * middle + 2], data[16 * middle + 3])
            if radius < rho:
                lower_index = middle + 1
            else:
                upper_index = middle
        i = lower_index
        if i == 0:
            h = self.field.hessian(self.x_l1)
            stiffness = -h[1][1] * cos(self.alpha) ** 2 - h[2][2] * sin(self.alpha) ** 2
            s = 2 * log(rho) + log(stiffness / 2)
            lower, upper = s - 2, self.log_depths[0]
        else:
            lower = self.log_depths[max(0, i - 2)]
            upper = self.log_depths[min(count, i + 1)]
            s = (lower + upper) / 2
        for _ in range(40):
            point = self.at_log_depth(s)
            radius = hypot(point[1], point[2])
            error = radius - rho
            if abs(error) <= WATERSHED_RIM_RADIUS_ROOT_TOLERANCE * max(
                rho, 1e-6 * self.x_l1
            ):
                return point
            if error < 0:
                lower = s
            else:
                upper = s
            if nextafter(lower, upper) == upper:
                p_lower, p_upper = self.at_log_depth(lower), self.at_log_depth(upper)
                chosen = _adjacent_float_root(
                    lower,
                    upper,
                    hypot(p_lower[1], p_lower[2]) - rho,
                    hypot(p_upper[1], p_upper[2]) - rho,
                    WATERSHED_RIM_RADIUS_ROOT_TOLERANCE * max(rho, 1e-6 * self.x_l1),
                )
                if chosen is not None:
                    return p_lower if chosen == lower else p_upper
                break
            canonical = (1 - point[0], point[1], point[2]) if self.mirrored else point
            v = _velocity(self.field, canonical, exp(s))
            derivative = (point[1] * v[1] + point[2] * v[2]) / radius
            candidate = s - error / derivative
            s = candidate if lower < candidate < upper else (lower + upper) / 2
        raise ValueError("transverse radius lies beyond the traced watershed")


@dataclass(frozen=True, slots=True)
class _TraceLine:
    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    tolerance: float
    depth_factor: float = WATERSHED_RIM_DEPTH_COVERAGE_FACTOR

    def __call__(self, field: RocheField, alpha: float, /) -> WatershedLine:
        l1 = self.l1_computer(field)
        x1 = l1.position[0]
        u1 = l1.potential.value
        gap = coverage_depth(field, l1, self.outer_computer)
        start = log(WATERSHED_SEED_DEPTH_FACTOR * abs(u1))
        stop = log(self.depth_factor * gap)
        checkpoint = log(WATERSHED_RIM_DEPTH_COVERAGE_FACTOR * gap)
        seed = _sheet_seed(field, x1, exp(start), alpha)
        state = ((seed[0] - x1) / x1, seed[1] / x1, seed[2] / x1)

        def derivative(s: float, q: Point3D) -> Point3D:
            p = (x1 * (1 + q[0]), x1 * q[1], x1 * q[2])
            v = _velocity(field, p, exp(s))
            return (v[0] / x1, v[1] / x1, v[2] / x1)

        s = start
        step = WATERSHED_INITIAL_LOG_STEP
        logs = array("d", [s])
        steps = array("d")
        for _ in range(20000):
            if s >= stop:
                break
            step = min(step, stop - s)
            if s < checkpoint:
                step = min(step, checkpoint - s)
            stages = []
            for a, c0 in zip(
                DORMAND_PRINCE_STAGE_COEFFICIENTS,
                DORMAND_PRINCE_STAGE_TIMES,
                strict=True,
            ):
                q = tuple(
                    state[j] + step * fsum(a[i] * stages[i][j] for i in range(len(a)))
                    for j in range(3)
                )
                stages.append(derivative(s + step * c0, q))  # type: ignore[arg-type]
            candidate = tuple(
                state[j]
                + step
                * fsum(
                    DORMAND_PRINCE_STAGE_COEFFICIENTS[-1][i] * stages[i][j]
                    for i in range(6)
                )
                for j in range(3)
            )
            error = max(
                abs(
                    step
                    * fsum(
                        DORMAND_PRINCE_ERROR_COEFFICIENTS[i] * stages[i][j]
                        for i in range(7)
                    )
                )
                / (self.tolerance * (1 + abs(candidate[j])))
                for j in range(3)
            )
            if error <= 1:
                steps.extend((step, *state))
                steps.extend(
                    fsum(
                        stages[i][j] * DORMAND_PRINCE_DENSE_COEFFICIENTS[i][k]
                        for i in range(7)
                    )
                    for k in range(4)
                    for j in range(3)
                )
                s += step
                state = candidate  # type: ignore[assignment]
                logs.append(s)
            step *= (
                min(
                    WATERSHED_RK_MAX_STEP_FACTOR,
                    max(
                        WATERSHED_RK_MIN_STEP_FACTOR,
                        WATERSHED_RK_SAFETY_FACTOR * error ** (-0.2),
                    ),
                )
                if error
                else WATERSHED_RK_MAX_STEP_FACTOR
            )
        else:
            raise RuntimeError("watershed gravity-line integration did not converge")
        return WatershedLine(field, alpha, x1, u1, logs, _DenseSteps(steps))


@dataclass(frozen=True, slots=True)
class _FrameLine:
    canonical: Memoized[[RocheField, float], WatershedLine]

    def __call__(self, field: RocheField, alpha: float, /) -> WatershedLine:
        alpha = float(alpha) % (2 * pi)
        if type(field) is RocheField and field.qi > 1:
            return replace(self.canonical(field.reciprocal(), alpha), mirrored=True)
        return self.canonical(field, alpha)


@dataclass(frozen=True, slots=True)
class _RimAtDepth:
    line: Callable[[RocheField, float], WatershedLine]
    l1_computer: CriticalPointComputer

    def __call__(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        """Match azimuth at continuous depth, without rounding through a level."""
        depth = _geometry_depth(depth, self.l1_computer(field))
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth == 0:
            return self.l1_computer(field).position
        cy, sz = cos(psi), sin(psi)
        target = atan2(abs(sz), abs(cy))
        s = log(depth)

        def point(alpha: float) -> Point3D:
            return self.line(field, alpha).at_log_depth(s)

        if target < WATERSHED_TRIG_CLEANUP_TOLERANCE:
            alpha = 0.0
        elif pi / 2 - target < WATERSHED_TRIG_CLEANUP_TOLERANCE:
            alpha = pi / 2
        else:
            alpha, previous = target, max(0.0, target - 1e-3)
            p = point(previous)
            old = atan2(p[2], p[1]) - target
            for _ in range(30):
                p = point(alpha)
                value = atan2(p[2], p[1]) - target
                if abs(value) <= WATERSHED_AZIMUTH_MATCH_TOLERANCE:
                    break
                candidate = (
                    alpha - value * (alpha - previous) / (value - old)
                    if value != old
                    else (alpha + target) / 2
                )
                previous, old = alpha, value
                alpha = min(pi / 2, max(0.0, candidate))
            else:
                raise RuntimeError("watershed rim azimuth solve did not converge")
        p = point(alpha)
        return (p[0], p[1] if cy >= 0 else -p[1], p[2] if sz >= 0 else -p[2])


@dataclass(frozen=True, slots=True)
class _Rim:
    l1_computer: CriticalPointComputer
    rim_at_depth: Callable[[RocheField, float, float], Point3D]
    outer_computer: CriticalPointComputer

    def __call__(self, field: RocheField, depth: float, psi: float) -> Point3D:
        l1 = self.l1_computer(field)
        depth = _geometry_depth(depth, l1)
        if depth == 0:
            return l1.position
        return self.rim_at_depth(field, depth, psi)


@dataclass(frozen=True, slots=True)
class _CutHeight:
    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    rim_at_depth: Callable[[RocheField, float, float], Point3D]

    def __call__(self, field: RocheField, rho: float, psi: float, /) -> float:
        """Invert the monotone transverse radius using a bracket in square-root depth.

        The bracket extends to the traced lines' terminal depth, just beyond L2.
        At zero transverse radius the height is exactly the L1 position.
        """
        if rho < 0:
            raise ValueError("rho must be non-negative")
        l1 = self.l1_computer(field)
        if rho == 0 or (type(field) is RocheField and field.qi == 1.0):
            return l1.position[0]
        gap = coverage_depth(field, l1, self.outer_computer)
        sigma_max = sqrt(WATERSHED_RIM_DEPTH_COVERAGE_FACTOR * gap)

        def point(scaled: float) -> Point3D:
            return (
                l1.position
                if scaled == 0
                else self.rim_at_depth(field, (scaled * sigma_max) ** 2, psi)
            )

        root = brent_root(
            lambda t: hypot(*point(t)[1:]) - rho,
            0.0,
            1.0,
            absolute_tolerance=WATERSHED_DEPTH_ROOT_RELATIVE_TOLERANCE,
            relative_tolerance=WATERSHED_DEPTH_ROOT_RELATIVE_TOLERANCE,
        )
        return point(root)[0]


@dataclass(frozen=True, slots=True)
class _SheetPointOnRay:
    line: Callable[[RocheField, float], WatershedLine]
    l1_computer: CriticalPointComputer

    def __call__(self, field: RocheField, direction: Point3D, /) -> Point3D:
        """Solve two transverse ray residuals by Newton in (alpha, log depth)."""
        dx, dy, dz = direction
        if dx <= 0:
            raise ValueError("a watershed ray must point towards the companion")
        l1 = self.l1_computer(field)
        x1 = l1.position[0]
        if dy == 0 and dz == 0:
            return l1.position
        ky, kz = dy / dx, dz / dx
        alpha = atan2(dz, dy)
        h = field.hessian(x1)
        depth = max(1e-300, -0.5 * x1 * x1 * (h[1][1] * ky * ky + h[2][2] * kz * kz))
        s = log(depth)

        def residual(a: float, t: float) -> tuple[Point3D, tuple[float, float]]:
            p = self.line(field, a).at_log_depth(t)
            return p, (p[1] - ky * p[0], p[2] - kz * p[0])

        for _ in range(30):
            p, f = residual(alpha, s)
            norm = hypot(*f)
            if norm <= WATERSHED_RIM_STATIONARITY_TOLERANCE * max(x1, hypot(*p)):
                return p
            delta = 1e-5
            _, fplus = residual(alpha + delta, s)
            _, fminus = residual(alpha - delta, s)
            ja = (
                (fplus[0] - fminus[0]) / (2 * delta),
                (fplus[1] - fminus[1]) / (2 * delta),
            )
            v = _velocity(field, p, exp(s))
            js = (v[1] - ky * v[0], v[2] - kz * v[0])
            determinant = ja[0] * js[1] - ja[1] * js[0]
            da = (-f[0] * js[1] + f[1] * js[0]) / determinant
            ds = (-ja[0] * f[1] + ja[1] * f[0]) / determinant
            factor = min(1.0, 0.5 / max(abs(da), 1e-300), 2 / max(abs(ds), 1e-300))
            for _ in range(12):
                _, candidate = residual(alpha + factor * da, s + factor * ds)
                if hypot(*candidate) < norm:
                    break
                factor *= 0.5
            alpha += factor * da
            s += factor * ds
        raise RuntimeError("watershed ray intersection did not converge")


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class _ExactWatershedComputer:
    """Keep the original secant and finite-difference queries as a test oracle.

    Memoized nodes provide thread-safe cache bookkeeping. Concurrent misses may
    compute the same value, as for every graph `Memoized` node. Construction
    performs no solve; pickling preserves configuration and clears all results.

    Attributes:
        l1_computer: Critical point defining the seed and canonical level.
        outer_computer: Both outer points define the integration depth.
        tolerance: Error target in coordinates scaled by the L1 distance.
        maxsize: Capacity of each Memoized node; None is unbounded, zero disables it.
        line: Implementation-specific memoized gravity-line computer.
        depth_computer: Memoized point at an unrounded depth below L1.
        rim_computer: Memoized point at a canonical level and transverse azimuth.
        cut_height_computer: Axial-height producer using the shared depth computer.
        ray_computer: Ray-intersection producer using the shared line computer.
    """

    _line_depth_factor: ClassVar[float] = WATERSHED_RIM_DEPTH_COVERAGE_FACTOR

    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    tolerance: float = 1e-13
    maxsize: int | None = 128
    line: Memoized[[RocheField, float], WatershedLine] = dataclass_field(init=False)
    depth_computer: Memoized[[RocheField, float, float], Point3D] = dataclass_field(
        init=False
    )
    rim_computer: Memoized[[RocheField, float, float], Point3D] = dataclass_field(
        init=False
    )
    cut_height_computer: Callable[[RocheField, float, float], float] = dataclass_field(
        init=False
    )
    ray_computer: Callable[[RocheField, Point3D], Point3D] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tolerance",
            _normalize_finite_positive_float(self.tolerance, name="tolerance"),
        )

        canonical = Memoized(
            _TraceLine(
                self.l1_computer,
                self.outer_computer,
                self.tolerance,
                self._line_depth_factor,
            ),
            maxsize=self.maxsize,
        )
        line = Memoized(_FrameLine(canonical), maxsize=self.maxsize)
        object.__setattr__(self, "line", line)
        self._build_nodes(line)

    def _build_nodes(self, line: Memoized[[RocheField, float], WatershedLine]) -> None:
        depth = Memoized(_RimAtDepth(line, self.l1_computer), maxsize=self.maxsize)
        object.__setattr__(self, "depth_computer", depth)
        object.__setattr__(
            self,
            "rim_computer",
            Memoized(
                _Rim(self.l1_computer, depth, self.outer_computer), maxsize=self.maxsize
            ),
        )
        object.__setattr__(
            self,
            "cut_height_computer",
            _CutHeight(self.l1_computer, self.outer_computer, depth),
        )
        object.__setattr__(
            self, "ray_computer", _SheetPointOnRay(line, self.l1_computer)
        )

    def rim_at_depth(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        """Return the rim at continuous depth without rounding through potential."""
        return self.depth_computer(field, depth, psi)

    def rim(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        """Return the rim at the requested canonical level and azimuth."""
        return self.rim_computer(field, depth, psi)

    def cut_height(self, field: RocheField, rho: float, psi: float, /) -> float:
        """Return the sheet's axial height at transverse coordinates."""
        return self.cut_height_computer(field, rho, psi)

    def sheet_point_on_ray(self, field: RocheField, direction: Point3D, /) -> Point3D:
        """Return the sheet intersection of a component-centred forward ray."""
        return self.ray_computer(field, direction)

    def __call__(self, field: RocheField, alpha: float, /) -> WatershedLine:
        """Read a diagnostic line traced to this class's L2-depth factor."""
        return self.line(field, float(alpha) % (2 * pi))

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild the configured nodes with empty caches."""
        return (
            partial(
                type(self),
                **{
                    item.name: getattr(self, item.name)
                    for item in fields(self)
                    if item.init
                },
            ),
            (),
        )


@dataclass(frozen=True, slots=True)
class _FamilyCutHeight:
    l1_computer: CriticalPointComputer
    outer_computer: CriticalPointComputer
    section: Callable[[RocheField, float], _Section]
    cut_section: Callable[[RocheField, float], _CutSection]
    tolerance: float

    def __call__(self, field: RocheField, rho: float, psi: float, /) -> float:
        if rho < 0:
            raise ValueError("rho must be non-negative")
        l1 = self.l1_computer(field)
        if rho == 0 or (type(field) is RocheField and field.qi == 1.0):
            return l1.position[0]
        cy, sz = _plane(psi)
        target = atan2(abs(sz), abs(cy))
        end = self.outer_section(field).evaluate(target)
        if (
            rho
            > end[1]
            + max(WATERSHED_DEPTH_ROOT_RELATIVE_TOLERANCE, self.tolerance / 10)
            * l1.position[0]
        ):
            raise ValueError("transverse radius lies beyond the traced watershed")
        if _unresolved_neck_position(l1.position[0], rho):
            return fsum(
                (
                    l1.position[0],
                    _neck_field(field, l1.position[0]).sheet_offset(rho * cy, rho * sz),
                )
            )
        return self.cut_section(field, rho).evaluate(target)[0]

    def outer_section(self, field: RocheField) -> _Section:
        """Return the common outer limit of public cut-height queries."""
        l1 = self.l1_computer(field)
        gap = coverage_depth(field, l1, self.outer_computer)
        maximum = WATERSHED_RIM_DEPTH_COVERAGE_FACTOR * gap
        return self.section(field, maximum)


@dataclass(frozen=True, slots=True)
class _FamilySheetPoint:
    l1_computer: CriticalPointComputer
    cut_section: _BuildCutSection

    def __call__(self, field: RocheField, direction: Point3D, /) -> Point3D:
        dx, dy, dz = direction
        if dx <= 0:
            raise ValueError("a watershed ray must point towards the companion")
        l1 = self.l1_computer(field)
        transverse = hypot(dy, dz)
        if transverse == 0:
            return l1.position
        target = atan2(abs(dz), abs(dy))
        k = transverse / dx
        rho = k * l1.position[0]
        if _unresolved_neck_position(l1.position[0], rho):
            return _neck_sheet_point(field, l1.position[0], direction)
        if type(field) is RocheField and field.qi == 1.0:
            return l1.position[0], rho * dy / transverse, rho * dz / transverse
        for _ in range(24):
            x, xr, _ = self.cut_section(field, rho).evaluate(target)
            residual = rho - k * x
            if abs(residual) <= WATERSHED_RIM_RADIUS_ROOT_TOLERANCE * max(
                l1.position[0], hypot(x, rho)
            ):
                return x, rho * dy / transverse, rho * dz / transverse
            candidate = rho - residual / (1 - k * xr)
            rho = candidate if candidate > 0 else rho / 2
        raise RuntimeError("watershed family ray intersection did not converge")


# Resolution budget: watershed azimuth line count.
WATERSHED_AZIMUTH_LINE_COUNT = 25
"""Fixed line count in each of the two quadrant sets; endpoints are shared."""

# Resolution budget: watershed cosine series terms.
WATERSHED_COSINE_SERIES_TERMS = 17
"""Number of cosine terms in each reached-azimuth fit."""


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TracedWatershedComputer(_ExactWatershedComputer):
    """Fit a fixed family of dense gravity lines in reached azimuth.

    Construction performs no tracing. Each field's fixed family is shared
    between every level. Two sets cover both starting and terminal azimuths;
    cosine least-squares fits at each depth or radius serve all azimuth queries.
    Reciprocal components reuse the canonical family.
    The callable diagnostic interface returns an individually traced line to
    twice the L2 depth. The private exact oracle traces to 1.05 times that depth.
    The underscore-prefixed fit controls are unsupported diagnostic overrides.

    Attributes:
        l1_computer: Critical point defining the seed and canonical level.
        outer_computer: Critical points defining the integration depth.
        tolerance: Integrator error target scaled by the L1 distance.
        maxsize: LRU capacity per memoized node; None allows unbounded caches.
        _line_count: Unsupported lines-per-set override, at least three.
        _series_terms: Unsupported fit-term override, between one and _line_count.
        family_maxsize: Number of retained families; defaults to four. Zero disables retention.
        family: Memoized fixed families in the canonical component frame.
        section_computer: Memoized cosine fits at a given depth.
        cut_section_computer: Memoized cosine fits at a given transverse radius.
    """

    # Constant-rho cuts need lines beyond the public 1.05-depth rim bound, so
    # every azimuth reaches the cylinders used by cut and ray queries.
    _line_depth_factor: ClassVar[float] = WATERSHED_FAMILY_DEPTH_COVERAGE_FACTOR

    _line_count: int = WATERSHED_AZIMUTH_LINE_COUNT
    _series_terms: int = WATERSHED_COSINE_SERIES_TERMS
    family_maxsize: int = 4
    family: Memoized[[RocheField], _LineFamily] = dataclass_field(init=False)
    section_computer: Memoized[[RocheField, float], _Section] = dataclass_field(
        init=False
    )
    cut_section_computer: Memoized[[RocheField, float], _CutSection] = dataclass_field(
        init=False
    )
    cut_height_computer: _FamilyCutHeight = dataclass_field(init=False)
    ray_computer: _FamilySheetPoint = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if require_int(self._line_count, name="_line_count") < 3:
            raise ValueError("_line_count: expected at least 3")
        if (
            not 1
            <= require_int(self._series_terms, name="_series_terms")
            <= self._line_count
        ):
            raise ValueError("_series_terms: expected between 1 and _line_count")
        if require_int(self.family_maxsize, name="family_maxsize") < 0:
            raise ValueError("family_maxsize: expected a non-negative integer")
        _ExactWatershedComputer.__post_init__(self)

    # typing.override is unavailable on the minimum supported Python (3.11).
    def _build_nodes(  # type: ignore[missing-override-decorator]
        self, line: Memoized[[RocheField, float], WatershedLine]
    ) -> None:
        family = Memoized(
            _BuildFamily(line, self._line_count, self.outer_computer),
            maxsize=self.family_maxsize,
        )
        section = Memoized(
            _BuildSection(family, self._series_terms), maxsize=self.maxsize
        )
        build_cut_section = _BuildCutSection(family, self._series_terms)
        cut_section = Memoized(build_cut_section, maxsize=self.maxsize)
        object.__setattr__(self, "cut_section_computer", cut_section)
        depth = Memoized(_FamilyRim(section, self.l1_computer), maxsize=self.maxsize)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "section_computer", section)
        object.__setattr__(self, "depth_computer", depth)
        object.__setattr__(
            self,
            "rim_computer",
            Memoized(
                _Rim(self.l1_computer, depth, self.outer_computer), maxsize=self.maxsize
            ),
        )
        object.__setattr__(
            self,
            "cut_height_computer",
            _FamilyCutHeight(
                self.l1_computer,
                self.outer_computer,
                section,
                cut_section,
                self.tolerance,
            ),
        )
        object.__setattr__(
            self,
            "ray_computer",
            _FamilySheetPoint(self.l1_computer, build_cut_section),
        )
