"""Two fixed gravity-line sets and cosine fits in the reached azimuth."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, copysign, cos, fsum, hypot, log, pi, sin
from typing import TYPE_CHECKING

from rocheometry._computers.contact import _geometry_depth
from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.neck import _neck_rim, _unresolved_neck_position
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._roche.opening import _coverage_gap

# Insensitive control: watershed fit rank relative tolerance.
WATERSHED_FIT_RANK_RELATIVE_TOLERANCE = 1e-10
# Insensitive control: watershed fit trig cleanup tolerance.
WATERSHED_FIT_TRIG_CLEANUP_TOLERANCE = 1e-15

if TYPE_CHECKING:
    from rocheometry._computers.watershed.traced import WatershedLine


def _cosine(coefficients: tuple[float, ...], c: float) -> float:
    first = second = 0.0
    for value in reversed(coefficients[1:]):
        first, second = value + 2 * c * first - second, first
    return coefficients[0] + c * first - second


def _sine(coefficients: tuple[float, ...], c: float, s: float) -> float:
    first = second = 0.0
    for value in reversed(coefficients[1:]):
        first, second = value + 2 * c * first - second, first
    return s * first


def _fit(
    azimuths: tuple[float, ...], values: tuple[tuple[float, ...], ...], terms: int
) -> tuple[tuple[float, ...], ...]:
    """Fit several cosine series with one Householder QR, without normal equations."""
    rows = len(azimuths)
    matrix = []
    for psi in azimuths:
        c = cos(2 * psi)
        row = [1.0, c]
        for _ in range(2, terms):
            row.append(2 * c * row[-1] - row[-2])
        matrix.append(row[:terms])
    right = [list(column) for column in values]
    leading_diagonal = 0.0
    for k in range(terms):
        column = [matrix[i][k] for i in range(k, rows)]
        diagonal = -copysign(hypot(*column), column[0])
        if k == 0:
            leading_diagonal = abs(diagonal)
        if abs(diagonal) < WATERSHED_FIT_RANK_RELATIVE_TOLERANCE * leading_diagonal:
            raise ValueError("watershed cosine fit is rank deficient")
        column[0] -= diagonal
        norm = hypot(*column)
        reflector = [v / norm for v in column]
        for j in range(k + 1, terms):
            dot = 2 * fsum(v * matrix[i][j] for i, v in enumerate(reflector, k))
            for i, v in enumerate(reflector, k):
                matrix[i][j] -= dot * v
        matrix[k][k] = diagonal
        for rhs in right:
            dot = 2 * fsum(v * rhs[i] for i, v in enumerate(reflector, k))
            for i, v in enumerate(reflector, k):
                rhs[i] -= dot * v
    result = []
    for rhs in right:
        coefficients = [0.0] * terms
        for k in range(terms - 1, -1, -1):
            coefficients[k] = (
                rhs[k]
                - fsum(matrix[k][j] * coefficients[j] for j in range(k + 1, terms))
            ) / matrix[k][k]
        result.append(tuple(coefficients))
    return tuple(result)


def _inverse_labels(
    azimuths: tuple[float, ...], alphas: tuple[float, ...], targets: tuple[float, ...]
) -> tuple[float, ...]:
    """Monotone cubic interpolation of the inverse reached-azimuth map."""
    widths = tuple(b - a for a, b in zip(azimuths[:-1], azimuths[1:], strict=True))
    slopes = tuple(
        (b - a) / h for a, b, h in zip(alphas[:-1], alphas[1:], widths, strict=True)
    )
    derivatives = [slopes[0]]
    for i in range(1, len(alphas) - 1):
        before, after = widths[i - 1], widths[i]
        w1, w2 = 2 * after + before, after + 2 * before
        derivatives.append((w1 + w2) / (w1 / slopes[i - 1] + w2 / slopes[i]))
    derivatives.append(slopes[-1])
    result = []
    for target in targets:
        i = min(len(widths) - 1, max(0, bisect_right(azimuths, target) - 1))
        h = widths[i]
        t = (target - azimuths[i]) / h
        result.append(
            (2 * t**3 - 3 * t**2 + 1) * alphas[i]
            + (t**3 - 2 * t**2 + t) * h * derivatives[i]
            + (-2 * t**3 + 3 * t**2) * alphas[i + 1]
            + (t**3 - t**2) * h * derivatives[i + 1]
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _LineFamily:
    lines: tuple[WatershedLine, ...]


@dataclass(frozen=True, slots=True)
class _BuildFamily:
    line: Callable[[RocheField, float], WatershedLine]
    count: int
    outer_computer: CriticalPointComputer

    def __call__(self, field: RocheField, /) -> _LineFamily:
        alphas = tuple(pi * k / (2 * (self.count - 1)) for k in range(self.count))
        first = tuple(self.line(field, alpha) for alpha in alphas)
        depth = _coverage_gap(
            field,
            first[0].x_l1,
            self.outer_computer(field).position[0],
            self.outer_computer(field.reciprocal()).position[0],
        )
        points = tuple(line.at_log_depth(log(depth)) for line in first)
        azimuths = tuple(atan2(p[2], p[1]) for p in points)
        labels = _inverse_labels(azimuths, alphas, alphas[1:-1])
        # Endpoints are shared. Do not deduplicate interior coincidences: the
        # fixed labels and row count must stay continuous as the ratio varies.
        return _LineFamily(first + tuple(self.line(field, alpha) for alpha in labels))


@dataclass(frozen=True, slots=True)
class _Section:
    field: RocheField
    x_l1: float
    depth: float
    height: tuple[float, ...]
    radius: tuple[float, ...]
    mirrored: bool

    def evaluate(self, psi: float) -> tuple[float, float]:
        """Evaluate height and radius, then polish the potential along rho."""
        c = cos(2 * psi)
        x = self.x_l1 + _cosine(self.height, c)
        rho = _cosine(self.radius, c)
        cy, sz = _plane(psi)
        for _ in range(2):
            p = (x, rho * cy, rho * sz)
            _, gy, gz = self.field.gradient(*p)
            rho -= (
                self.field.potential_difference(p, (self.x_l1, 0.0, 0.0)) + self.depth
            ) / (gy * cy + gz * sz)
        return (1 - x if self.mirrored else x), rho


@dataclass(frozen=True, slots=True)
class _BuildSection:
    family: Callable[[RocheField], _LineFamily]
    terms: int

    def __call__(self, field: RocheField, depth: float, /) -> _Section:
        mirrored = type(field) is RocheField and field.qi > 1
        canonical = field.reciprocal() if mirrored else field
        lines = self.family(canonical).lines
        points = tuple(line.at_log_depth(log(depth)) for line in lines)
        x1 = lines[0].x_l1
        heights = tuple(p[0] - x1 for p in points)
        radii = tuple(hypot(p[1], p[2]) for p in points)
        azimuths = tuple(atan2(p[2], p[1]) for p in points)
        height, radius = _fit(azimuths, (heights, radii), self.terms)
        return _Section(
            canonical,
            x1,
            depth,
            height,
            radius,
            mirrored,
        )


def _plane(psi: float) -> tuple[float, float]:
    """Retain exact coordinate planes when their trig value rounds to zero."""
    cy, sz = cos(psi), sin(psi)
    return (
        0.0 if abs(cy) < WATERSHED_FIT_TRIG_CLEANUP_TOLERANCE else cy,
        0.0 if abs(sz) < WATERSHED_FIT_TRIG_CLEANUP_TOLERANCE else sz,
    )


@dataclass(frozen=True, slots=True)
class _FamilyRim:
    section: Callable[[RocheField, float], _Section]
    l1_computer: CriticalPointComputer

    def __call__(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        depth = _geometry_depth(depth, self.l1_computer(field))
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth == 0:
            return self.l1_computer(field).position
        cy, sz = _plane(psi)
        x1 = self.l1_computer(field).position[0]
        h = field.hessian(x1)
        rho = (2 * depth / -(h[1][1] * cy * cy + h[2][2] * sz * sz)) ** 0.5
        if _unresolved_neck_position(x1, rho):
            return _neck_rim(field, x1, depth, cy, sz)
        x, rho = self.section(field, depth).evaluate(atan2(abs(sz), abs(cy)))
        return x, rho * cy, rho * sz


@dataclass(frozen=True, slots=True)
class _CutSection:
    field: RocheField
    x_l1: float
    rho: float
    height: tuple[float, ...]
    height_psi: tuple[float, ...]
    mirrored: bool

    def evaluate(self, psi: float) -> tuple[float, float, float]:
        """Return x and its rho/psi derivatives on a constant-radius sheet cut."""
        c, s = cos(2 * psi), sin(2 * psi)
        if psi == 0.0:
            c, s = 1.0, 0.0
        elif psi == pi / 2:
            c, s = -1.0, 0.0
        x = self.x_l1 + _cosine(self.height, c)
        xp = _sine(self.height_psi, c, s)
        cy, sz = _plane(psi)
        gx, gy, gz = self.field.gradient(x, self.rho * cy, self.rho * sz)
        # Gravity is tangent to the invariant watershed sheet. This supplies
        # its radial derivative analytically from the fitted psi derivative.
        xr = (gx - (gz * cy - gy * sz) * xp / self.rho) / (gy * cy + gz * sz)
        return (1 - x, -xr, -xp) if self.mirrored else (x, xr, xp)


@dataclass(frozen=True, slots=True)
class _BuildCutSection:
    family: Callable[[RocheField], _LineFamily]
    terms: int

    def __call__(self, field: RocheField, rho: float, /) -> _CutSection:
        mirrored = type(field) is RocheField and field.qi > 1
        canonical = field.reciprocal() if mirrored else field
        lines = self.family(canonical).lines
        points = tuple(line.at_radius(rho) for line in lines)
        x1 = lines[0].x_l1
        azimuths = tuple(atan2(p[2], p[1]) for p in points)
        heights = tuple(p[0] - x1 for p in points)
        (height,) = _fit(azimuths, (heights,), self.terms)
        return _CutSection(
            canonical,
            x1,
            rho,
            height,
            tuple(-2 * j * a for j, a in enumerate(height)),
            mirrored,
        )
