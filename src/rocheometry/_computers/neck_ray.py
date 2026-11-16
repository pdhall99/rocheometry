"""Untruncated, displacement-preserving roots on structurally thin necks.

The sixth-order local model is sufficient only on its certified Taylor domain.
Quadrature rays outside that domain still need an unrounded root and gradient.
Double-double factored differences retain these without an angular window or
an additional solve tolerance.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import fsum, sqrt

from rocheometry._computers.neck import _DD, _add, _mul, _scale
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


def _neg(a: _DD) -> _DD:
    return -a[0], -a[1]


def _sub(a: _DD, b: _DD) -> _DD:
    return _add(a, _neg(b))


def _div(a: _DD, b: _DD) -> _DD:
    high = a[0] / b[0]
    remainder = _sub(a, _scale(b, high))
    return _add((high, 0.0), (fsum(remainder) / b[0], 0.0))


def _sqrt(a: _DD) -> _DD:
    high = sqrt(a[0])
    remainder = _sub(a, _mul((high, 0.0), (high, 0.0)))
    return _add((high, 0.0), (fsum(remainder) / (2 * high), 0.0))


@dataclass(frozen=True, slots=True)
class _ExactNeckRay:
    m: _DD
    c: _DD
    total: _DD
    anchor: _DD
    companion: _DD

    def evaluate(
        self, radius: _DD, direction: Point3D, depth: float
    ) -> tuple[_DD, tuple[_DD, _DD, _DD]]:
        x, y, z = (_scale(radius, d) for d in direction)
        displacement = _sub(x, self.anchor)
        y2, z2 = _mul(y, y), _mul(z, z)
        transverse = _add(y2, z2)
        r1 = _sqrt(_add(_mul(x, x), transverse))
        other_x = _sub(x, (1.0, 0.0))
        r2 = _sqrt(_add(_mul(other_x, other_x), transverse))
        d2 = _mul(displacement, displacement)
        first = _neg(
            _add(_add(_scale(_mul(self.anchor, displacement), 2), d2), transverse)
        )
        second = _sub(
            _scale(_mul(self.companion, displacement), 2), _add(d2, transverse)
        )
        first = _div(
            _mul(self.m, first), _mul(_mul(r1, self.anchor), _add(r1, self.anchor))
        )
        second = _div(
            _mul(self.c, second),
            _mul(_mul(r2, self.companion), _add(r2, self.companion)),
        )
        centrifugal = _add(
            _mul(
                displacement,
                _sub(_scale(_mul(self.total, _add(x, self.anchor)), 0.5), self.c),
            ),
            _scale(_mul(self.total, y2), 0.5),
        )
        value = _add(_add(first, second), _add(centrifugal, (depth, 0.0)))
        a = _div(self.m, _mul(_mul(r1, r1), r1))
        b = _div(self.c, _mul(_mul(r2, r2), r2))
        gradient = (
            _sub(_sub(_mul(self.total, x), self.c), _add(_mul(a, x), _mul(b, other_x))),
            _mul(y, _sub(self.total, _add(a, b))),
            _neg(_mul(z, _add(a, b))),
        )
        return value, gradient


@lru_cache(maxsize=4096)
def _exact_neck_ray(field: RocheField, x1: float) -> _ExactNeckRay:
    m, c = (field.component_mass, 0.0), (field.companion_mass, 0.0)
    return _ExactNeckRay(m, c, _add(m, c), (x1, 0.0), _sub((1.0, 0.0), (x1, 0.0)))


def _refined_neck_ray(
    field: RocheField, x1: float, depth: float, direction: Point3D, radius: float
) -> tuple[float, Point3D]:
    """Polish a bracketed envelope root until double-double representation stops.

    This does not select a branch: the ordinary bracketed solve supplies the
    star-facing root. Corrections retain their low component through gradient
    evaluation, even when adding them to the absolute radius changes no bit.
    """
    model = _exact_neck_ray(field, x1)
    current = (radius, 0.0)
    previous = None
    previous_error = float("inf")
    for _ in range(32):
        value, gradient = model.evaluate(current, direction, depth)
        derivative = (0.0, 0.0)
        for g, d in zip(gradient, direction, strict=True):
            derivative = _add(derivative, _scale(g, d))
        if derivative[0] >= 0:
            raise RuntimeError("neck refinement left the star-facing envelope root")
        error = abs(fsum(value))
        if (
            previous is not None
            and current[0] == previous[0]
            and error >= previous_error
        ):
            _, gradient = model.evaluate(previous, direction, depth)
            return fsum(previous), (
                fsum(gradient[0]),
                fsum(gradient[1]),
                fsum(gradient[2]),
            )
        updated = _sub(current, _div(value, derivative))
        updated_value, updated_gradient = model.evaluate(updated, direction, depth)
        # The public root and its analytic gradient are binary64 values.
        # Once both reports stop changing, further low-component iterations
        # cannot improve either report; no accuracy threshold is introduced.
        reports_unchanged = updated[0] == current[0] and all(
            fsum(a) == fsum(b) for a, b in zip(gradient, updated_gradient, strict=True)
        )
        if updated == current or updated == previous or reports_unchanged:
            if abs(fsum(updated_value)) < abs(fsum(value)):
                current = updated
            _, gradient = model.evaluate(current, direction, depth)
            return fsum(current), (
                fsum(gradient[0]),
                fsum(gradient[1]),
                fsum(gradient[2]),
            )
        previous_error = error
        previous, current = current, updated
    raise RuntimeError(
        f"double-double neck refinement did not reach representable resolution: {field.qi=}, {depth=}, {direction=}, {radius=}, {current=}, {previous=}"
    )
