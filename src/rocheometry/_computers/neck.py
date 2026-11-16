"""Contact-neck geometry before an L1 displacement is rounded away."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import fsum, hypot, sqrt, ulp
from sys import float_info

from rocheometry._geometry import Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._roche.field import RocheField

_DD = tuple[float, float]


def _add(a: _DD, b: _DD) -> _DD:
    high = a[0] + b[0]
    v = high - a[0]
    low = fsum((a[0] - (high - v), b[0] - v, a[1], b[1]))
    result = high + low
    return result, low - (result - high)


def _mul(a: _DD, b: _DD) -> _DD:
    # Dekker's exact product split also works on the supported Python 3.11,
    # which has no math.fma. Neck coordinates are normal, bounded numbers.
    splitter = 2**27 + 1.0
    ah = splitter * a[0]
    ah -= ah - a[0]
    al = a[0] - ah
    bh = splitter * b[0]
    bh -= bh - b[0]
    bl = b[0] - bh
    high = a[0] * b[0]
    low = fsum(
        (
            ah * bh - high,
            ah * bl,
            al * bh,
            al * bl,
            a[0] * b[1],
            a[1] * b[0],
            a[1] * b[1],
        )
    )
    result = high + low
    return result, low - (result - high)


def _scale(a: _DD, b: float) -> _DD:
    return _mul(a, (b, 0.0))


def _split(value: Fraction) -> _DD:
    high = float(value)
    return high, float(value - Fraction(high))


@dataclass(frozen=True, slots=True)
class _NeckField:
    x1: float
    linear: _DD
    xx: _DD
    yy: _DD
    zz: _DD
    cubic: _DD
    quartic: _DD
    fifth: _DD
    sixth: _DD

    def value(self, displacement: tuple[_DD, _DD, _DD], depth: float) -> float:
        x, y, z = displacement
        x2, y2, z2 = _mul(x, x), _mul(y, y), _mul(z, z)
        t2 = _add(y2, z2)
        quadratic = _add(_add(_mul(self.xx, x2), _mul(self.yy, y2)), _mul(self.zz, z2))
        third = _mul(self.cubic, _mul(x, _add(_scale(x2, -1), _scale(t2, 1.5))))
        fourth = _mul(
            self.quartic,
            _add(
                _add(_mul(x2, x2), _scale(_mul(x2, t2), -3)),
                _scale(_mul(t2, t2), 0.375),
            ),
        )
        x4, t4 = _mul(x2, x2), _mul(t2, t2)
        fifth = _mul(
            self.fifth,
            _mul(
                x,
                _add(
                    _add(_scale(x4, -1), _scale(_mul(x2, t2), 5)), _scale(t4, -15 / 8)
                ),
            ),
        )
        sixth = _mul(
            self.sixth,
            _add(
                _add(_mul(x4, x2), _scale(_mul(x4, t2), -15 / 2)),
                _add(_scale(_mul(x2, t4), 45 / 8), _scale(_mul(t4, t2), -5 / 16)),
            ),
        )
        return fsum(
            _add(
                _add(
                    _add(
                        _add((depth, 0.0), _mul(self.linear, x)), _scale(quadratic, 0.5)
                    ),
                    third,
                ),
                _add(fourth, _add(fifth, sixth)),
            )
        )

    def gradient(self, displacement: tuple[_DD, _DD, _DD]) -> Point3D:
        x, y, z = displacement
        x2 = _mul(x, x)
        t2 = _add(_mul(y, y), _mul(z, z))
        gx = _add(self.linear, _mul(self.xx, x))
        gx = _add(gx, _mul(self.cubic, _add(_scale(x2, -3), _scale(t2, 1.5))))
        gx = _add(gx, _mul(self.quartic, _mul(x, _add(_scale(x2, 4), _scale(t2, -6)))))
        transverse = _add(
            _scale(_mul(self.cubic, x), 3),
            _mul(self.quartic, _add(_scale(x2, -6), _scale(t2, 1.5))),
        )
        x4, t4 = _mul(x2, x2), _mul(t2, t2)
        gx = _add(
            gx,
            _mul(
                self.fifth,
                _add(
                    _add(_scale(x4, -5), _scale(_mul(x2, t2), 15)), _scale(t4, -15 / 8)
                ),
            ),
        )
        gx = _add(
            gx,
            _mul(
                self.sixth,
                _mul(
                    x,
                    _add(
                        _add(_scale(x4, 6), _scale(_mul(x2, t2), -30)),
                        _scale(t4, 45 / 4),
                    ),
                ),
            ),
        )
        transverse = _add(
            transverse,
            _mul(self.fifth, _mul(x, _add(_scale(x2, 10), _scale(t2, -15 / 2)))),
        )
        transverse = _add(
            transverse,
            _mul(
                self.sixth,
                _add(
                    _add(_scale(x4, -15), _scale(_mul(x2, t2), 45 / 2)),
                    _scale(t4, -15 / 8),
                ),
            ),
        )
        return (
            fsum(gx),
            fsum(_mul(y, _add(self.yy, transverse))),
            fsum(_mul(z, _add(self.zz, transverse))),
        )

    def on_ray(self, offset: float, direction: Point3D) -> tuple[_DD, _DD, _DD]:
        # Division correction retains the exact float ray rather than a
        # separately rounded slope. The axial root itself is a displacement.
        axial = _add((self.x1, 0.0), (offset, 0.0))
        radius = axial[0] / direction[0]
        remainder = _add(axial, _scale(_mul((radius, 0.0), (direction[0], 0.0)), -1))
        r = _add((radius, 0.0), (fsum(remainder) / direction[0], 0.0))
        return (offset, 0.0), _scale(r, direction[1]), _scale(r, direction[2])

    def sheet_offset(self, y: float, z: float) -> float:
        hxx, hyy, hzz = fsum(self.xx), fsum(self.yy), fsum(self.zz)
        third = 3 * fsum(self.cubic)
        return -fsum(self.linear) / hxx + third / 2 * (
            y * y / (2 * hyy - hxx) + z * z / (2 * hzz - hxx)
        )


@lru_cache(maxsize=4096)
def _neck_field(field: RocheField, x1: float) -> _NeckField:
    m, c, x = (
        Fraction(field.component_mass),
        Fraction(field.companion_mass),
        Fraction(x1),
    )
    total = m + c
    s3 = m / x**3 + c / (1 - x) ** 3
    s4 = m / x**4 - c / (1 - x) ** 4
    s5 = m / x**5 + c / (1 - x) ** 5
    s6 = m / x**6 - c / (1 - x) ** 6
    s7 = m / x**7 + c / (1 - x) ** 7
    return _NeckField(
        x1,
        _split(-m / x**2 + c / (1 - x) ** 2 + total * x - c),
        _split(2 * s3 + total),
        _split(total - s3),
        _split(-s3),
        _split(s4),
        _split(s5),
        _split(s6),
        _split(s7),
    )


@lru_cache(maxsize=4096)
def _neck_surface(field: RocheField, x1: float, depth: float) -> _NeckField | None:
    if depth <= 0 or type(field) is not RocheField:
        return None
    model = _neck_field(field, x1)
    rho = sqrt(depth) * sqrt(2 / -fsum(model.zz))
    return model if _unresolved_neck_position(x1, rho) else None


def _unresolved_neck_ray(
    depth: float, direction: Point3D, x1: float, field: RocheField
) -> bool:
    """Detect transverse information lost by either point-mass distance.

    This is a representation test on a contact-neck point, not a distance
    threshold: adding rho squared to one axial squared distance changes no bit.
    """
    x, y, z = direction
    if depth <= 0 or x <= 0 or y == z == 0 or _neck_surface(field, x1, depth) is None:
        return False
    rho = x1 * hypot(y, z) / x
    # The sixth-order field has a relative gradient remainder O((rho/a)^5)
    # at a saddle. This fourth-power certificate leaves an extra rho/a of
    # margin below roundoff, without a fitted angular window.
    return (rho / min(x1, 1 - x1)) ** 4 <= float_info.epsilon


def _certified_neck_ray(
    depth: float, direction: Point3D, x1: float, field: RocheField
) -> bool:
    """Certify displacement arithmetic for a contact-envelope arm.

    The sixth-order remainder certificate also covers resolved transverse
    distances and arm data whose radial gradient subtracts nearly cancelling
    components near the rim.
    Node/rim dispatch keeps its additional distance-loss requirement.
    """
    x, y, z = direction
    if depth <= 0 or type(field) is not RocheField or x <= 0 or y == z == 0:
        return False
    rho = x1 * hypot(y, z) / x
    # The sixth-order field has a relative gradient remainder O((rho/a)^5)
    # at a saddle. This fourth-power certificate leaves an extra rho/a of
    # margin below roundoff, without a fitted angular window.
    ratio = rho / min(x1, 1 - x1)
    # Outside the nearest point-mass singularity the certificate cannot hold;
    # reject before exponentiation can overflow on an almost transverse ray.
    return ratio <= 1 and ratio**4 <= float_info.epsilon


def _unresolved_neck_position(x1: float, rho: float) -> bool:
    distance = max(x1, 1 - x1)
    return rho > 0 and hypot(distance, rho) == distance


def _neck_rim(
    field: RocheField, x1: float, depth: float, cy: float, sz: float
) -> Point3D:
    """Intersect the analytic local invariant sheet and the exact depth."""
    model = _neck_field(field, x1)
    scale = sqrt(depth) * sqrt(
        2 / -(fsum(model.yy) * cy * cy + fsum(model.zz) * sz * sz)
    )

    def point(q: float) -> tuple[_DD, _DD, _DD]:
        rho = scale * q
        y, z = rho * cy, rho * sz
        return (model.sheet_offset(y, z), 0.0), (y, 0.0), (z, 0.0)

    root = brent_root(
        lambda q: model.value(point(q), depth),
        0.0,
        2.0,
        absolute_tolerance=ulp(1.0),
        relative_tolerance=0.0,
    )
    displacement = point(root)
    return fsum((x1, *displacement[0])), fsum(displacement[1]), fsum(displacement[2])


def _neck_displacement(
    field: RocheField, x1: float, depth: float, direction: Point3D
) -> tuple[_DD, _DD, _DD]:
    model = _neck_field(field, x1)
    y, z = x1 * direction[1] / direction[0], x1 * direction[2] / direction[0]
    upper = model.sheet_offset(y, z)
    scale = hypot(sqrt(depth) * sqrt(2 / fsum(model.xx)), y, z)

    def value(coordinate: float) -> float:
        return model.value(model.on_ray(scale * coordinate, direction), depth)

    hi = upper / scale
    if value(hi) >= 0:
        # This ray meets the sheet before the envelope. The geometric part,
        # rather than a rounded rim-angle comparison, determines ownership.
        return model.on_ray(upper, direction)
    lo = hi - 1
    while value(lo) <= 0:
        lo = hi - 2 * (hi - lo)
    root = brent_root(
        value, lo, hi, absolute_tolerance=ulp(1.0), relative_tolerance=0.0
    )
    return model.on_ray(root * scale, direction)


def _neck_radius(
    field: RocheField, x1: float, depth: float, direction: Point3D
) -> float:
    displacement = _neck_displacement(field, x1, depth, direction)
    return fsum((x1, *displacement[0])) / direction[0]


def _neck_sheet_point(field: RocheField, x1: float, direction: Point3D) -> Point3D:
    model = _neck_field(field, x1)
    radius = x1 / direction[0]
    while True:
        y, z = radius * direction[1], radius * direction[2]
        updated = fsum((x1, model.sheet_offset(y, z))) / direction[0]
        if updated == radius:
            return radius * direction[0], y, z
        radius = updated
