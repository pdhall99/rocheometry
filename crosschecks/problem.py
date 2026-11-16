"""The Roche problem and its exact potential, shared by every crosscheck.

The library and the crosschecks solve the same problem: a component of mass
fraction `m` at the origin, its companion of mass fraction `c` at x = 1, and a
level at depth `U(L1) - U` from the L1 anchor: positive for contact levels and
negative for detached ones. `Binary` and `Level` state that
problem as plain binary64 numbers, and `Level.of` is the one place that reads
it from a Rocheometry surface. Everything here is evaluated in the caller's own
number type: exact `Fraction`, 100-digit `Decimal` or `mpmath` at the working
precision.

With the component at the origin, the potential is

    U = m/r1 + c/r2 + (m + c) ((x - c/(m + c))^2 + y^2) / 2,

where r1 and r2 are the distances to the component and the companion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache
from typing import TypeVar

import mpmath as mp

import rocheometry
from rocheometry._roche.field import RocheField

#: A high-precision real: 100-digit `Decimal` or an `mpmath` number.
Real = TypeVar("Real", Decimal, mp.mpf)


@dataclass(frozen=True)
class Binary:
    """The mass fractions of a component and its companion."""

    component_mass: float
    companion_mass: float

    @classmethod
    def of(cls, field: RocheField) -> Binary:
        """Return the masses a Rocheometry field stores."""
        return cls(field.component_mass, field.companion_mass)


@dataclass(frozen=True)
class Level:
    """A level set: the binary, its L1 anchor on the x-axis, and its depth `U(L1) - U`.

    The anchor is where the library places L1. The potential is stationary
    there, so the rounding of the anchor moves the level only at second order.
    """

    binary: Binary
    l1_x: float
    depth: float

    @classmethod
    def of(cls, surface: rocheometry.ComponentSurfaceGeometry) -> Level:
        """Return the problem a Rocheometry surface solves."""
        g = surface.component
        return cls(Binary.of(g.field), g.l1().position[0], surface._depth)

    def value(self) -> Fraction:
        """Return the level's potential, `U(L1) - depth`, exactly."""
        return _level_value(self)


def axis_potential(binary: Binary, x: float | Fraction) -> Fraction:
    """Return the exact potential at a point on the x-axis."""
    m, c, a = (
        Fraction(binary.component_mass),
        Fraction(binary.companion_mass),
        Fraction(x),
    )
    return m / abs(a) + c / abs(1 - a) + (m + c) * (a - c / (m + c)) ** 2 / 2


@cache
def _level_value(level: Level) -> Fraction:
    return axis_potential(level.binary, level.l1_x) - Fraction(level.depth)


def masses(binary: Binary, kind: type[Real]) -> tuple[Real, Real]:
    """Return the two masses, converted exactly to a high-precision type."""
    return kind(binary.component_mass), kind(binary.companion_mass)


def potential(m: Real, c: Real, x: Real, y: Real, z: Real) -> Real:
    """Return the potential at a point, in the point's number type."""
    total = m + c
    r1 = _sqrt(x * x + y * y + z * z)
    r2 = _sqrt((x - 1) ** 2 + y * y + z * z)
    return m / r1 + c / r2 + total * ((x - c / total) ** 2 + y * y) / 2


def gradient(m: Real, c: Real, x: Real, y: Real, z: Real) -> tuple[Real, Real, Real]:
    """Return the potential's gradient at a point, in the point's number type."""
    total = m + c
    r1 = _sqrt(x * x + y * y + z * z)
    r2 = _sqrt((x - 1) ** 2 + y * y + z * z)
    return (
        -m * x / r1**3 - c * (x - 1) / r2**3 + total * x - c,
        -m * y / r1**3 - c * y / r2**3 + total * y,
        -m * z / r1**3 - c * z / r2**3,
    )


def exact_values(
    binary: Binary, coordinates: Iterable[Fraction]
) -> tuple[Decimal, tuple[Decimal, Decimal, Decimal]]:
    """Return the potential and gradient at exact rational coordinates.

    The coordinates are rounded once to the current `Decimal` context.
    """
    x, y, z = (Decimal(v.numerator) / Decimal(v.denominator) for v in coordinates)
    m, c = masses(binary, Decimal)
    return potential(m, c, x, y, z), gradient(m, c, x, y, z)


def _sqrt(value: Real) -> Real:
    return value.sqrt() if isinstance(value, Decimal) else mp.sqrt(value)
