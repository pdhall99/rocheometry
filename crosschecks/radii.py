"""Exact surface radius along a direction, by an independent ray root solve."""

from __future__ import annotations

from math import atan2, hypot

import mpmath as mp

import rocheometry
from crosschecks.problem import Level, gradient, masses, potential
from rocheometry._computers.watershed.traced import _ExactWatershedComputer
from rocheometry._geometry import Point3D


def exact_radius(
    surface: rocheometry.ComponentSurfaceGeometry, direction: Point3D
) -> float:
    """Return the radius of a surface along a direction.

    Inside a contact part's watershed cut, the radius is where the ray meets
    the sheet, from the library's exact watershed tracer. Elsewhere it is the
    ray root of the level, solved at 40 digits.
    """
    g = surface.component
    f = g.field
    length = hypot(*direction)
    x, y, z = direction
    direction = (x / length, y / length, z / length)
    if direction == (1.0, 0.0, 0.0) and surface._depth >= 0:
        return g.l1().position[0]
    if surface._depth > 0:
        rim = g._computers.watershed.rim(
            f, surface._depth, atan2(direction[2], direction[1])
        )
        if atan2(hypot(*direction[1:]), direction[0]) < atan2(hypot(*rim[1:]), rim[0]):
            tracer = _ExactWatershedComputer(
                l1_computer=g._computers.l1,
                outer_computer=g._computers.outer,
                tolerance=1e-14,
            )
            return hypot(*tracer.sheet_point_on_ray(f, direction))
    return ray_root(Level.of(surface), direction, surface.radius(direction))


def ray_root(level: Level, direction: Point3D, guess: float) -> float:
    """Return where a ray from the component meets a level, by 40-digit Newton steps."""
    with mp.workdps(40):
        m, c = masses(level.binary, mp.mpf)
        d = tuple(map(mp.mpf, direction))
        value = level.value()
        target = mp.mpf(value.numerator) / value.denominator
        r = mp.mpf(guess)
        for _ in range(40):
            point = (r * d[0], r * d[1], r * d[2])
            residual = potential(m, c, *point) - target
            derivative = sum(g * v for g, v in zip(gradient(m, c, *point), d))
            step = residual / derivative
            r -= step
            if abs(step) < mp.mpf("1e-28") * r:
                return float(r)
        raise RuntimeError("40-digit ray root did not converge")
