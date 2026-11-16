"""Optional bracket-confined Newton polish after the scalar junction solve."""

from __future__ import annotations

from math import atan2, hypot
from typing import TYPE_CHECKING

from rocheometry._computers.ownership import certify_part
from rocheometry._computers.section_path.traced import _trig
from rocheometry._computers.silhouette.projected import ProjectedField

# Insensitive control: limb junction newton step relative tolerance.
LIMB_JUNCTION_NEWTON_STEP_RELATIVE_TOLERANCE = 4e-16

if TYPE_CHECKING:
    from rocheometry._computers.silhouette.limb import _Junction


def _newton_polish(
    field: ProjectedField, left: float, right: float, *, seed_angle: float | None = None
) -> _Junction:
    """Solve both branch levels and stationarity with a Schur-complement Newton step.

    The interval was certified to contain two maxima with opposite height
    ordering at its ends. Each Newton step retains both negative curvatures.
    """
    angle = (left + right) / 2 if seed_angle is None else seed_angle
    radius = field.radius(angle)
    c, s = _trig(angle)
    x, t = radius * c, radius * s
    maxima = tuple(
        p
        for p in field.maxima(x, t)
        if field._los(p)[1] < 0 and hypot(*p) < field.outer
    )
    w = [p[1] * field.s + p[2] * field.c for p in (maxima[0], maxima[-1])]
    previous = float("inf")
    for _ in range(64):
        points = [field._point(x, t, v) for v in w]
        residual = [field._residual(p) for p in points]
        derivatives = [field._los(p) for p in points]
        gradients = [field.field.gradient(*p) for p in points]
        sky = [(g[0], g[1] * field.c - g[2] * field.s) for g in gradients]
        # U_v is zero to the stationary-root precision; eliminate its Newton
        # correction as well, so the four equations share one evaluation chain.
        cross = []
        rhs = []
        for p, value, (d, h), (gx, gt) in zip(
            points, residual, derivatives, sky, strict=True
        ):
            if h >= 0:
                raise RuntimeError("junction Newton left a maximum branch")
            matrix = field.field.hessian(*p)
            vx = matrix[0][1] * field.s + matrix[0][2] * field.c
            vt = field.c * (
                matrix[1][1] * field.s + matrix[1][2] * field.c
            ) - field.s * (matrix[2][1] * field.s + matrix[2][2] * field.c)
            cross.append((vx, vt))
            rhs.append((gx - d * vx / h, gt - d * vt / h, -value + d * d / h))
        a, b, f = rhs[0]
        c0, d0, g = rhs[1]
        determinant = a * d0 - b * c0
        if determinant == 0:
            raise RuntimeError(
                f"degenerate equal-height limb junction: {left, right, angle, x, t, w, len(maxima)}"
            )
        dx, dt = (f * d0 - b * g) / determinant, (a * g - f * c0) / determinant
        dw = [
            -(d + vx * dx + vt * dt) / h
            for (d, h), (vx, vt) in zip(derivatives, cross, strict=True)
        ]
        error = max(abs(v) for v in residual)
        if (x + dx, t + dt) == (x, t) or (
            error >= previous
            and hypot(dx, dt)
            < LIMB_JUNCTION_NEWTON_STEP_RELATIVE_TOLERANCE * hypot(x, t)
        ):
            break
        previous = error
        fraction = 1.0
        while True:
            candidate = [
                field._point(x + fraction * dx, t + fraction * dt, v + fraction * dv)
                for v, dv in zip(w, dw, strict=True)
            ]
            if (
                all(field._los(p)[1] < 0 for p in candidate)
                and w[0] + fraction * dw[0] < w[1] + fraction * dw[1]
            ):
                break
            fraction /= 2
            if fraction == 0:
                raise RuntimeError("junction Newton could not retain its branches")
        x, t = x + fraction * dx, t + fraction * dt
        w = [v + fraction * dv for v, dv in zip(w, dw, strict=True)]
    else:
        raise RuntimeError("equal-height limb junction did not converge")
    angle = atan2(t, x)
    if not left <= angle <= right:
        raise RuntimeError("limb junction lies outside its switching bracket")
    points = tuple(field._point(x, t, v) for v in w)
    for point in points:
        certify_part(
            field.field, field.l1, field.watershed, point, contact=field.contact
        )
    floor = field._value_roundoff(x, t)
    if (
        max(hypot(*p) for p in points) >= field.outer
        or abs(field.value(x, t)) > floor
        or any(abs(field._residual(p)) > floor for p in points)
    ):
        raise RuntimeError("equal-height junction is not on the global limb")
    radius = hypot(x, t)
    c, s = _trig(angle)
    slopes = []
    for point in points:
        gx, gy, gz = field.field.gradient(*point)
        gt = gy * field.c - gz * field.s
        radial = c * gx + s * gt
        if radial >= 0:
            raise RuntimeError("limb junction is not star-shaped")
        slopes.append(-radius * (-s * gx + c * gt) / radial)
    # The outer envelope is max(r1,r2). The smaller radial slope wins
    # before the crossing, and the larger one after it.
    before, after = points if slopes[0] < slopes[1] else points[::-1]
    from rocheometry._computers.silhouette.limb import _Junction

    return _Junction(angle, (x, t), before, after)
