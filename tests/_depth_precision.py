"""Independent high-precision contact-neck roots with exact float inputs."""

import math

import mpmath as mp

from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


def ray_at_depth(
    field: RocheField, x1: float, depth: float, direction: Point3D
) -> tuple[float, Point3D]:
    """Return a first enclosing root and its gradient, solved at 80 digits."""
    with mp.workdps(80):
        m, c = mp.mpf(field.component_mass), mp.mpf(field.companion_mass)
        total, a = m + c, mp.mpf(x1)
        target = m / a + c / (1 - a) + total / 2 * (a - c / total) ** 2 - mp.mpf(depth)
        d = tuple(map(mp.mpf, direction))

        def value(radius: mp.mpf) -> mp.mpf:
            x, y, z = (radius * v for v in d)
            return (
                m / mp.sqrt(x * x + y * y + z * z)
                + c / mp.sqrt((x - 1) ** 2 + y * y + z * z)
                + total / 2 * ((x - c / total) ** 2 + y * y)
                - target
            )

        upper = a / d[0]
        width = max(mp.sqrt(mp.mpf(depth)), a * mp.sqrt(d[1] ** 2 + d[2] ** 2))
        lower = upper - width
        while value(lower) <= 0:
            width *= 2
            lower = upper - width
        assert value(upper) < 0
        for _ in range(220):
            middle = (lower + upper) / 2
            if value(middle) > 0:
                lower = middle
            else:
                upper = middle
        root = (lower + upper) / 2
        x, y, z = (root * v for v in d)
        r0, r1 = mp.sqrt(x * x + y * y + z * z), mp.sqrt((x - 1) ** 2 + y * y + z * z)
        gradient = (
            -m * x / r0**3 - c * (x - 1) / r1**3 + total * x - c,
            -m * y / r0**3 - c * y / r1**3 + total * y,
            -m * z / r0**3 - c * z / r1**3,
        )
        return float(root), (float(gradient[0]), float(gradient[1]), float(gradient[2]))


def direction_angle(a: Point3D, b: Point3D) -> float:
    """Return a stable angle between near-parallel vectors, as a diagnostic."""
    cross = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    return math.atan2(math.hypot(*cross), sum(x * y for x, y in zip(a, b, strict=True)))


def projected_at_depth(
    projected: ProjectedField, angle: float
) -> tuple[float, Point3D]:
    """Solve the limb's level/stationarity equations independently at 80 digits."""
    radius = projected.radius(angle)
    c0, s0 = math.cos(angle), math.sin(angle)
    sky = (radius * c0, radius * s0)
    displacement = projected._neck_points[sky]
    point = projected._points[angle][1]
    with mp.workdps(80):
        m, c = (
            mp.mpf(projected.field.component_mass),
            mp.mpf(projected.field.companion_mass),
        )
        total = m + c
        a, depth = mp.mpf(projected.l1.position[0]), mp.mpf(projected.depth)
        scale = mp.sqrt(depth)
        ct, st = mp.mpf(c0), mp.mpf(s0)
        ci, si = mp.mpf(projected.c), mp.mpf(projected.s)
        target = m / a + c / (1 - a) + total / 2 * (a - c / total) ** 2 - depth

        def evaluate(
            q: mp.mpf, v: mp.mpf
        ) -> tuple[mp.mpf, tuple[mp.mpf, mp.mpf, mp.mpf]]:
            x = a + q * scale
            t, w = x / ct * st, v * scale
            y, z = t * ci + w * si, -t * si + w * ci
            r0, r1 = (
                mp.sqrt(x * x + y * y + z * z),
                mp.sqrt((x - 1) ** 2 + y * y + z * z),
            )
            value = (
                m / r0 + c / r1 + total / 2 * ((x - c / total) ** 2 + y * y) - target
            )
            gradient = (
                -m * x / r0**3 - c * (x - 1) / r1**3 + total * x - c,
                -m * y / r0**3 - c * y / r1**3 + total * y,
                -m * z / r0**3 - c * z / r1**3,
            )
            return value, gradient

        def residual(q: mp.mpf, v: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
            value, (_, gy, gz) = evaluate(q, v)
            return value / depth, (gy * si + gz * ci) / scale

        q0 = mp.mpf(math.fsum(displacement[0])) / scale
        v0 = (mp.mpf(point[1]) * si + mp.mpf(point[2]) * ci) / scale
        q, v = mp.findroot(residual, (q0, v0), tol=mp.mpf("1e-60"), maxsteps=50)
        _, (gx, gy, gz) = evaluate(q, v)
        return float((a + q * scale) / ct), (float(gx), float(gy * ci - gz * si), 0.0)
