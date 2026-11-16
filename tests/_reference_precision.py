"""Independent 40-digit radius roots on unchanged binary64 quadrature rays."""

import mpmath as mp

from rocheometry._computers.surface_sample.sample import SurfaceSample
from rocheometry._roche.field import RocheField


def volume_at_40_digits(
    field: RocheField,
    sample: SurfaceSample,
    level: float,
    *,
    saddle_x: float | None = None,
) -> tuple[str, float]:
    """Return high-precision volume and largest relative binary64 radius error.

    Preserve float vectors, weights and mass coefficients exactly. A designated
    critical level means the exact potential at its computed axis position;
    every other level is the exact value of its absolute binary64 float.
    """
    with mp.workdps(40):
        m, c = mp.mpf(field.component_mass), mp.mpf(field.companion_mass)
        total = m + c
        target = mp.mpf(level)
        if saddle_x is not None:
            a = mp.mpf(saddle_x)
            target = m / abs(a) + c / abs(1 - a) + total * (a - c / total) ** 2 / 2
        terms: list[mp.mpf] = []
        largest = mp.mpf(0)
        for (direction, weight), guess in zip(sample.rays, sample.radii, strict=True):
            x, y, z = map(mp.mpf, direction)
            norm2, xy2 = x * x + y * y + z * z, x * x + y * y
            r = mp.mpf(guess)
            for _ in range(40):
                r0 = r * mp.sqrt(norm2)
                r1 = mp.sqrt(1 - 2 * r * x + r * r * norm2)
                residual = (
                    m / r0
                    + c / r1
                    + total * ((r * x - c / total) ** 2 + (r * y) ** 2) / 2
                    - target
                )
                derivative = (
                    -m / (r0 * r)
                    + c * (x - r * norm2) / r1**3
                    + total * r * xy2
                    - c * x
                )
                step = residual / derivative
                r -= step
                if abs(step) < mp.mpf("1e-26") * r:
                    break
            else:
                raise AssertionError("40-digit radial root did not converge")
            assert r > 0
            terms.append(mp.mpf(weight) * r**3 / 3)
            largest = max(largest, abs(r / mp.mpf(guess) - 1))
        return str(mp.fsum(terms)), float(largest)


def geometry_at_40_digits(
    field: RocheField,
    sample: SurfaceSample,
    level: float,
) -> tuple[str, str, tuple[tuple[str, str, str], ...]]:
    """Return volume, integral H dA and node normals at a literal level.

    Solve the identical binary64 rays with independent 40-digit potential,
    gradient and Hessian formulas. This isolates radius/geometry arithmetic
    from the quadrature's discretization error.
    """
    with mp.workdps(40):
        m, c, target = map(mp.mpf, (field.component_mass, field.companion_mass, level))
        total = m + c
        volumes: list[mp.mpf] = []
        curvatures: list[mp.mpf] = []
        normals: list[tuple[str, str, str]] = []
        for (direction, weight), guess in zip(sample.rays, sample.radii, strict=True):
            d = tuple(map(mp.mpf, direction))
            dx, dy, dz = d
            norm2, xy2 = sum(v * v for v in d), dx * dx + dy * dy
            r = mp.mpf(guess)
            for _ in range(40):
                r0 = r * mp.sqrt(norm2)
                r1 = mp.sqrt(1 - 2 * r * dx + r * r * norm2)
                residual = (
                    m / r0
                    + c / r1
                    + total * ((r * dx - c / total) ** 2 + (r * dy) ** 2) / 2
                    - target
                )
                derivative = (
                    -m / (r0 * r)
                    + c * (dx - r * norm2) / r1**3
                    + total * r * xy2
                    - c * dx
                )
                step = residual / derivative
                r -= step
                if abs(step) < mp.mpf("1e-26") * r:
                    break
            else:
                raise AssertionError("40-digit radial root did not converge")
            p = tuple(r * v for v in d)
            a = (p[0] - 1, p[1], p[2])
            r0 = mp.sqrt(sum(v * v for v in p))
            r1 = mp.sqrt(sum(v * v for v in a))
            g = tuple(
                -m * p[j] / r0**3
                - c * a[j] / r1**3
                + (total * p[j] - (c if j == 0 else 0) if j < 2 else 0)
                for j in range(3)
            )
            magnitude2 = sum(v * v for v in g)
            magnitude = mp.sqrt(magnitude2)
            h = tuple(
                tuple(
                    m * (3 * p[j] * p[k] / r0**5 - (1 / r0**3 if j == k else 0))
                    + c * (3 * a[j] * a[k] / r1**5 - (1 / r1**3 if j == k else 0))
                    + (total if j == k and j < 2 else 0)
                    for k in range(3)
                )
                for j in range(3)
            )
            hnn = (
                sum(g[j] * h[j][k] * g[k] for j in range(3) for k in range(3))
                / magnitude2
            )
            mean = (hnn - 2 * total) / (2 * magnitude)
            area = r * r * magnitude / -sum(v * w for v, w in zip(g, d, strict=True))
            volumes.append(mp.mpf(weight) * r**3 / 3)
            curvatures.append(mp.mpf(weight) * mean * area)
            normals.append(
                (str(-g[0] / magnitude), str(-g[1] / magnitude), str(-g[2] / magnitude))
            )
        return str(mp.fsum(volumes)), str(mp.fsum(curvatures)), tuple(normals)
