"""Surface integrals by an independent rule, for smooth detached surfaces.

The library's quadrature clusters its nodes towards the necks and reads
curvatures from its own sample. This module shares neither: it is a plain
tensor rule (Gauss-Legendre in the angle from the +x axis, the trapezoid rule
in azimuth around it), with the area element, mean curvature and gravity
written directly from the implicit-surface formulas of the field. Only the
field and the ray radius are shared; radii have their own independent check
(``crosschecks.radii``).

The rule converges spectrally on smooth surfaces, so it is evaluated at two
orders and the difference is reported as its own error. It is not meant for
contact surfaces, whose watershed cut is not smooth, nor for surfaces so close
to critical that the L1 neck needs clustering: there its own error estimate
shows it.
"""

from __future__ import annotations

from math import cos, fsum, hypot, pi, sin, sqrt
from typing import Any

from rocheometry._numerics.quadrature import gauss_legendre

#: Orders (polar, azimuth over the full circle) of the two evaluations.
ORDERS = ((96, 96), (160, 160))
#: Floor on a value's relative error: eight units in the last place.
ROUNDING = 8 * 2.0**-52
NAMES = (
    "volume",
    "area",
    "integrated_mean_curvature",
    "mean_gravity",
    "x2",
    "z2",
    "gravity4",
)


def _integrals(surface: Any, polar: int, azimuth: int) -> dict[str, float]:
    if azimuth % 4:
        raise ValueError("the azimuth order must be divisible by four")
    field = surface.component.field
    nodes, weights = gauss_legendre(polar)
    # Reflection symmetry in y and in z: one quadrant of azimuths, weight four.
    psis = [(k + 0.5) * 2 * pi / azimuth for k in range(azimuth // 4)]
    dpsi = 4 * 2 * pi / azimuth
    terms: dict[str, list[float]] = {
        name: [] for name in NAMES[:3] + ("flux",) + NAMES[4:]
    }
    for node, weight in zip(nodes, weights, strict=True):
        theta = pi * (node + 1) / 2
        st, ct = sin(theta), cos(theta)
        omega = weight * pi / 2 * st * dpsi
        for psi in psis:
            d = (ct, st * cos(psi), st * sin(psi))
            r = surface.radius(d)
            p = (r * d[0], r * d[1], r * d[2])
            g = field.gradient(*p)
            h = field.hessian(*p)
            gnorm = hypot(*g)
            radial = g[0] * d[0] + g[1] * d[1] + g[2] * d[2]
            area = omega * r * r * gnorm / abs(radial)
            laplacian = h[0][0] + h[1][1] + h[2][2]
            ghg = sum(g[i] * h[i][j] * g[j] for i in range(3) for j in range(3))
            # Level sets enclose the region of larger potential, so the outward
            # normal is -grad U; this sign makes a sphere's curvature 1/r.
            mean = -(gnorm * gnorm * laplacian - ghg) / (2 * gnorm**3)
            terms["volume"].append(omega * r**3 / 3)
            terms["area"].append(area)
            terms["integrated_mean_curvature"].append(area * mean)
            terms["flux"].append(area * gnorm)
            terms["x2"].append(area * p[0] ** 2)
            terms["z2"].append(area * p[2] ** 2)
            terms["gravity4"].append(area * gnorm**4)
    sums = {name: fsum(values) for name, values in terms.items()}
    sums["mean_gravity"] = sums.pop("flux") / sums["area"]
    return {name: sums[name] for name in NAMES}


def surface_integrals(surface: Any) -> dict[str, dict[str, float]]:
    """Values at the higher order, and their relative errors.

    An error is the change from the lower order, but never below
    ``ROUNDING``, the rounding of a sum over the nodes.
    """
    low, high = (_integrals(surface, p, a) for p, a in ORDERS)
    return dict(
        value=high,
        error={n: max(abs(high[n] / low[n] - 1), ROUNDING) for n in NAMES},
    )


def volume_radius(volume: float) -> float:
    return (3 * volume / (4 * pi)) ** (1 / 3)


def area_radius(area: float) -> float:
    return sqrt(area / (4 * pi))
