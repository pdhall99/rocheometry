"""Gaussian-curvature diagnostics on one shared surface sample."""

from math import cos, fsum, pi, sin, sqrt

import rocheometry
from rocheometry._numerics.quadrature import gauss_legendre


def cone_gaussian(geometry: rocheometry.ComponentGeometry) -> float:
    """Return 4*pi minus the Hessian cone deficit, independently of surface rays."""
    h = geometry.l1().hessian
    a, b = sqrt(h[0][0] / -h[1][1]), sqrt(h[0][0] / -h[2][2])
    ns, ws = gauss_legendre(64)
    terms = []
    for n, w in zip(ns, ws, strict=True):
        psi = pi * (n + 1) / 4
        terms.append(
            w
            * sqrt(a * a * b * b + b * b * cos(psi) ** 2 + a * a * sin(psi) ** 2)
            / (1 + a * a * cos(psi) ** 2 + b * b * sin(psi) ** 2)
        )
    return 2 * pi + pi * fsum(terms)
