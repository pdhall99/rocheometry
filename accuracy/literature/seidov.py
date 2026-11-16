"""L1 and the outer collinear point against Seidov's (2004) exact inverse formulas.

Seidov writes the mass ratio as a rational function of a collinear point's
position, and the potential there as a rational function of the position
alone. With M1 at the origin, M2 at x = 1 and q = M2/M1 (his eq. 1), L1 at
0 < x1 < 1 has (eqs. 3 and 4)

    q = (1 - x1)^3 (1 + x1 + x1^2) / (x1^3 (3 - 3 x1 + x1^2)),
    Psi = (3 - 12 t + 15 t^2 - 10 t^3 - 4 t^4) / (-1 + 2 t + t^2)^2,  t = x1 (1 - x1),

and the point behind M2 at 1 < x2 < 2 has (eqs. 6 and 9)

    q = (x2 - 1)^3 (1 + x2 + x2^2) / (x2^2 (2 - x2) (1 - x2 + x2^2)),
    Psi = (-1 - 4 x2 + 27 x2^2 - 36 x2^3 + 9 x2^4 + 18 x2^5 - 14 x2^6 + 4 x2^7)
          / (-1 + 2 x2 + x2^2 - 2 x2^3 + x2^4)^2,

with Psi in units of G(M1 + M2)/(2a): twice Rocheometry's Jacobi potential.
A component's outer point is the point behind it, so it is Seidov's x2 with
the companion as M1 and q = qi. His eq. (8) for the point behind M1 is not
used: as printed in the arXiv version (astro-ph/0311272, the one checked
here), it is the reciprocal of eq. (6) at the same x, which contradicts his
eq. (7) and his own identity q(x2) q(x3) = 1 at x3 = 1 - x2.

The formulas are exact, so each position is the root of Seidov's equation,
bisected at 40 digits, and the bound is Reference's contract bound.
"""

from __future__ import annotations

from collections.abc import Callable

import mpmath as mp

import rocheometry
from accuracy.harness.common import decade_jobs
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study

DIGITS = 40


class Seidov(Study):
    name = "seidov"
    kind = "literature"
    quantities = ("critical.l1", "critical.outer")
    domain = "qi from 1e-8 to 1e8, one per decade"
    source = "Seidov 2004, ApJ, 603, 283 (2004ApJ...603..283S; arXiv:astro-ph/0311272)"
    bound = 1e-10

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
        case = dict(qi=qi, source=self.source)
        rows: list[Row] = []

        def compare(
            quantity: str, actual: float, expected: mp.mpf, aspect: str
        ) -> None:
            error = abs(mp.mpf(actual) - expected) / abs(expected)
            rows.append(
                record(
                    f"{quantity}.{aspect}", "reference", float(error), self.bound, case
                )
            )

        l1 = g.l1()
        with mp.workdps(DIGITS):
            x1 = position(q_l1, 1 / mp.mpf(qi), 0, 1)
            compare("critical.l1", l1.position[0], x1, "position")
            compare("critical.l1", l1.potential.value, psi_l1(x1) / 2, "potential")
            # The point behind this component, seen from the companion: x2 = 1 - x.
            outer = g.outer()
            x2 = position(q_l2, mp.mpf(qi), 1, 2)
            compare("critical.outer", outer.position[0], 1 - x2, "position")
            compare(
                "critical.outer", outer.potential.value, psi_l2(x2) / 2, "potential"
            )
        return rows


def q_l1(x: mp.mpf) -> mp.mpf:
    return (1 - x) ** 3 * (1 + x + x * x) / (x**3 * (3 - 3 * x + x * x))


def psi_l1(x: mp.mpf) -> mp.mpf:
    t = x * (1 - x)
    return (3 - 12 * t + 15 * t**2 - 10 * t**3 - 4 * t**4) / (-1 + 2 * t + t * t) ** 2


def q_l2(x: mp.mpf) -> mp.mpf:
    return (x - 1) ** 3 * (1 + x + x * x) / (x * x * (2 - x) * (1 - x + x * x))


def psi_l2(x: mp.mpf) -> mp.mpf:
    numerator = (
        -1 - 4 * x + 27 * x**2 - 36 * x**3 + 9 * x**4 + 18 * x**5 - 14 * x**6 + 4 * x**7
    )
    return numerator / (-1 + 2 * x + x**2 - 2 * x**3 + x**4) ** 2


def position(
    q_of_x: Callable[[mp.mpf], mp.mpf], q: mp.mpf, low: float, high: float
) -> mp.mpf:
    """Solve q_of_x(x) = q in (low, high) by bisection at 40 digits.

    Each formula is monotonic on its interval, so the root is unique.
    """
    with mp.workdps(DIGITS):
        low, high = mp.mpf(low), mp.mpf(high)
        rising = q_of_x((3 * low + high) / 4) < q_of_x((low + 3 * high) / 4)
        while high - low > mp.mpf(10) ** (5 - DIGITS) * high:
            middle = (low + high) / 2
            if (q_of_x(middle) < q) == rising:
                low = middle
            else:
                high = middle
        return (low + high) / 2
