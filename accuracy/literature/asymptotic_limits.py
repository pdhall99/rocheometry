"""The lobe volume radius and its slope at both ends of the mass-ratio range.

Guillochon & Prochaska (2026) derive the asymptotic expansions of the
volume-equivalent lobe radius R_L/a in q_E = M1/M2 (our qi). In the Hill
limit, with t = qi^(1/3) (their eq. hillexp),

    R_L/a = C t (1 + a1 t + a3/2 t^(3/2) + a2 t^2 + O(t^(5/2))),

and in the dominant-star limit (their eq. logexp)

    R_L/a = R0 - K ln(qi) / qi^(2/3) - c2 / qi^(2/3) + O(ln(qi) / qi).

R0 and K are closed forms; C, a1, a3/2, a2 and c2 are their quadratures to
the digits printed. The slope d ln R_L / d ln qi is the derivative of the
same expansion.

The bound is twice the order of the first omitted term: t^(5/2) as qi -> 0
and ln(qi)/qi as qi -> infinity, relative for the radius and absolute for
the slope, which tends to 1/3 and to 0. This compares Rocheometry with the
expansions' coefficients and structure; the expansions are not accurate
enough to measure Rocheometry's own error.

Two more jobs compare the endpoint limits the lobe volume radius fit is built
on (``fits/lobe_volume_radius/config.yaml``) with C and R0, against
Reference's bound.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import rocheometry
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study

# The Hill limit: C = kappa 3^(-1/3), and the coefficients of t, t^(3/2), t^2.
C = 0.493306231152633
A1 = -0.26475047
A32 = 0.20568147
A2 = -0.348082
# The dominant-star limit: Kopal's critical rotating star, and the log term.
R0 = (3 * math.sqrt(3) - 4 + 3 * math.log(6 - 3 * math.sqrt(3))) ** (1 / 3)
K = 3 ** (-1 / 6) / (2 * R0**2)
C2 = -0.983192566


class AsymptoticLimits(Study):
    name = "asymptotic_limits"
    kind = "literature"
    quantities = ("lobe.volume_radius", "lobe.slope")
    domain = (
        "qi from 1e-8 to 1e-4 and from 1e4 to 1e8, one per decade, and the fit's "
        "two endpoint limits"
    )
    #: A manuscript, without a bibcode: github.com/guillochon/roche-lobe at 1c8e174.
    source = "Guillochon & Prochaska 2026 (manuscript), eqs. hillexp and logexp"

    def jobs(self) -> list[float | dict[str, str]]:
        decades = [10.0**k for k in (-8, -7, -6, -5, -4, 4, 5, 6, 7, 8)]
        return decades + [dict(limit=name) for name in LIMITS]

    def smoke(self) -> float:
        return 1e-6

    def measure(self, qi: float | dict[str, str]) -> list[Row]:
        if isinstance(qi, dict):  # an endpoint limit of the fit
            name = qi["limit"]
            exact = LIMITS[name]
            return [
                record(
                    "lobe.volume_radius.limit",
                    "fitted",
                    abs(fit_limit(name) - exact) / exact,
                    1e-10,
                    dict(limit=name, source=self.source),
                )
            ]
        g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
        radius, slope, scale = hill(qi) if qi < 1 else dominant(qi)
        case = dict(qi=qi, source=self.source)
        actual = g.lobe_volume_radius()
        actual_slope = g.lobe_volume_radius_slope()
        return [
            record(
                "lobe.volume_radius",
                "reference",
                abs(actual - radius) / radius,
                2 * scale,
                case,
                actual=actual,
                expansion=radius,
            ),
            record(
                "lobe.slope",
                "reference",
                abs(actual_slope - slope),
                2 * scale,
                case,
                metric="absolute",
                actual=actual_slope,
                expansion=slope,
            ),
        ]


def hill(qi: float) -> tuple[float, float, float]:
    """(R_L/a, slope, bound scale) as qi -> 0."""
    t = qi ** (1 / 3)
    series = 1 + A1 * t + A32 * t**1.5 + A2 * t**2
    slope = (1 + (A1 * t + 1.5 * A32 * t**1.5 + 2 * A2 * t**2) / series) / 3
    return C * t * series, slope, t**2.5


def dominant(qi: float) -> tuple[float, float, float]:
    """(R_L/a, slope, bound scale) as qi -> infinity."""
    log, power = math.log(qi), qi ** (-2 / 3)
    radius = R0 - K * log * power - C2 * power
    slope = (2 / 3 * (K * log + C2) - K) * power / radius
    return radius, slope, log / qi


FIT_CONFIG = Path(__file__).parents[2] / "fits/lobe_volume_radius/config.yaml"
LIMITS = {"hill_limit": C, "infinite_limit": R0}


def fit_limit(name: str) -> float:
    """Read an endpoint limit from the fit's config (a plain ``name: value`` line)."""
    text = FIT_CONFIG.read_text(encoding="utf-8")
    match = re.search(rf"^{name}:\s*(\S+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} is not in {FIT_CONFIG}")
    return float(match.group(1))
