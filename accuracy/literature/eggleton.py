"""Lobe volume radius against Eggleton's (1983) formula, from qi = 1e-8 to 1e8.

Eggleton fitted r_L = 0.49 q^(2/3) / (0.6 q^(2/3) + ln(1 + q^(1/3))) to
better than 1% for all mass ratios, with q the lobe's own mass over its
companion's (our qi). That is coarse, but it is the only published value
beyond the tables' q = 100, so it catches a gross error at the extremes,
such as a flipped mass ratio or a wrong limit.
"""

from __future__ import annotations

import math

import rocheometry
from accuracy.harness.common import decade_jobs
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class Eggleton(Study):
    name = "eggleton"
    kind = "literature"
    quantities = ("lobe.volume_radius",)
    domain = "qi from 1e-8 to 1e8, one per decade, against a 1% bound"
    source = "Eggleton 1983, ApJ, 268, 368 (1983ApJ...268..368E)"
    bound = 0.01

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        actual = rocheometry.ComponentGeometry(
            qi, accuracy=rocheometry.Reference()
        ).lobe_volume_radius()
        published = eggleton(qi)
        return [
            record(
                "lobe.volume_radius",
                "reference",
                abs(actual - published) / published,
                self.bound,
                dict(qi=qi, source=self.source),
                actual=actual,
                published=published,
            )
        ]


def eggleton(qi: float) -> float:
    a = qi ** (2.0 / 3.0)
    return 0.49 * a / (0.6 * a + math.log1p(qi ** (1.0 / 3.0)))
