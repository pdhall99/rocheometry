"""Binary contact joins across the decade grid."""

from math import dist

from crosschecks.geometry import crossing_at_depth

import rocheometry
from accuracy.contract.outlines import INCLINATIONS
from accuracy.harness.common import decade_jobs
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class Joins(Study):
    name = "joins"
    kind = "contract"
    quantities = ("binary.join",)
    domain = "Binary contact joins at the 17 decades of qi, four fill-out factors and four inclinations."

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        binary = rocheometry.BinaryGeometry(qi)
        first, second = binary.geometry1, binary.geometry2
        rows = []
        for F in (1e-6, 0.01, 0.5, 0.99):
            surface = first.surface(fill_out_factor=F)
            depth = surface._depth
            for inclination in INCLINATIONS:
                case = dict(qi=qi, kind="F", level=F, inclination=inclination)
                try:
                    plane = rocheometry.CentrelinePlane(inclination)
                    a, b = (
                        crossing_at_depth(first, depth, plane),
                        crossing_at_depth(second, depth, plane),
                    )
                    rows.append(
                        record(
                            "binary.join",
                            "fitted",
                            dist(a, (1 - b[0], b[1])),
                            1e-12,
                            case,
                            metric="distance / separation",
                        )
                    )
                except Exception as exc:
                    rows.append(
                        record(
                            "binary.join",
                            "fitted",
                            None,
                            1e-12,
                            case,
                            metric="distance / separation",
                            error=str(exc),
                        )
                    )
        return rows
