"""Independent 40-digit ray roots and exact-tracer cut intersections."""

from crosschecks.radii import exact_radius

import rocheometry
from accuracy.contract.scalars import DIRECTIONS
from accuracy.harness.common import (
    FACTORS,
    FILLS,
    clear_samples,
    decade_jobs,
    relative,
    select,
)
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class Radii(Study):
    name = "radii"
    kind = "contract"
    quantities = (
        "radii.named",
        "radii.direction",
    )
    domain = "Named and directional radii of Reference() at the 17 decades of qi and every grid level, against independent 40-digit ray roots."

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
        rows = []
        for kind, value in [("fill", f) for f in FILLS] + [("F", f) for f in FACTORS]:
            try:
                s = select(g, kind, value)
            except Exception as exc:
                rows.append(
                    record(
                        "radii.construction",
                        "reference",
                        None,
                        1e-10,
                        dict(qi=qi, kind=kind, level=value),
                        error=str(exc),
                    )
                )
                continue
            errors = []
            for index, d in enumerate(DIRECTIONS):
                case = dict(qi=qi, kind=kind, level=value, direction=d)
                name = (
                    ("point", "back", "side", "pole")[index]
                    if index < 4
                    else "direction_set"
                )
                try:
                    truth = exact_radius(s, d)
                    actual = s.radius(d)
                    error = relative(actual, truth)
                    errors.append((error, d))
                    if index < 4:
                        rows.append(
                            record(
                                "radii.named." + name,
                                "reference",
                                error,
                                1e-10,
                                case,
                                actual=actual,
                                truth=truth,
                            )
                        )
                except Exception as exc:
                    rows.append(
                        record(
                            "radii.named." + name,
                            "reference",
                            None,
                            1e-10,
                            case,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            if errors:
                error, d = max(errors)
                rows.append(
                    record(
                        "radii.direction",
                        "reference",
                        error,
                        1e-10,
                        dict(qi=qi, kind=kind, level=value, direction=d),
                    )
                )
            clear_samples(g)
        return rows
