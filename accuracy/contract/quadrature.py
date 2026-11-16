"""Quadrature nodes and smooth integrals, including the high-order node checks.

The base checks in ``quadrature_checks.py`` compute backward node results at
256x128 and 512x192 while checking smooth-integral convergence. This study
gates those high-order node observations as well as the default-order ones.
"""

from typing import Any
from unittest.mock import patch

import rocheometry
from accuracy.contract import quadrature_checks
from accuracy.harness.common import LEVEL_SMOKE, level_jobs
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class Quadrature(Study):
    name = "quadrature"
    kind = "contract"
    quantities = (
        "quadrature.nodes",
        "quadrature.integrals",
    )
    domain = "The permanent grid at every level; high-order Reference node checks at the 17 decades."

    def jobs(self) -> list[tuple[float, list[tuple[str, float]]]]:
        return level_jobs()

    def smoke(self) -> tuple[float, list[tuple[str, float]]]:
        return LEVEL_SMOKE

    def measure(self, job: tuple[float, list[tuple[str, float]]]) -> list[Row]:
        _, ((kind, level),) = job
        extra = []
        original = quadrature_checks.inspect

        def inspect(surface: rocheometry.ComponentSurfaceGeometry) -> dict[str, Any]:
            result = original(surface)
            accuracy = surface.component.accuracy
            if (
                isinstance(accuracy, rocheometry.Reference)
                and accuracy.polar_order > 128
            ):
                case = dict(
                    qi=surface.component.qi,
                    kind=kind,
                    level=level,
                    depth=surface._depth,
                    polar=accuracy.polar_order,
                    azimuth=accuracy.azimuth_order,
                )
                for name, metric, bound in (
                    (
                        "level_backward",
                        "exact-ray level offset / running error bound",
                        1.0,
                    ),
                    (
                        "gradient_backward",
                        "represented-gradient error / running component bound",
                        1.0,
                    ),
                    ("gravity_backward", "magnitude error / running hypot bound", 1.0),
                    ("unit", "absolute", 1e-12),
                ):
                    extra.append(
                        record(
                            "quadrature.nodes." + name,
                            "reference",
                            result[name],
                            bound,
                            case,
                            metric=metric,
                        )
                    )
            return result

        with patch.object(quadrature_checks, "inspect", inspect):
            rows = quadrature_checks.measure(job)
        return rows + extra
