"""Reference scalars on the frozen near-saddle surfaces.

Every scalar is computed at 128x64 and 512x192 and compared with 1024x192 at
the same depth.
"""

from typing import Any

import rocheometry
from accuracy.contract.bounds import scalar_bound
from accuracy.contract.scalars import physical
from accuracy.harness.cases import load
from accuracy.harness.common import INTEGRALS, clear_samples, relative
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class SaddleQuadrature(Study):
    name = "saddle_quadrature"
    kind = "contract"
    quantities = (
        "integrals.volume",
        "integrals.area",
        "integrals.mean_curvature",
        "integrals.mean_gravity",
    )
    domain = "The frozen near-saddle surfaces (saddle.json), at 128x64 and 512x192 against 1024x192."

    def jobs(self) -> list[dict[str, Any]]:
        return cases()

    def smoke(self) -> dict[str, Any]:
        return cases()[0]

    def measure(self, job: dict[str, Any] | tuple[float, Any]) -> list[Row]:
        if not isinstance(job, dict):
            qi, _ = job
            return [
                row
                for case in cases()
                if case["qi"] == qi
                for row in self.measure(case)
            ]
        measured = []
        for p, a in ((128, 64), (512, 192), (1024, 192)):
            g = rocheometry.ComponentGeometry(
                job["qi"],
                accuracy=rocheometry.Reference(polar_order=p, azimuth_order=a),
            )
            s = g._surface_at_depth(job["depth"])
            measured.append(physical(s))
            clear_samples(g)
        rows = []
        for order, values in zip(((128, 64), (512, 192)), measured[:2], strict=True):
            p, a = order
            for name, value in values.items():
                rows.append(
                    record(
                        INTEGRALS[name],
                        "reference",
                        relative(value, measured[-1][name]),
                        scalar_bound(
                            "reference", INTEGRALS[name], polar_order=p, azimuth_order=a
                        ),
                        dict(job, measured_polar=p, measured_azimuth=a),
                        actual=value,
                        truth=measured[-1][name],
                        metric="relative at 512x192" if p == 512 else "relative",
                        measured_order=dict(polar=p, azimuth=a),
                        truth_order=dict(polar=1024, azimuth=192),
                    )
                )
        return rows


def cases() -> list[dict[str, Any]]:
    return load("saddle.json")["cases"]
