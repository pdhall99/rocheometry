"""The lobe volume radius fit: built by ``build.py``, judged on its frozen holdout.

The holdout is 253 mass ratios over 1e-12 <= qi <= 1e12, chosen at interval
midpoints of three spacings (log qi, the sixth-root coordinate and the
cube-root coordinate) so that none coincides with a training node, with
``Reference()`` radii recorded when it was drawn.
"""

from __future__ import annotations

import json
from typing import Any

from fits.fit import Fit, Holdout
from fits.lobe_volume_radius.build import build as build_artefact


class LobeVolumeRadius(Fit):
    name = "lobe_volume_radius"

    def build(
        self, config: dict[str, Any], *, workers: int = 1
    ) -> tuple[str, dict[str, Any]]:
        return build_artefact(config, workers=workers)

    def holdout(self) -> list[Holdout]:
        """Measure the packaged model's worst relative error on the holdout."""
        from rocheometry._computers.packaged import packaged_lobe_volume_radius_computer
        from rocheometry._roche.field import RocheField

        model = packaged_lobe_volume_radius_computer()
        data = json.loads((self.folder / "holdout.json").read_text())
        worst, worst_qi = 0.0, None
        for qi_hex, truth_hex in data["cases"]:
            qi, truth = float.fromhex(qi_hex), float.fromhex(truth_hex)
            error = abs(model(RocheField(qi)) / truth - 1)
            if error > worst:
                worst, worst_qi = error, qi
        return [
            Holdout(
                "lobe.volume_radius.fast",
                worst,
                self.config()["accuracy_bound"],
                len(data["cases"]),
                dict(qi=worst_qi),
            )
        ]

    def draw_holdout(self, count: int = 85, exponent: int = 12) -> dict[str, Any]:
        """Draw the holdout once; it is frozen from then on and never redrawn."""
        from datetime import date

        from fits.lobe_volume_radius.build import _radius

        order = self.config()["reference"]
        edges = [-exponent + 2 * exponent * i / count for i in range(count + 1)]
        log_mid = [10 ** ((a + b) / 2) for a, b in zip(edges, edges[1:], strict=False)]

        def compact(power: float) -> list[float]:
            lo = 10.0 ** (-exponent / power)
            hi = 10.0 ** (exponent / power)
            a, b = lo / (1 + lo), hi / (1 + hi)
            mids = [a + (b - a) * (i + 0.5) / count for i in range(count)]
            return [(m / (1 - m)) ** power for m in mids]

        qis = sorted(set(log_mid + compact(6) + compact(3) + [1.0]))
        cases = [
            [
                qi.hex(),
                _radius((qi, order["polar_order"], order["azimuth_order"])).hex(),
            ]
            for qi in qis
        ]
        return dict(
            description="Reference() lobe volume radii at mass ratios between training nodes.",
            drawn=date.today().isoformat(),
            reference=order,
            fields=["qi", "radius"],
            cases=cases,
        )
