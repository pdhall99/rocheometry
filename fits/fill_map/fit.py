"""The fill map: built by ``build.py``, judged on its frozen holdout.

The holdout is 450 Reference measurements at random (qi, fill), seed 11,
drawn independently of the training nodes. For each, the packaged map gives a
potential; the check measures that surface's fill with Reference() and
compares both the potential and the fill with the holdout.
"""

from __future__ import annotations

import json
from typing import Any

from fits.fill_map.build import build as build_artefact
from fits.fit import Fit, Holdout


class FillMap(Fit):
    name = "fill_map"

    def build(
        self, config: dict[str, Any], *, workers: int = 1
    ) -> tuple[str, dict[str, Any]]:
        return build_artefact(config, workers=workers)

    def holdout(self) -> list[Holdout]:
        """Measure the map's worst potential and fill errors on the holdout."""
        from rocheometry import ComponentGeometry, Fitted, Reference
        from rocheometry._computers.packaged import load_packaged_surface_potential_map
        from rocheometry._computers.surface_potential.fitted.computer import (
            FittedSurfacePotentialComputer,
        )
        from rocheometry._roche.field import RocheField

        graph = Fitted()._computers()
        forward = FittedSurfacePotentialComputer(
            model=load_packaged_surface_potential_map(),
            l1_computer=graph.l1,
            lobe_volume_radius_computer=graph.selection.lobe_volume_radius_computer,  # type: ignore[attr-defined]
        )
        rows = json.loads((self.folder / "holdout.json").read_text())["samples"]
        worst_potential = worst_fill = 0.0
        at_potential = at_fill = None
        for qi, fill, potential, _ in rows:
            field = RocheField(qi)
            depth = forward(field, fill)
            level = field._level_value(forward.l1_computer(field).position[0], depth)
            if abs(level / potential - 1) >= worst_potential:
                worst_potential, at_potential = abs(level / potential - 1), (qi, fill)
            measured = (
                ComponentGeometry(qi, accuracy=Reference())
                ._surface_at_depth(depth)
                .volume_radius_fill()
            )
            if abs(measured / fill - 1) >= worst_fill:
                worst_fill, at_fill = abs(measured / fill - 1), (qi, fill)
        bound = self.config()["accuracy_bound"]
        return [
            Holdout(
                quantity,
                value,
                bound,
                len(rows),
                dict(qi=at[0], fill=at[1]) if at else {},
            )
            for quantity, value, at in (
                ("selection.by_fill.potential", worst_potential, at_potential),
                ("selection.by_fill", worst_fill, at_fill),
            )
        ]
