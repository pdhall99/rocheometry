"""Admissibility rounding cannot alter the field-only watershed approximation."""

import json
from dataclasses import replace
from math import pi
from pathlib import Path
from typing import Any

import pytest

import rocheometry
from rocheometry import _levels as levels_module
from rocheometry._computers.defaults import _Settings, _wire_computers
from rocheometry._roche import opening as opening_module
from tests._fingerprints import digest as _digest
from tests._helpers import _Wired

_CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/gap-independence.json").read_text()
)["rows"]


@pytest.mark.parametrize("case", _CASES, ids=lambda r: f"{r['set']}-{r['qi']}")
def test_fixed_literal_level_is_independent_of_admissibility_rounding(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = opening_module._round_admissibility_gap
    settings = _Settings()
    if case["set"] == "reference":
        settings = replace(
            settings,
            polar_order=128,
            azimuth_order=64,
            watershed_tolerance=1e-13,
            fitted_selection=False,
            fill_tolerance=1e-13,
            fixed_paths=False,
            path_tolerance=1e-10,
            exact_anchor_everywhere=True,
        )

    def snapshot(nearest: bool) -> tuple[object, ...]:
        monkeypatch.setattr(
            opening_module, "_round_admissibility_gap", float if nearest else original
        )
        levels_module._admissibility_gap.cache_clear()
        # New graph nodes avoid confusing invariance with a warm cache hit.
        graph = _wire_computers(settings)
        geo = rocheometry.ComponentGeometry(case["qi"], accuracy=_Wired(graph))
        surface = geo.surface(potential=rocheometry.Potential.jacobi(case["potential"]))
        state = levels_module._levels(geo.field, graph)
        assert state.coverage_gap == float(state.l1 - state.opening)
        assert (
            state.admissibility_gap == state.coverage_gap
            if nearest
            else state.admissibility_gap < state.coverage_gap
        )
        return (
            surface._depth,
            surface.potential.value,
            surface.volume(),
            surface.area(),
            surface.integrated_mean_curvature(),
            surface.mean_gravity(),
            surface.quadrature(),
            surface.section(rocheometry.CentrelinePlane(pi / 3)),
            surface.silhouette_envelope(rocheometry.CentrelinePlane(pi / 3)),
            tuple(
                graph.watershed.rim(geo.field, surface._depth, psi)
                for psi in (0.0, pi / 8, pi / 3, pi / 2)
            ),
        )

    inward = snapshot(False)
    nearest = snapshot(True)
    assert _digest(inward) == _digest(nearest)
    assert inward[0] == case["depth"]
    levels_module._admissibility_gap.cache_clear()
