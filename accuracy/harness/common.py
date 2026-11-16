"""Shared domains, measurement helpers and the observation store.

What a study is lives in ``accuracy.harness.study``. The helpers below build
the common domains its jobs are drawn from, select and compare surfaces, and
read and write the observation store.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import rocheometry
from accuracy.harness.cases import load
from rocheometry._computers.silhouette.paths import GaugeFixedSilhouettePathComputer
from rocheometry._computers.silhouette.projected import ProjectedField

ROOT = Path(__file__).resolve().parents[2]
_GRID = load("permanent.json")
DECADES = tuple(_GRID["decades"])
SWEEP = tuple(_GRID["sweep"])
RANDOM_Q = tuple(_GRID["random_q"])
FILLS = tuple(_GRID["fills"])
FACTORS = tuple(_GRID["factors"])
RANDOM_LEVELS = tuple((kind, level) for kind, level in _GRID["random_levels"])
#: Observation store of the current run; the runner sets it before any study.
OUTPUT = Path(
    os.environ.get("ROCHEOMETRY_STUDY_OUTPUT", ROOT / "build/accuracy/current")
)
FRESH_RUN = os.environ.get("ROCHEOMETRY_FRESH_STUDY") == "1"

try:
    COMMIT = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()
except (OSError, subprocess.CalledProcessError):
    COMMIT = "unknown"


#: The four outline quantities, judged by the two-way normal distance.
OUTLINES = tuple(
    "outlines." + m
    for m in ("section", "section_envelope", "silhouette", "silhouette_envelope")
)
#: Recorded name of each surface integral, keyed as ``physical`` returns them.
#: A row is named for its catalogue quantity, with an aspect after a dot.
INTEGRALS = {
    "volume": "integrals.volume",
    "sampled_volume": "integrals.volume.sampled",
    "volume_radius": "integrals.volume.radius",
    "area": "integrals.area",
    "area_radius": "integrals.area.radius",
    "integrated_mean_curvature": "integrals.mean_curvature",
    "mean_curvature_radius": "integrals.mean_curvature.radius",
    "mean_gravity": "integrals.mean_gravity",
}


def decade_jobs() -> list[float]:
    """One job per decade of the mass ratio, 1e-8 to 1e8."""
    return list(DECADES)


def level_jobs() -> list[tuple[float, list[tuple[str, float]]]]:
    """One job per mass ratio and level of the permanent grid."""
    return [(qi, [level]) for qi, levels in grid() for level in levels]


#: The smoke job of the level-grid studies: equal masses at fill 0.8.
LEVEL_SMOKE = (1.0, [("fill", 0.8)])


def grid() -> list[tuple[float, list[tuple[str, float]]]]:
    dense = sorted(set(DECADES + SWEEP))
    levels = [("fill", f) for f in FILLS] + [("F", f) for f in FACTORS]
    return [(q, levels) for q in dense] + [(q, list(RANDOM_LEVELS)) for q in RANDOM_Q]


def select(
    g: rocheometry.ComponentGeometry, kind: str, value: float
) -> rocheometry.ComponentSurfaceGeometry:
    """Return the surface at a volume-radius fill or, otherwise, a fill-out factor."""
    if kind == "fill":
        return g.surface(volume_radius_fill=value)
    return g.surface(fill_out_factor=value)


def clear_samples(g: rocheometry.ComponentGeometry) -> None:
    """Clear a geometry's cached samples and quadratures, so it measures afresh."""
    sample = g._computers.sample
    if hasattr(sample, "cache_clear"):
        sample.cache_clear()
    quadrature = g._computers.quadrature
    if hasattr(quadrature, "cache_clear"):
        quadrature.cache_clear()


def relative(actual: float, truth: float) -> float:
    """Return the relative error, or the absolute error against a zero truth."""
    return abs(actual / truth - 1) if truth else abs(actual)


def projected_field(
    surface: rocheometry.ComponentSurfaceGeometry, plane: rocheometry.CentrelinePlane
) -> ProjectedField:
    """Return the library's projected field for a surface's silhouette."""
    g = surface.component
    return ProjectedField(
        g.field,
        surface._depth,
        plane,
        g._computers.radius,
        g.l1(),
        g._computers.outer,
        g._computers.watershed,
    )


def silhouette_segments(g: rocheometry.ComponentGeometry) -> int:
    """Return the number of upper-half cubics in a Fitted silhouette."""
    silhouette = g._computers.silhouette
    assert isinstance(silhouette, GaugeFixedSilhouettePathComputer)
    return silhouette.segments_per_half


def script(name: str) -> str:
    """Return the source file of a study, contract or literature."""
    from accuracy.harness.run import LITERATURE

    return f"accuracy/{'literature' if name in LITERATURE else 'contract'}/{name}.py"


def summarize(
    name: str, rows: list[dict[str, Any]], seconds: float, metadata: dict[str, Any]
) -> dict[str, Any]:
    summaries = []
    keys = sorted({(r["quantity"], r["set"], r["metric"]) for r in rows})
    for quantity, setting, metric in keys:
        selected = [
            r
            for r in rows
            if (r["quantity"], r["set"], r["metric"]) == (quantity, setting, metric)
        ]
        contracted = [
            r for r in selected if r["status"] != "note" and r["bound"] is not None
        ]
        pool = contracted or selected
        measured = [
            r for r in pool if r["value"] is not None and math.isfinite(r["value"])
        ]
        worst = max(measured, key=lambda r: r["value"]) if measured else pool[0]
        # Where bounds differ between rows (literature tables), the largest
        # value need not be the one closest to its own bound.
        bounded = [r for r in measured if r["bound"]]
        closest = max(bounded, key=lambda r: r["value"] / r["bound"], default=None)
        note_values = [
            r for r in selected if r["status"] == "note" and r["value"] is not None
        ]
        note_worst = max(note_values, key=lambda r: r["value"]) if note_values else None
        summaries.append(
            dict(
                quantity=quantity,
                set=setting,
                metric=metric,
                domain=dict(
                    qi_min=min(r["case"].get("qi", 1) for r in selected),
                    qi_max=max(r["case"].get("qi", 1) for r in selected),
                    cases=len(selected),
                    measured=len(measured),
                    contracted_cases=len(contracted),
                    grid=metadata.get("grid", "see study metadata"),
                ),
                contracted_bound=worst["bound"],
                measured_worst=worst["value"],
                worst_case=worst["case"],
                closest=(
                    {k: closest[k] for k in ("value", "bound", "case")}
                    if closest
                    else None
                ),
                failures=sum(r["status"] in ("fail", "error") for r in selected),
                notes=sorted({r["note"] for r in selected if r.get("note")}),
                known_note_worst=(
                    {k: note_worst[k] for k in ("value", "case", "note")}
                    if note_worst
                    else None
                ),
                script=script(name),
                commit=COMMIT,
            )
        )
    return dict(
        study=name,
        commit=COMMIT,
        runtime_seconds=seconds,
        metadata=metadata,
        quantities=summaries,
    )


def write_study(
    name: str, rows: list[dict[str, Any]], seconds: float, metadata: dict[str, Any]
) -> None:
    if FRESH_RUN:
        for row in rows:
            if (
                row["quantity"] in OUTLINES
                and row["value"] is not None
                and row["bound"] is not None
                and not row["metric"].startswith("two-way normal distance")
            ):
                raise ValueError(
                    "Release observations must measure the two-way outline metric"
                )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = (
        "quantity",
        "set",
        "metric",
        "value",
        "bound",
        "status",
        "case",
        "error",
        "note",
        "details",
    )
    with gzip.open(OUTPUT / f"{name}.csv.gz", "wt") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        k: json.dumps(row[k], separators=(",", ":"))
                        if k == "case"
                        else row.get(k)
                        for k in fields
                        if k != "details"
                    },
                    "details": json.dumps(
                        {k: v for k, v in row.items() if k not in fields},
                        separators=(",", ":"),
                    ),
                }
            )
    (OUTPUT / f"{name}.summary.json").write_text(
        json.dumps(summarize(name, rows, seconds, metadata), indent=2, allow_nan=False)
        + "\n"
    )


def read_rows(name: str, directory: Path | None = None) -> list[dict[str, Any]]:
    """Read typed measurement rows from the portable CSV representation."""
    rows = []
    directory = OUTPUT if directory is None else directory
    path = directory / f"{name}.csv.gz"
    opener = (
        gzip.open(path, "rt") if path.exists() else (directory / f"{name}.csv").open()
    )
    with opener as file:
        for row in csv.DictReader(file):
            details = json.loads(row.pop("details") or "{}")
            row["case"] = json.loads(row["case"])
            for key in ("value", "bound"):
                row[key] = float(row[key]) if row[key] else None
            for key in ("error", "note"):
                row[key] = row[key] or None
            rows.append({**row, **details})
    return rows
