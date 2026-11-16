"""One source of contracted observations for the map, failures and test cases."""

from __future__ import annotations

import json
from collections.abc import Iterable

from accuracy.contract.bounds import curvature_contract_row
from accuracy.harness.common import DECADES
from accuracy.harness.study import Row

FORWARD_NODE_DIAGNOSTICS = frozenset(
    (
        "quadrature.nodes.displacement",
        "quadrature.nodes.normal_angle",
        "quadrature.nodes.near_displacement",
        "quadrature.nodes.near_normal_angle",
    )
)


def case_key(row: Row) -> str:
    return json.dumps(
        [row["quantity"], row["set"], row["metric"], row["case"]],
        sort_keys=True,
        separators=(",", ":"),
    )


def contracted_rows(name: str, rows: Iterable[Row]) -> list[Row]:
    rows = [curvature_contract_row(row) for row in rows]
    # Corner observations are diagnostics: a corner contributes to the same
    # two-way outline distance and bound, not to a separate target.
    rows = [
        {
            **r,
            "bound": None,
            "status": "note",
            "note": "Corner-to-curve distance is included in the two-way outline bound; the separate target is retired.",
        }
        if r["quantity"] == "outlines.shape.corner_rounding"
        else r
        for r in rows
    ]
    rows = [
        {
            **row,
            "bound": None,
            "status": "note",
            "note": "Forward node diagnostic; backward measurements at the exact points determine node acceptance.",
        }
        if row["quantity"] in FORWARD_NODE_DIAGNOSTICS
        else row
        for row in rows
    ]
    if name != "scalars":
        return rows
    result = []
    for r in rows:
        quantity = r["quantity"]
        # The quadrature has its own study; scalars keeps only Gauss-Bonnet.
        if (
            quantity.startswith("quadrature.")
            and quantity != "quadrature.integrals.gauss_bonnet"
        ):
            continue
        # Reference compared with itself is trivially exact (this includes
        # integrals.fill.at_potential, always 0); its accuracy is the
        # convergence to higher orders, renamed below.
        if r["set"] == "reference" and (
            (
                quantity.startswith("integrals.")
                and quantity != "integrals.mean_gravity.gauss_law"
            )
            or quantity.startswith("radii.")
            or quantity.startswith("lobe.")
        ):
            continue
        result.append(r)
        if (
            quantity.startswith("reference_convergence.")
            and not quantity.startswith("reference_convergence.256_to_512.")
            and not quantity.endswith("evaluation")
        ):
            result.append(
                {**r, "quantity": quantity.removeprefix("reference_convergence.")}
            )
    return result


def test_cases(observations: Iterable[Row], failures: Iterable[Row]) -> list[Row]:
    ids = {case_key(r): r["id"] for r in failures}
    selected = {}
    for row in observations:
        if row["quantity"] in FORWARD_NODE_DIAGNOSTICS:
            continue
        key = case_key(row)
        if key not in ids and (
            row["status"] == "note"
            or row["quantity"].startswith("reference_convergence.")
        ):
            continue
        if key not in ids and (
            (
                row["case"].get("qi") not in DECADES
                and row.get("study")
                not in ("heavy_silhouette_dense", "heavy_silhouette_confirmation")
            )
            or row["bound"] is None
        ):
            continue
        value = {
            **{
                k: row[k]
                for k in (
                    "quantity",
                    "set",
                    "metric",
                    "case",
                    "value",
                    "bound",
                    "status",
                    "error",
                    "study",
                )
            },
            "failure_id": ids.get(key),
        }
        if key not in selected or row["status"] in ("fail", "error"):
            selected[key] = value
    return sorted(
        selected.values(),
        key=lambda r: (
            r["study"],
            r["case"].get("qi", 1),
            r["case"].get("kind", ""),
            r["case"].get("level", 0),
            case_key(r),
        ),
    )
