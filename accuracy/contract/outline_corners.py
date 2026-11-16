"""Fitted silhouettes on the frozen holdout of views with an interior corner."""

from typing import Any

from accuracy.contract.bounds import OUTLINE_BOUND, OUTLINE_MARGIN
from accuracy.contract.outline_views import evaluate
from accuracy.harness.cases import grouped, load, views
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study

SOURCE = "holdout-outline-corners.json"
SEED = load(SOURCE)["seed"]


class OutlineCorners(Study):
    name = "outline_corners"
    kind = "contract"
    quantities = (
        "outlines.silhouette_envelope",
        "outlines.shape",
    )
    domain = "The frozen corner holdout (holdout-outline-corners.json): views whose silhouette has an interior corner."

    def jobs(self) -> list[tuple[float, list[Any]]]:
        return grouped(views(SOURCE))

    def smoke(self) -> tuple[float, list[Any]]:
        qi, levels = self.jobs()[0]
        return (qi, levels[:1])

    def measure(self, job: tuple[float, list[Any]]) -> list[Row]:
        qi, levels = job
        cases = views(SOURCE)
        return [
            r
            for c in cases
            if c["qi"] == qi and (c["kind"], c["level"]) in levels
            for r in observations(evaluate(c))
        ]


def observations(row: dict[str, Any]) -> list[Row]:
    case = {**row["case"], "seed": SEED, "plane": "corner-holdout"}
    return [
        record(
            "outlines.silhouette_envelope",
            "fitted",
            row["error"],
            OUTLINE_BOUND,
            case,
            metric="two-way normal distance / R_vol",
            holdout_margin=OUTLINE_MARGIN,
            worst_point=row.get("worst_point"),
            corner_error=row["corner_error"],
            diagnosis=(
                f"Upper-half segment {row['worst_segment']}; the corner itself has distance {row['corner_error']:.6g} R_vol. Fixed-gauge front-shoulder interpolation miss."
                if row["error"] > OUTLINE_BOUND
                else None
            ),
        ),
        record(
            "outlines.shape.corner_rounding",
            "fitted",
            row["corner_error"],
            None,
            case,
            metric="distance / R_vol",
            junctions=len(row["corners"]),
        ),
    ]
