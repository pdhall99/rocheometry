"""Fitted silhouettes on the two frozen random-view silhouette holdouts."""

from typing import Any

from accuracy.contract.bounds import OUTLINE_BOUND, OUTLINE_MARGIN
from accuracy.contract.outline_views import evaluate
from accuracy.harness.cases import OUTLINE_HOLDOUTS, grouped, load, views
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study

SOURCES = OUTLINE_HOLDOUTS


class OutlineHoldouts(Study):
    name = "outline_holdouts"
    kind = "contract"
    quantities = (
        "outlines.silhouette_envelope",
        "outlines.shape",
    )
    domain = "The two frozen random-view silhouette holdouts."

    def jobs(self) -> list[tuple[float, list[Any]]]:
        return grouped(c for source in SOURCES for c in views(source))

    def smoke(self) -> tuple[float, list[Any]]:
        qi, levels = self.jobs()[0]
        return (qi, levels[:1])

    def measure(self, job: tuple[float, list[Any]]) -> list[Row]:
        qi, levels = job
        rows: list[Row] = []
        for case in cases():
            if case["qi"] != qi or (case["kind"], case["level"]) not in levels:
                continue
            r = evaluate(case)
            rows += [
                record(
                    "outlines.silhouette_envelope",
                    "fitted",
                    r["error"],
                    OUTLINE_BOUND,
                    case,
                    metric="two-way normal distance / R_vol",
                    margin_bound=OUTLINE_MARGIN,
                    corner_count=len(r["corners"]),
                ),
                record(
                    "outlines.shape.corner_rounding",
                    "fitted",
                    r["corner_error"],
                    None,
                    case,
                    metric="distance / R_vol",
                ),
            ]
        return rows


def cases() -> list[dict[str, Any]]:
    result = []
    for name in SOURCES:
        seed = load(name)["seed"]
        result.extend(c | dict(seed=seed, source=name) for c in views(name))
    return result
