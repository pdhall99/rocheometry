"""Continuity in q: control-point differences along the outline study's sweep."""

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from accuracy.harness.common import SWEEP
from accuracy.harness.record import record
from accuracy.harness.study import DerivedStudy, Row


class Continuity(DerivedStudy):
    name = "continuity"
    quantities = ("outlines.shape",)
    domain = (
        "31-point log-q sweep from 1e5 to 1e8; all 15 levels; four centreline "
        "sections and five silhouette inclinations, from the outline and scalar "
        "studies' rows. Second differences are measurements, not a certificate of "
        "continuity; the fixed layout is a tested guarantee."
    )

    def derive(self, read: Callable[[str], list[Row]]) -> list[Row]:
        """Second differences in q of the outline controls the outlines study recorded."""
        radii = {
            (r["case"]["qi"], r["case"]["kind"], r["case"]["level"]): r["truth"]
            for r in read("scalars")
            if r["quantity"] == "integrals.volume.radius" and r["set"] == "fitted"
        }
        groups = defaultdict(dict)
        for row in read("outlines"):
            if "controls" not in row:
                continue
            case = row["case"]
            key = (row["quantity"], case["kind"], case["level"], case["inclination"])
            groups[key][case["qi"]] = np.asarray(row["controls"])
        rows = []
        for (quantity, kind, level, inclination), samples in groups.items():
            for lower, qi, upper in zip(SWEEP, SWEEP[1:], SWEEP[2:]):
                case = dict(
                    qi=qi,
                    kind=kind,
                    level=level,
                    inclination=inclination,
                    method=quantity.removeprefix("outlines."),
                )
                if not all(q in samples for q in (lower, qi, upper)):
                    rows.append(
                        record(
                            "outlines.shape.q_second_difference",
                            "fitted",
                            None,
                            None,
                            case,
                            note="A neighboring outline could not be measured; its construction failure is listed separately.",
                        )
                    )
                    continue
                a, b, c = (samples[q] for q in (lower, qi, upper))
                # The fixed layout is a guarantee, tested by construction; a change
                # here would be a defect, reported rather than differenced.
                if a.shape != b.shape or b.shape != c.shape:
                    rows.append(
                        record(
                            "outlines.shape.q_second_difference",
                            "fitted",
                            None,
                            None,
                            case,
                            error="the segment layout changed between neighbouring outlines",
                        )
                    )
                    continue
                # The log-q nodes are evenly spaced by 0.1 decade. Report the
                # unscaled second difference rather than inventing a bound for it.
                value = (
                    float(np.linalg.norm(a - 2 * b + c, axis=2).max())
                    / radii[qi, kind, level]
                )
                rows.append(
                    record(
                        "outlines.shape.q_second_difference",
                        "fitted",
                        value,
                        None,
                        case,
                        metric="control second difference / R_vol",
                        note="Finite-grid continuity diagnostic at 0.1-decade spacing; existing continuity/rate bounds remain unchanged.",
                    )
                )
        return rows
