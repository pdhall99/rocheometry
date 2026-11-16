"""Normal-distance outlines, reference exact-field checks and local guarantees."""

from collections.abc import Sequence
from math import pi
from typing import Any

from crosschecks.distance.curves import controls, two_way_distance
from crosschecks.distance.outlines import normal_error

import rocheometry
from accuracy.contract.bounds import OUTLINE_MARGIN, PATH_BOUND
from accuracy.harness.common import (
    LEVEL_SMOKE,
    SWEEP,
    clear_samples,
    level_jobs,
    select,
)
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study

INCLINATIONS = (0.0, pi / 6, pi / 3, pi / 2)
SILHOUETTE_INCLINATIONS = (pi / 6, pi / 3, 15 * pi / 36, 17 * pi / 36, pi / 2)


class Outlines(Study):
    name = "outlines"
    kind = "contract"
    quantities = (
        "outlines.section",
        "outlines.section_envelope",
        "outlines.silhouette",
        "outlines.silhouette_envelope",
        "outlines.shape",
    )
    domain = "The permanent grid at every level: four centreline and nine axis-normal planes for sections, five inclinations for silhouettes. Fitted outlines are judged at the 2.25e-4 R_vol release margin under the 2.5e-4 public contract."

    def jobs(self) -> list[tuple[float, list[tuple[str, float]]]]:
        return level_jobs()

    def smoke(self) -> tuple[float, list[tuple[str, float]]]:
        return LEVEL_SMOKE

    def measure(
        self,
        job: tuple[float, list[tuple[str, float]]],
        *,
        sections: bool = True,
        silhouette_inclinations: Sequence[float] | None = None,
    ) -> list[Row]:
        qi, levels = job
        rows = []
        g = rocheometry.ComponentGeometry(qi)
        ref = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
        for kind, value in levels:
            base = dict(qi=qi, kind=kind, level=value)
            try:
                s = select(g, kind, value)
                r = ref._surface_at_depth(s._depth)
                rv = r.volume_radius()
            except Exception as exc:
                rows.append(
                    record(
                        "outlines.construction",
                        "reference",
                        None,
                        1e-10,
                        base,
                        error=str(exc),
                    )
                )
                continue
            todo = (
                [("section", label, plane) for label, plane in planes(r)]
                if sections
                else []
            )
            if sections:
                todo += [
                    ("section_envelope", label, plane) for label, plane in planes(r)
                ]
            views = (
                SILHOUETTE_INCLINATIONS
                if silhouette_inclinations is None
                else silhouette_inclinations
            )
            # Add 80-degree views, which are the hardest at this mass ratio, to the
            # permanent five-view grid. The closed critical method shares the same construction.
            if (
                silhouette_inclinations is None
                and qi == 1e8
                and (kind, value) in (("fill", 1.0), ("F", 0.99))
            ):
                views = (*views, 16 * pi / 36)
            todo += [
                (method, f"centreline-{i}", rocheometry.CentrelinePlane(i))
                for i in views
                for method in (
                    ("silhouette_envelope",)
                    if kind == "F"
                    else ("silhouette", "silhouette_envelope")
                )
            ]
            saved = {}
            for method, label, plane in todo:
                case = {
                    **base,
                    "plane": label,
                    "inclination": plane.inclination
                    if isinstance(plane, rocheometry.CentrelinePlane)
                    else None,
                    "axis": plane.axis
                    if isinstance(plane, rocheometry.AxisNormalPlane)
                    else None,
                    "offset": plane.offset
                    if isinstance(plane, rocheometry.AxisNormalPlane)
                    else None,
                }
                key = (method, label)
                note = None
                paths = {}
                for setting, surface in (("fitted", s), ("reference", r)):
                    try:
                        paths[setting] = getattr(surface, method)(plane)
                    except Exception as exc:
                        rows.append(
                            record(
                                "outlines." + method,
                                setting,
                                None,
                                PATH_BOUND[setting],
                                case,
                                metric="two-way normal distance / R_vol",
                                error=f"{type(exc).__name__}: {exc}",
                                **construction_diagnosis(
                                    exc, case, setting, method, s.potential.value
                                ),
                            )
                        )
                if len(paths) != 2:
                    continue
                a, b = paths["fitted"], paths["reference"]
                saved[key] = (a, b)
                if not a.paths and not b.paths:
                    for setting in ("fitted", "reference"):
                        rows.append(
                            record(
                                "outlines." + method,
                                setting,
                                0.0,
                                PATH_BOUND[setting],
                                case,
                                metric="two-way normal distance / R_vol",
                                note=note,
                                empty=True,
                            )
                        )
                    continue
                try:
                    if len(a.paths) != len(b.paths):
                        raise RuntimeError(
                            f"{len(a.paths)} fitted components, {len(b.paths)} in the truth"
                        )
                    distances = two_way_distance(a, b, count=101)
                    row = record(
                        "outlines." + method,
                        "fitted",
                        distances["value"] / rv,
                        PATH_BOUND["fitted"],
                        case,
                        metric="two-way normal distance / R_vol",
                        note=note,
                        forward_distance=distances["forward"] / rv,
                        reverse_distance=distances["reverse"] / rv,
                        samples_per_segment=101,
                        certified_maximum_checks=distances["checks"],
                        margin_bound=OUTLINE_MARGIN,
                    )
                    if (
                        qi in SWEEP
                        and isinstance(plane, rocheometry.CentrelinePlane)
                        and method in ("section", "silhouette_envelope")
                    ):
                        row["controls"] = controls(a).tolist()
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        record(
                            "outlines." + method,
                            "fitted",
                            None,
                            PATH_BOUND["fitted"],
                            case,
                            metric="two-way normal distance / R_vol",
                            error=f"{type(exc).__name__}: {exc}",
                            note=note,
                        )
                    )
                try:
                    env = getattr(
                        r,
                        "silhouette_envelope"
                        if method.startswith("silhouette")
                        else "section_envelope",
                    )(plane)
                    error, details, knots = normal_error(
                        r, b, plane, env, method.startswith("silhouette")
                    )
                    rows.append(
                        record(
                            "outlines." + method,
                            "reference",
                            error,
                            1e-10,
                            case,
                            metric="two-way normal distance / R_vol",
                            note=note,
                            **details,
                        )
                    )
                    rows.append(
                        record(
                            "outlines.shape.reference_knots." + method,
                            "reference",
                            knots,
                            1e-10,
                            case,
                            metric="normal distance / R_vol",
                        )
                    )
                except Exception as exc:
                    rows.append(
                        record(
                            "outlines." + method,
                            "reference",
                            None,
                            1e-10,
                            case,
                            metric="two-way normal distance / R_vol",
                            error=f"exact-field check: {type(exc).__name__}: {exc}",
                            note=note,
                        )
                    )
            clear_samples(g)
            clear_samples(ref)
        return rows


def construction_diagnosis(
    exc: Exception, case: dict[str, Any], setting: str, method: str, potential: float
) -> dict[str, str]:
    """Keep new construction defects diagnosable without a triage whitelist."""
    tb = exc.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    location = (
        f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno} ({tb.tb_frame.f_code.co_name})"
        if tb
        else "unknown"
    )
    plane = (
        f"AxisNormalPlane({case['axis']!r}, {case['offset']!r})"
        if case.get("axis")
        else f"CentrelinePlane({case['inclination']!r})"
    )
    result = dict(
        classification="defect",
        construction_location=location,
        diagnosis=f"{type(exc).__name__}: {exc} at {location}; construction failed on a valid contract-grid input.",
        reproducer=f"ComponentGeometry({case['qi']!r}, accuracy={setting.title()}()).surface(potential=Potential.jacobi({potential!r})).{method}({plane})",
    )
    return result


def planes(
    surface: rocheometry.ComponentSurfaceGeometry,
) -> list[tuple[str, rocheometry.SectionPlane]]:
    out: list[tuple[str, rocheometry.SectionPlane]] = [
        (f"centreline-{i}", rocheometry.CentrelinePlane(i)) for i in INCLINATIONS
    ]
    for axis, name in (("z", "pole"), ("y", "side"), ("x", "point")):
        offsets = (
            (0.0, 0.4 * surface.named_radius(name), 0.8 * surface.named_radius(name))
            if axis != "x"
            else (
                -0.5 * surface.named_radius("back"),
                0.0,
                0.5 * surface.named_radius("point"),
            )
        )
        out.extend(
            (f"{axis}-{j}", rocheometry.AxisNormalPlane(axis, float(x)))
            for j, x in enumerate(offsets)
        )
    return out
