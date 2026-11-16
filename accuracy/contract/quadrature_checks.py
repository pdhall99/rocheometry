"""Public quadrature positions, gradient normals and smooth test integrals."""

from crosschecks.instrumented.quadrature import inspect

import rocheometry
from accuracy.harness.common import DECADES, clear_samples, relative, select
from accuracy.harness.record import record
from accuracy.harness.study import Row

CONSISTENCY = ("area_consistency", "curvature_consistency", "gravity_consistency")


def quadrature_name(name: str) -> str:
    """Node checks belong to quadrature.nodes, reader agreement to the integrals."""
    family = "integrals" if name in CONSISTENCY else "nodes"
    return f"quadrature.{family}.{name}"


def measure(job: tuple[float, list[tuple[str, float]]]) -> list[Row]:
    qi, levels = job
    g = rocheometry.ComponentGeometry(qi)
    ref = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    rows = []
    for kind, value in levels:
        case = dict(qi=qi, kind=kind, level=value)
        try:
            s = select(g, kind, value)
            r = ref._surface_at_depth(s._depth)
            truth = inspect(r)
            for setting, surface in (("fitted", s), ("reference", r)):
                actual = inspect(surface) if setting == "fitted" else truth
                for name, metric, bound in (
                    (
                        "level_backward",
                        "exact-ray level offset / running error bound",
                        1.0,
                    ),
                    (
                        "gradient_backward",
                        "represented-gradient error / running component bound",
                        1.0,
                    ),
                    ("gravity_backward", "magnitude error / running hypot bound", 1.0),
                    ("unit", "absolute", 1e-12),
                    ("area_consistency", "relative", 1e-13),
                    ("curvature_consistency", "relative", 1e-13),
                    ("gravity_consistency", "relative", 1e-13),
                ):
                    rows.append(
                        record(
                            quadrature_name(name),
                            setting,
                            actual[name],
                            bound,
                            case,
                            metric=metric,
                        )
                    )
                for name, metric in (
                    ("displacement", "normal distance / R_vol"),
                    ("normal_angle", "radians"),
                ):
                    rows.append(
                        record(
                            quadrature_name(name),
                            setting,
                            actual[name],
                            None,
                            case,
                            metric=metric,
                            note="Forward node diagnostic; acceptance uses exact-point backward error",
                        )
                    )
                if kind == "fill" and value == 1:
                    for name, metric in (
                        ("near_displacement", "normal distance / R_vol"),
                        ("near_normal_angle", "radians"),
                    ):
                        rows.append(
                            record(
                                quadrature_name(name),
                                setting,
                                actual[name],
                                None,
                                case,
                                metric=metric,
                                note="critical-apex conditioning; nodes within 1e-3 R_vol of L1",
                            )
                        )
                for name, a in actual["smooth"].items():
                    if setting == "fitted":
                        rows.append(
                            record(
                                "quadrature.integrals." + name,
                                setting,
                                relative(a, truth["smooth"][name]),
                                1e-6,
                                case,
                                actual=a,
                                truth=truth["smooth"][name],
                            )
                        )
            if qi in DECADES:
                for p, a in ((256, 128), (512, 192)):
                    high = rocheometry.ComponentGeometry(
                        qi,
                        accuracy=rocheometry.Reference(polar_order=p, azimuth_order=a),
                    )
                    x = inspect(high._surface_at_depth(s._depth))
                    for name, v in x["smooth"].items():
                        rows.append(
                            record(
                                "quadrature.integrals." + name,
                                "reference",
                                relative(truth["smooth"][name], v),
                                1e-10,
                                {**case, "polar": p, "azimuth": a},
                                actual=truth["smooth"][name],
                                truth=v,
                            )
                        )
                    clear_samples(high)
        except Exception as exc:
            rows.append(
                record(
                    "quadrature.evaluation",
                    "reference",
                    None,
                    1e-10,
                    case,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        clear_samples(g)
        clear_samples(ref)
    return rows
