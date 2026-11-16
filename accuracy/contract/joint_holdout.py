"""The joint holdout: 400 random views over scalars, outlines and junctions.

Rules and seed were fixed before drawing. Scalars use 512x192 truth at the
same depth; Reference outlines use the exact field. Both outline directions
and exact junction-to-curve distances share one bound.
"""

import math
from typing import Any
from unittest.mock import patch

from crosschecks.distance.curves import two_way_distance
from crosschecks.distance.outlines import normal_error
from crosschecks.instrumented.quadrature import inspect as inspect_quadrature

import rocheometry
from accuracy.contract.bounds import OUTLINE_BOUND, OUTLINE_MARGIN, scalar_bound
from accuracy.contract.scalars import DIRECTIONS, physical
from accuracy.harness.cases import load
from accuracy.harness.common import INTEGRALS, clear_samples
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study
from rocheometry._computers.silhouette import limb
from rocheometry._levels import _levels

MARGIN = OUTLINE_MARGIN


class JointHoldout(Study):
    name = "joint_holdout"
    kind = "contract"
    quantities = (
        "lobe.volume_radius",
        "lobe.slope",
        "lobe.max_fill",
        "selection.by_fill",
        "selection.by_fill_out_factor",
        "integrals.volume",
        "integrals.fill",
        "integrals.area",
        "integrals.mean_curvature",
        "integrals.mean_gravity",
        "radii.named",
        "radii.direction",
        "quadrature.nodes",
        "quadrature.integrals",
        "outlines.section_envelope",
        "outlines.silhouette_envelope",
    )
    domain = "The frozen joint holdout (holdout-joint.json): 400 random cases in four level populations, every quantity at once."

    def jobs(self) -> list[dict[str, Any]]:
        return load("holdout-joint.json")["cases"]

    def smoke(self) -> dict[str, Any]:
        return self.jobs()[0]

    def measure(self, case: dict[str, Any]) -> list[Row]:
        rows = []
        g = rocheometry.ComponentGeometry(case["qi"])
        reference = rocheometry.ComponentGeometry(
            case["qi"], accuracy=rocheometry.Reference()
        )
        high = rocheometry.ComponentGeometry(
            case["qi"],
            accuracy=rocheometry.Reference(polar_order=512, azimuth_order=192),
        )
        try:
            fitted = select(g, case)
            exact = reference._surface_at_depth(fitted._depth)
            truth_surface = high._surface_at_depth(fitted._depth)
            truth = physical(truth_surface)
            true_fill = truth_surface.volume_radius_fill()
            true_quadrature = truth_surface.quadrature()
            smooth_truth = {
                "x2": math.fsum(
                    w * p[0] ** 2
                    for w, p in zip(true_quadrature.weights, true_quadrature.positions)
                ),
                "z2": math.fsum(
                    w * p[2] ** 2
                    for w, p in zip(true_quadrature.weights, true_quadrature.positions)
                ),
                "gravity4": math.fsum(
                    w * v**4
                    for w, v in zip(true_quadrature.weights, true_quadrature.gravities)
                ),
            }
            # Release the high-order node tuples before constructing paths.
            del true_quadrature
            clear_samples(high)
            rv = truth["volume_radius"]
            for setting, geometry, surface, bound in (
                ("fitted", g, fitted, 1e-6),
                ("reference", reference, exact, 1e-10),
            ):
                values = physical(surface)
                for name, value in values.items():
                    rows.append(
                        record(
                            INTEGRALS[name],
                            setting,
                            relative(value, truth[name]),
                            scalar_bound(setting, INTEGRALS[name]),
                            case,
                            actual=value,
                            truth=truth[name],
                        )
                    )
                rows.append(
                    record(
                        "integrals.fill",
                        setting,
                        relative(surface.volume_radius_fill(), true_fill),
                        bound,
                        case,
                        metric="relative",
                        actual=surface.volume_radius_fill(),
                        truth=true_fill,
                    )
                )
                mass = geometry.field.component_mass
                total = math.fsum(
                    (geometry.field.component_mass, geometry.field.companion_mass)
                )
                gravity = (4 * math.pi * mass - 2 * total * values["volume"]) / values[
                    "area"
                ]
                rows.append(
                    record(
                        "integrals.mean_gravity.gauss_law",
                        setting,
                        relative(values["mean_gravity"], gravity),
                        bound,
                        case,
                    )
                )
                radii = [
                    relative(surface.radius(d), truth_surface.radius(d))
                    for d in DIRECTIONS
                ]
                rows.append(
                    record(
                        "radii.direction",
                        setting,
                        max(radii),
                        bound,
                        case,
                        direction=DIRECTIONS[
                            max(range(len(radii)), key=radii.__getitem__)
                        ],
                    )
                )
                for name in ("point", "back", "side", "pole"):
                    rows.append(
                        record(
                            "radii.named." + name,
                            setting,
                            relative(
                                surface.named_radius(name),
                                truth_surface.named_radius(name),
                            ),
                            bound,
                            case,
                        )
                    )
                q = inspect_quadrature(surface)
                for name in ("level_backward", "gradient_backward", "gravity_backward"):
                    rows.append(
                        record(
                            "quadrature.nodes." + name,
                            setting,
                            q[name],
                            1.0,
                            case,
                            metric="exact-ray/float-gradient error / running bound",
                        )
                    )
                for name in ("displacement", "normal_angle"):
                    rows.append(
                        record(
                            "quadrature.nodes." + name,
                            setting,
                            q[name],
                            None,
                            case,
                            note="Forward node diagnostic; backward acceptance is separate",
                        )
                    )
                for name, value in q["smooth"].items():
                    rows.append(
                        record(
                            "quadrature.integrals." + name,
                            setting,
                            relative(value, smooth_truth[name]),
                            bound,
                            case,
                        )
                    )
                for name in (
                    "area_consistency",
                    "curvature_consistency",
                    "gravity_consistency",
                ):
                    rows.append(
                        record(
                            "quadrature.integrals." + name,
                            setting,
                            q[name],
                            1e-13,
                            case,
                        )
                    )
                rows.append(
                    record(
                        "quadrature.nodes.unit",
                        setting,
                        q["unit"],
                        1e-12,
                        case,
                        metric="absolute",
                    )
                )
                selected = select(geometry, case)
                if case["kind"] == "fill":
                    error = abs(selected.volume_radius_fill() - case["level"])
                    selection_bound = 2e-6 if setting == "fitted" else 1e-13
                    quantity = "selection.by_fill"
                elif case["kind"] == "F":
                    error = abs(selected.fill_out_factor() - case["level"])
                    selection_bound = bound
                    quantity = "selection.by_fill_out_factor"
                else:
                    error = abs(selected._depth / fitted._depth - 1)
                    selection_bound = bound
                    quantity = "selection.by_fill.depth"
                rows.append(
                    record(
                        quantity,
                        setting,
                        error,
                        selection_bound,
                        case,
                        metric="absolute",
                    )
                )
                for name, method in (
                    ("lobe.volume_radius.fast", lambda x: x.lobe_volume_radius()),
                    ("lobe.slope", lambda x: x.lobe_volume_radius_slope()),
                    ("lobe.max_fill", lambda x: x.max_volume_radius_fill()),
                ):
                    actual, expected = method(geometry), method(high)
                    rows.append(
                        record(
                            name,
                            setting,
                            abs(actual - expected)
                            if name == "lobe.slope"
                            else relative(actual, expected),
                            bound,
                            case,
                            metric="absolute" if name == "lobe.slope" else "relative",
                        )
                    )
            plane = rocheometry.CentrelinePlane(case["inclination"])
            for method in ("section_envelope", "silhouette_envelope"):
                junctions = []
                original = limb._junction

                def capture(*args: Any, **kwargs: Any) -> limb._Junction:
                    j = original(*args, **kwargs)
                    junctions.append(j)
                    return j

                try:
                    with patch.object(limb, "_junction", capture):
                        reference_path = getattr(exact, method)(plane)
                    fitted_path = getattr(fitted, method)(plane)
                    distances = two_way_distance(
                        fitted_path,
                        reference_path,
                        count=101,
                        corners=[j.point for j in junctions],
                    )
                    value = distances["value"] / rv
                    rows.append(
                        record(
                            "outlines." + method,
                            "fitted",
                            value,
                            OUTLINE_BOUND,
                            case,
                            metric="two-way normal distance / R_vol",
                            margin=MARGIN,
                            margin_pass=value <= MARGIN,
                            forward=distances["forward"] / rv,
                            reverse=distances["reverse"] / rv,
                            corner=distances["corner"] / rv,
                            junctions=len(junctions),
                        )
                    )
                    value, details, _ = normal_error(
                        exact,
                        reference_path,
                        plane,
                        reference_path,
                        method.startswith("silhouette"),
                        samples_per_segment=51,
                    )
                    rows.append(
                        record(
                            "outlines." + method,
                            "reference",
                            value,
                            1e-10,
                            case,
                            metric="two-way normal distance / R_vol",
                            **details,
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
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        except Exception as exc:
            rows.append(
                record(
                    "joint_holdout.evaluation",
                    "both",
                    None,
                    0.0,
                    case,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            for geometry in (g, reference, high):
                clear_samples(geometry)
        return rows


def select(
    g: rocheometry.ComponentGeometry, case: dict[str, Any]
) -> rocheometry.ComponentSurfaceGeometry:
    if case["kind"] == "depth_fraction":
        return g._surface_at_depth(
            case["level"] * _levels(g.field, g._computers).admissibility_gap
        )
    return g.surface(
        **{
            "volume_radius_fill" if case["kind"] == "fill" else "fill_out_factor": case[
                "level"
            ]
        }
    )


def relative(a: float, b: float) -> float:
    return abs(a / b - 1) if b else abs(a - b)
