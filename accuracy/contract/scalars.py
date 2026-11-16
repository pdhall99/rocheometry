"""Public readers, sampled volume, selection, radii and quadrature geometry."""

from math import fsum, hypot, pi, ulp
from typing import Any

from crosschecks.cone_deficit import cone_gaussian

import rocheometry
from accuracy.contract.bounds import SCALAR_BOUND, scalar_bound
from accuracy.harness.common import (
    DECADES,
    INTEGRALS,
    LEVEL_SMOKE,
    clear_samples,
    level_jobs,
    relative,
    select,
)
from accuracy.harness.contracted import contracted_rows
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study
from rocheometry._computers.fitted import FITTED_FILL_ABSOLUTE_TOLERANCE
from rocheometry._computers.surface_volume import SampledVolumeComputer

DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1e-4, 0.0),
    (1.0, 0.0, 1e-4),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 1.0, 0.0),
    (-1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
)


class Scalars(Study):
    name = "scalars"
    kind = "contract"
    quantities = (
        "lobe.volume_radius",
        "lobe.slope",
        "lobe.max_fill",
        "integrals.volume",
        "integrals.fill",
        "integrals.area",
        "integrals.mean_curvature",
        "integrals.mean_gravity",
        "radii.named",
        "radii.direction",
        "selection.by_fill",
        "selection.by_fill_out_factor",
        "selection.fill_for_radius",
        "quadrature.integrals",
    )
    domain = "The permanent grid: 17 decades, a 31-point sweep and 200 random mass ratios, each at its fills and fill-out factors."

    def jobs(self) -> list[tuple[float, list[tuple[str, float]]]]:
        return level_jobs()

    def smoke(self) -> tuple[float, list[tuple[str, float]]]:
        return LEVEL_SMOKE

    def measure(self, job: tuple[float, list[tuple[str, float]]]) -> list[Row]:
        qi, levels = job
        rows = []
        fitted = rocheometry.ComponentGeometry(qi)
        reference = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
        for label, method in (
            ("lobe.volume_radius", lambda g: g.lobe().volume_radius()),
            ("lobe.volume_radius.fast", lambda g: g.lobe_volume_radius()),
            ("lobe.slope", lambda g: g.lobe_volume_radius_slope()),
            ("lobe.max_fill", lambda g: g.max_volume_radius_fill()),
        ):
            try:
                truth = method(reference)
                for setting, g in (("fitted", fitted), ("reference", reference)):
                    rows.append(
                        record(
                            label,
                            setting,
                            abs(method(g) - truth)
                            if label == "lobe.slope"
                            else relative(method(g), truth),
                            SCALAR_BOUND[setting],
                            dict(qi=qi),
                            metric="absolute" if label == "lobe.slope" else "relative",
                        )
                    )
                if qi in DECADES:
                    for p, a in ((256, 128), (512, 192)):
                        h = rocheometry.ComponentGeometry(
                            qi,
                            accuracy=rocheometry.Reference(
                                polar_order=p, azimuth_order=a
                            ),
                        )
                        v = method(h)
                        rows.append(
                            record(
                                "reference_convergence." + label,
                                "reference",
                                abs(truth - v)
                                if label == "lobe.slope"
                                else relative(truth, v),
                                1e-10,
                                dict(qi=qi, polar=p, azimuth=a),
                                metric="absolute"
                                if label == "lobe.slope"
                                else "relative",
                            )
                        )
                        clear_samples(h)
            except Exception as exc:
                rows.append(
                    record(
                        label,
                        "reference",
                        None,
                        1e-10,
                        dict(qi=qi),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        for kind, selector in levels:
            case = dict(qi=qi, kind=kind, level=selector)
            try:
                chosen = select(fitted, kind, selector)
                u = chosen.potential
                exact = reference._surface_at_depth(chosen._depth)
                truth = physical(exact)
            except Exception as exc:
                rows.append(
                    record(
                        "selection.evaluation",
                        "fitted",
                        None,
                        1e-6,
                        case,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            for setting, g, s in (
                ("fitted", fitted, chosen),
                ("reference", reference, exact),
            ):
                bound = SCALAR_BOUND[setting]
                try:
                    values = physical(s)
                    for name, value in values.items():
                        rows.append(
                            record(
                                INTEGRALS[name],
                                setting,
                                relative(value, truth[name]),
                                scalar_bound(setting, INTEGRALS[name]),
                                case,
                                potential=u.value,
                                actual=value,
                                truth=truth[name],
                            )
                        )
                    expected = (
                        4 * pi * g.field.component_mass - 2 * values["volume"]
                    ) / values["area"]
                    rows.append(
                        record(
                            "integrals.mean_gravity.gauss_law",
                            setting,
                            relative(values["mean_gravity"], expected),
                            bound,
                            case,
                        )
                    )
                    for name in ("point", "back", "side", "pole"):
                        a, b = s.named_radius(name), exact.named_radius(name)
                        rows.append(
                            record(
                                "radii.named." + name,
                                setting,
                                relative(a, b),
                                bound,
                                case,
                                actual=a,
                                truth=b,
                            )
                        )
                    errors = [
                        relative(s.radius(d), exact.radius(d)) for d in DIRECTIONS
                    ]
                    i = max(range(len(errors)), key=errors.__getitem__)
                    rows.append(
                        record(
                            "radii.direction",
                            setting,
                            errors[i],
                            bound,
                            case,
                            direction=DIRECTIONS[i],
                        )
                    )
                    rows.extend(quadrature_rows(s, setting, case))
                    selected = select(g, kind, selector)
                    obtained = (
                        selected.volume_radius_fill()
                        if kind == "fill"
                        else selected.fill_out_factor()
                    )
                    error = abs(obtained - selector)
                    rows.append(
                        record(
                            "selection.by_fill.round_trip"
                            if kind == "fill"
                            else "selection.by_fill_out_factor.round_trip",
                            setting,
                            error,
                            (
                                FITTED_FILL_ABSOLUTE_TOLERANCE
                                if setting == "fitted"
                                else 1e-13
                            )
                            if kind == "fill"
                            else bound,
                            case,
                            metric="absolute",
                        )
                    )
                    rows.append(
                        record(
                            "selection.fill_for_radius",
                            setting,
                            abs(
                                g.volume_radius_fill_for(s.volume_radius())
                                - s.volume_radius_fill()
                            ),
                            bound,
                            case,
                            metric="absolute",
                        )
                    )
                    measured_fill = (
                        truth["sampled_volume"]
                        / SampledVolumeComputer(reference._computers.sample)(
                            reference.field, 0.0
                        )
                    ) ** (1 / 3)
                    rows.append(
                        record(
                            "integrals.fill.at_potential",
                            setting,
                            abs(s.volume_radius_fill() - measured_fill),
                            bound,
                            case,
                            metric="absolute",
                            reported_fill=s.volume_radius_fill(),
                            reference_sampled_fill=measured_fill,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        record(
                            "integrals.evaluation",
                            setting,
                            None,
                            bound,
                            case,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            if qi in DECADES:
                previous = None
                for p, a in ((256, 128), (512, 192)):
                    high = rocheometry.ComponentGeometry(
                        qi,
                        accuracy=rocheometry.Reference(polar_order=p, azimuth_order=a),
                    )
                    s = high._surface_at_depth(chosen._depth)
                    try:
                        values = physical(s)
                        for name, value in values.items():
                            rows.append(
                                record(
                                    "reference_convergence." + INTEGRALS[name],
                                    "reference",
                                    relative(truth[name], value),
                                    scalar_bound("reference", INTEGRALS[name]),
                                    {**case, "polar": p, "azimuth": a},
                                    actual=truth[name],
                                    truth=value,
                                )
                            )
                            if previous is not None:
                                rows.append(
                                    record(
                                        "reference_convergence.256_to_512."
                                        + INTEGRALS[name],
                                        "reference",
                                        relative(previous[name], value),
                                        1e-10,
                                        case,
                                    )
                                )
                        previous = values
                    except Exception as exc:
                        rows.append(
                            record(
                                "reference_convergence.evaluation",
                                "reference",
                                None,
                                1e-10,
                                {**case, "polar": p, "azimuth": a},
                                error=str(exc),
                            )
                        )
                    clear_samples(high)
            clear_samples(fitted)
            clear_samples(reference)
        # Reference accuracy comes from higher-order convergence, not comparison
        # of the default reference with itself. Quadrature has its own full study.
        return contracted_rows("scalars", rows)


def physical(surface: rocheometry.ComponentSurfaceGeometry) -> dict[str, float]:
    g = surface.component
    return dict(
        volume=surface.volume(),
        sampled_volume=SampledVolumeComputer(g._computers.sample)(
            g.field, surface._depth
        ),
        volume_radius=surface.volume_radius(),
        area=surface.area(),
        area_radius=surface.area_radius(),
        integrated_mean_curvature=surface.integrated_mean_curvature(),
        mean_curvature_radius=surface.mean_curvature_radius(),
        mean_gravity=surface.mean_gravity(),
    )


def quadrature_rows(
    surface: rocheometry.ComponentSurfaceGeometry, setting: str, case: dict[str, Any]
) -> list[Row]:
    g = surface.component
    f = g.field
    q = surface.quadrature()
    rv = surface.volume_radius()
    bound = SCALAR_BOUND[setting]
    level = surface.potential.value
    l1 = g.l1()
    outer = g.outer()
    critical = surface._depth == 0
    reference = (
        outer
        if not critical
        and surface._depth > 0
        and abs(f._depth_offset(surface._depth, l1.position[0], outer.position[0]))
        < abs(surface._depth)
        else l1
    )
    offset = f._depth_offset(surface._depth, l1.position[0], reference.position[0])
    worst = unit = 0.0
    largest_floor = 0.0
    for j in range(0, len(q), 4):
        p, n, gravity = q.positions[j], q.normals[j], q.gravities[j]
        residual = f.potential_difference(p, reference.position) + offset
        value = abs(residual) / (gravity * rv)
        if value > worst:
            worst = value
        largest_floor = max(largest_floor, 4 * ulp(level) / (gravity * rv))
        unit = max(unit, abs(hypot(*n) - 1))
    area = fsum(q.weights)
    flux = fsum(w * v for w, v in zip(q.weights, q.gravities))
    curvature = fsum(w * v for w, v in zip(q.weights, q.mean_curvatures))
    rows = [
        record(
            "quadrature.nodes.level_normal_displacement",
            setting,
            worst,
            bound,
            case,
            metric="normal distance / R_vol",
            rounding_bound=largest_floor,
        ),
        record(
            "quadrature.nodes.unit_normals",
            setting,
            unit,
            1e-12,
            case,
            metric="absolute",
        ),
    ]
    for name, a, b in (
        ("area", area, surface.area()),
        ("curvature", curvature, surface.integrated_mean_curvature()),
        ("gravity", flux / area, surface.mean_gravity()),
    ):
        rows.append(
            record(
                "quadrature.integrals.reader_consistency." + name,
                setting,
                relative(a, b),
                1e-13,
                case,
            )
        )
    gaussian = fsum(w * k for w, k in zip(q.weights, q.gaussian_curvatures))
    if surface._depth <= 0:
        target = cone_gaussian(g) if critical else 4 * pi
        note = (
            "critical-apex conditioning; 128-node guard only"
            if critical
            else "near-critical fitted Gauss–Bonnet diagnostic"
            if case["kind"] == "fill" and case["level"] > 0.99
            else "Gauss–Bonnet diagnostic"
        )
        rows.append(
            record(
                "quadrature.integrals.gauss_bonnet",
                setting,
                abs(gaussian - target),
                None,
                case,
                metric="absolute",
                note=note,
                integral=gaussian,
                target=target,
            )
        )
    return rows
