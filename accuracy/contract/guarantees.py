"""Measured outline guarantees: knots, the L1 cone and approach, binary joins.

Properties that hold exactly (closed, nonempty, mirror symmetric, starting at
L1) are tests, swept over every decade in the slow tier; this study measures
the ones that carry a numerical error.
"""

from __future__ import annotations

from math import atan2, cos, dist, hypot, pi, sin, sqrt

import numpy as np
from crosschecks.distance.curves import controls
from crosschecks.distance.outlines import (
    level_residual,
    plane_gradient,
    sheet_value_gradient,
)
from crosschecks.geometry import crossing_at_depth

import rocheometry
from accuracy.contract.bounds import OUTLINE_MARGIN
from accuracy.contract.outlines import INCLINATIONS
from accuracy.harness.common import (
    clear_samples,
    decade_jobs,
    projected_field,
    select,
    silhouette_segments,
)
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study
from rocheometry._computers.section_path.gauge_fixed import (
    NECK_SLOT_PLACEMENT_MULTIPLIER,
)


class Guarantees(Study):
    name = "guarantees"
    kind = "contract"
    quantities = (
        "outlines.shape",
        "binary.join",
    )
    domain = "Knots, the L1 cone and approach and binary joins at the 17 decades of qi, four levels, four inclinations and three outline kinds."

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        rows = []
        for setting, accuracy in (
            ("fitted", rocheometry.Fitted()),
            ("reference", rocheometry.Reference()),
        ):
            g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
            for kind, value in (("fill", 0.8), ("fill", 1.0), ("F", 0.01), ("F", 0.99)):
                s = select(g, kind, value)
                for i in INCLINATIONS:
                    plane = rocheometry.CentrelinePlane(i)
                    for method in (
                        "section",
                        "section_envelope",
                        "silhouette_envelope",
                    ):
                        case = dict(
                            qi=qi, kind=kind, level=value, inclination=i, method=method
                        )
                        try:
                            knot, path = knot_on_surface(s, method, plane)
                            rows.append(
                                record(
                                    "outlines.shape.knots",
                                    setting,
                                    fitted_knot_radius_error(s, method, plane)
                                    if setting == "fitted"
                                    else knot,
                                    1e-12 if setting == "fitted" else 1e-10,
                                    case,
                                    metric="prescribed-ray radius relative"
                                    if setting == "fitted"
                                    else "normal distance / R_vol",
                                )
                            )
                            if setting == "fitted":
                                rows.append(
                                    record(
                                        "outlines.shape.knots_normal",
                                        setting,
                                        knot,
                                        None,
                                        case,
                                        metric="normal distance / R_vol",
                                        note="Unbounded fitted-knot normal-displacement diagnostic; the guarantee is along each prescribed ray.",
                                    )
                                )
                            if kind == "fill" and value == 1:
                                angle = critical_corner(
                                    g, path, plane, method.startswith("silhouette")
                                )
                                rows.append(
                                    record(
                                        "outlines.shape.cone_tangent",
                                        setting,
                                        angle,
                                        1e-8,
                                        case,
                                        metric="radians",
                                    )
                                )
                        except Exception as exc:
                            rows.append(
                                record(
                                    "outlines.shape.evaluation",
                                    setting,
                                    None,
                                    1e-12,
                                    case,
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            )
                clear_samples(g)
        binary = rocheometry.BinaryGeometry(qi, accuracy=rocheometry.Reference())
        heavy, other = binary.geometry1, binary.geometry2
        for F in (1e-6, 0.01, 0.5, 0.99):
            chosen = heavy.surface(fill_out_factor=F)
            depth = chosen._depth
            for i in INCLINATIONS:
                case = dict(qi=qi, kind="F", level=F, inclination=i)
                try:
                    plane = rocheometry.CentrelinePlane(i)
                    a = crossing_at_depth(heavy, depth, plane)
                    b = crossing_at_depth(other, depth, plane)
                    gap = dist(a, (1 - b[0], b[1]))
                    rows.append(
                        record(
                            "binary.join",
                            "reference",
                            gap,
                            1e-12,
                            case,
                            metric="distance / separation",
                        )
                    )
                except Exception as exc:
                    rows.append(
                        record(
                            "binary.join",
                            "reference",
                            None,
                            1e-12,
                            case,
                            error=str(exc),
                        )
                    )
        # A fill sweep is recorded as geometric second differences, not an arbitrary
        # invented quantitative continuity contract. Existing quarter-power bounds
        # provide the actual pass/fail continuity check where levels are representable.
        g = rocheometry.ComponentGeometry(qi)
        Fs = (
            -0.1,
            -0.01,
            -0.001,
            -1e-4,
            -1e-6,
            -1e-8,
            0.0,
            1e-8,
            1e-6,
            1e-4,
            0.001,
            0.01,
            0.1,
        )
        for i in INCLINATIONS:
            plane = rocheometry.CentrelinePlane(i)
            for method in ("section", "silhouette_envelope"):
                values = []
                case = dict(qi=qi, inclination=i, method=method)
                for F in Fs:
                    try:
                        s = g.surface(fill_out_factor=F)
                        values.append(
                            (F, controls(getattr(s, method)(plane)), s.volume_radius())
                        )
                    except Exception as exc:
                        rows.append(
                            record(
                                "outlines.shape.fill_sweep",
                                "fitted",
                                None,
                                None,
                                {**case, "F": F},
                                error=str(exc),
                            )
                        )
                for a, b, c in zip(values, values[1:], values[2:]):
                    if a[1].shape != b[1].shape or b[1].shape != c[1].shape:
                        continue

                    def x(t: float) -> float:
                        return (1 if t >= 0 else -1) * abs(t) ** 0.25

                    weight = (x(b[0]) - x(a[0])) / (x(c[0]) - x(a[0]))
                    error = (
                        float(
                            np.max(
                                np.linalg.norm(
                                    b[1] - (1 - weight) * a[1] - weight * c[1], axis=2
                                )
                            )
                        )
                        / b[2]
                    )
                    rows.append(
                        record(
                            "outlines.shape.fill_second_difference",
                            "fitted",
                            error,
                            None,
                            {**case, "F": b[0]},
                            metric="control deviation / R_vol",
                            note="finite-grid continuity diagnostic; no numeric bound specified",
                        )
                    )
                if method.startswith("silhouette"):
                    rows.extend(silhouette_l1_guarantees(g, plane))
                    continue
                try:
                    rows.extend(section_l1_guarantees(g, plane))
                except Exception as exc:
                    rows.append(
                        record(
                            "outlines.shape.L1_continuity",
                            "fitted",
                            None,
                            1e-3,
                            case,
                            error=str(exc),
                        )
                    )
        return rows


def neck_transition_depth(g: rocheometry.ComponentGeometry, count: int) -> float:
    """Depth where sqrt(beta*nu*phi1)=phi1/2 for this gauge's first gap.

    nu=sqrt(2*delta/U_xx)/x_L1, so delta_t is
    U_xx*x_L1**2*phi1**2/(32*beta**2). No absolute level is materialized.
    """
    m = count - 2
    phi1 = pi * (1 / m - sin(2 * pi / m) / (2 * pi))
    l1 = g.l1()
    return (
        l1.hessian[0][0]
        * l1.position[0] ** 2
        * phi1**2
        / (32 * NECK_SLOT_PLACEMENT_MULTIPLIER**2)
    )


def section_l1_guarantees(
    g: rocheometry.ComponentGeometry, plane: rocheometry.CentrelinePlane
) -> list[Row]:
    """Keep fixed-depth continuity bounds; measure rate below the transition."""
    case = dict(qi=g.qi, inclination=plane.inclination, method="section")
    pole = g.lobe().named_radius("pole")

    def error(contact: rocheometry.ComponentSurfaceGeometry) -> float:
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        detached = g._surface_at_depth(-contact._depth)
        assert detached.regime is rocheometry.LevelRegime.DETACHED
        return (
            float(
                np.linalg.norm(
                    controls(contact.section(plane))
                    - controls(detached.section(plane)),
                    axis=-1,
                ).max()
            )
            / pole
        )

    values = [error(g.surface(fill_out_factor=f)) for f in (1e-4, 1e-6, 1e-8)]
    transition = neck_transition_depth(g, 26)
    pair = [g._surface_at_depth(transition / divisor) for divisor in (100, 10000)]
    assert all(s.regime is rocheometry.LevelRegime.CONTACT for s in pair)
    rate_errors = [error(s) for s in pair]
    return [
        record(
            "outlines.shape.L1_continuity",
            "fitted",
            values[-1],
            1e-3,
            case,
            metric="control difference / pole radius",
            E=values,
        ),
        record(
            "outlines.shape.L1_rate",
            "fitted",
            rate_errors[1] / rate_errors[0],
            0.4,
            case,
            metric="ratio",
            transition=transition,
            depth_divisors=[100, 10000],
            rate_errors=rate_errors,
            fixed_depth_ratio=values[-1] / values[-2],
        ),
    ]


def silhouette_l1_guarantees(
    g: rocheometry.ComponentGeometry, plane: rocheometry.CentrelinePlane
) -> list[Row]:
    """Measure fixed-depth decrease, the margin and transition-relative rate."""
    case = dict(qi=g.qi, inclination=plane.inclination, method="silhouette_envelope")

    def error(contact: rocheometry.ComponentSurfaceGeometry) -> float:
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        detached = g._surface_at_depth(-contact._depth)
        return float(
            np.linalg.norm(
                controls(contact.silhouette_envelope(plane))
                - controls(detached.silhouette_envelope(plane)),
                axis=-1,
            ).max()
        ) / contact.named_radius("pole")

    values = []
    for factor in (1e-6, 1e-8, 1e-10, 1e-12):
        contact = g.surface(fill_out_factor=factor)
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        values.append(error(contact))
    assert len(values) == 4
    rows = [
        record(
            "outlines.shape.L1_continuity",
            "fitted",
            values[-1],
            OUTLINE_MARGIN,
            case,
            metric="control difference / pole radius",
            E=values,
        )
    ]
    transition = neck_transition_depth(g, silhouette_segments(g))
    pair = [g._surface_at_depth(transition / divisor) for divisor in (100, 10000)]
    assert all(s.regime is rocheometry.LevelRegime.CONTACT for s in pair)
    rate_errors = [error(s) for s in pair]
    rows.append(
        record(
            "outlines.shape.L1_rate",
            "fitted",
            rate_errors[1] / rate_errors[0],
            0.4,
            case,
            metric="ratio",
            transition=transition,
            depth_divisors=[100, 10000],
            rate_errors=rate_errors,
            fixed_depth_ratio=values[1] / values[0],
        )
    )
    return rows


def knot_on_surface(
    surface: rocheometry.ComponentSurfaceGeometry,
    method: str,
    plane: rocheometry.CentrelinePlane,
) -> tuple[float, rocheometry.CompoundBezierPath]:
    """Largest normal distance of a knot from the level set, in R_vol."""
    path = getattr(surface, method)(plane)
    g = surface.component
    rv = surface.volume_radius()
    env = getattr(
        surface,
        "silhouette_envelope"
        if method.startswith("silhouette")
        else "section_envelope",
    )(plane)
    envelope = {s for p in env.paths for s in p.segments}
    proj = projected_field(surface, plane) if method.startswith("silhouette") else None
    knot = 0.0
    for part in path.paths:
        for segment in part.segments:
            for point in (segment.p0, segment.p3):
                if plane.lift(point) == g.l1().position and surface._depth == 0:
                    continue
                if not method.startswith("silhouette") and segment not in envelope:
                    value, gradient = sheet_value_gradient(surface, plane, point)
                elif proj:
                    value = proj.value(*point)
                    gradient = plane_gradient(
                        plane, g.field.gradient(*proj.maximum(*point))
                    )
                else:
                    p = plane.lift(point)
                    value = level_residual(surface, p)
                    gradient = plane_gradient(plane, g.field.gradient(*p))
                norm = hypot(*gradient)
                if norm:
                    knot = max(knot, abs(value) / (norm * rv))
    return knot, path


def fitted_knot_radius_error(
    surface: rocheometry.ComponentSurfaceGeometry,
    method: str,
    plane: rocheometry.CentrelinePlane,
    envelope: rocheometry.CompoundBezierPath | None = None,
) -> float:
    """Compare fixed knots on their prescribed rays, without recovering angles.

    Recovering the angle with atan2 of rounded coordinates perturbs a nearly
    radial neck ray. The exact-knot tests instead use the schedule's original
    sine/cosine pair; this audit uses that same public guarantee.
    """
    from rocheometry._computers.saddle import _neck_width
    from rocheometry._computers.section_path.contact import ContactCut
    from rocheometry._computers.section_path.gauge_fixed import _gauge_angles
    from rocheometry._planes import _transverse_direction

    g = surface.component
    silhouette = method.startswith("silhouette")
    if envelope is None:
        envelope = getattr(
            surface, "silhouette_envelope" if silhouette else "section_envelope"
        )(plane)
    if silhouette:
        proj = projected_field(surface, plane)
        start = proj.start
        radius = proj.radius
    else:
        start = (
            ContactCut(
                g.field, surface._depth, plane, g.l1(), g._computers.watershed
            ).rim_angle
            if surface.regime is rocheometry.LevelRegime.CONTACT
            else 0.0
        )
        second = _transverse_direction(plane)

        def radius(angle: float) -> float:
            c, s = (-1.0, 0.0) if angle == pi else (cos(angle), sin(angle))
            return g._computers.radius(
                g.field, surface._depth, (c, s * second[1], s * second[2])
            )

    nu = _neck_width(g.field, surface._depth, g.l1())
    count = silhouette_segments(g) if silhouette else 26
    angles = _gauge_angles(count, start, nu)
    upper = envelope.paths[0].segments[: len(envelope.paths[0].segments) // 2]
    if len(upper) == count:
        upper = upper[1:]  # detached/front slot or silhouette endpoint slot
    points = [segment.p0 for segment in upper] + [upper[-1].p3]
    assert len(points) == len(angles)
    worst = 0.0
    for angle, point in zip(angles, points, strict=True):
        if angle == 0.0 and surface._depth == 0:
            assert point == g.l1().position[:2]
            continue
        expected = radius(angle)
        worst = max(worst, abs(hypot(*point) / expected - 1))
    return worst


def critical_corner(
    g: rocheometry.ComponentGeometry,
    path: rocheometry.CompoundBezierPath,
    plane: rocheometry.CentrelinePlane,
    silhouette: bool,
) -> float:
    x1 = g.l1().position[0]
    h = g.l1().hessian
    c, s = cos(plane.inclination), sin(plane.inclination)
    yy, zz = h[1][1], h[2][2]
    tt = c * c * yy + s * s * zz
    if silhouette:
        tt -= ((yy - zz) * s * c) ** 2 / (s * s * yy + c * c * zz)
    expected = (-sqrt(-tt), sqrt(h[0][0]))
    length = hypot(*expected)
    expected = tuple(v / length for v in expected)
    for part in path.paths:
        for seg in part.segments:
            if seg.p0 == (x1, 0.0) and seg.p1 != seg.p0:
                v = (seg.p1[0] - x1, seg.p1[1])
                n = hypot(*v)
                v = (v[0] / n, v[1] / n)
                angle = atan2(
                    abs(v[0] * expected[1] - v[1] * expected[0]),
                    v[0] * expected[0] + v[1] * expected[1],
                )
                return angle
    raise RuntimeError("no nondegenerate outgoing cone arm")
