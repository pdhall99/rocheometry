"""Two-way normal distance between an outline and the exact field, in R_vol.

Every outline point is measured against the exact level set in its plane, and
every exact point against the outline, by independent root solves.
"""

from __future__ import annotations

from math import atan2, fsum, hypot
from typing import Any

import mpmath as mp

import rocheometry
from crosschecks.distance.curves import NearestCurve
from crosschecks.problem import Level, gradient, masses, potential
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._geometry import Point2D, Point3D


def _projected(
    surface: rocheometry.ComponentSurfaceGeometry, plane: rocheometry.CentrelinePlane
) -> ProjectedField:
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


def level_residual(
    surface: rocheometry.ComponentSurfaceGeometry, point: Point3D
) -> float:
    g = surface.component
    l1 = g.l1()
    outer = g.outer()
    depth = surface._depth
    anchor = (
        outer
        if point[0] < 0
        and depth > 0
        and abs(g.field._depth_offset(depth, l1.position[0], outer.position[0]))
        < abs(depth)
        else l1
    )
    return g.field.potential_difference(point, anchor.position) + g.field._depth_offset(
        depth, l1.position[0], anchor.position[0]
    )


def plane_gradient(plane: rocheometry.SectionPlane, gradient: Point3D) -> Point2D:
    a = plane.project((0.0, 0.0, 0.0))
    b = plane.project(gradient)
    return b[0] - a[0], b[1] - a[1]


def sheet_value_gradient(
    surface: rocheometry.ComponentSurfaceGeometry,
    plane: rocheometry.SectionPlane,
    p: Point2D,
) -> tuple[float, Point2D]:
    g = surface.component
    point = plane.lift(p)
    x, y, z = point
    rho = hypot(y, z)
    if rho == 0:
        return x - g.l1().position[0], (1.0, 0.0)
    psi = atan2(z, y)
    watershed = g._computers.watershed
    assert isinstance(watershed, TracedWatershedComputer)
    section = watershed.cut_section_computer(g.field, rho)
    height, xr, xp = section.evaluate(psi)
    cy, sz = y / rho, z / rho
    normal = (1.0, -xr * cy + xp * sz / rho, -xr * sz - xp * cy / rho)
    return x - height, plane_gradient(plane, normal)


def precise_slice_radius(
    level: Level, origin: Point3D, direction: Point3D, guess: float
) -> float:
    """Independent 40-digit root when a binary64 Newton check stalls."""
    with mp.workdps(40):
        m, c = masses(level.binary, mp.mpf)
        o, d = tuple(map(mp.mpf, origin)), tuple(map(mp.mpf, direction))
        value = level.value()
        target = mp.mpf(value.numerator) / value.denominator
        radius = mp.mpf(guess)
        for _ in range(32):
            point = tuple(a + radius * b for a, b in zip(o, d))
            residual = potential(m, c, *point) - target
            derivative = sum(g * v for g, v in zip(gradient(m, c, *point), d))
            step = residual / derivative
            radius -= step
            if abs(step) <= mp.mpf("1e-30") * abs(radius):
                return float(radius)
        raise RuntimeError("40-digit offset-slice root did not converge")


def normal_error(
    surface: rocheometry.ComponentSurfaceGeometry,
    path: rocheometry.CompoundBezierPath,
    plane: rocheometry.SectionPlane,
    envelope: rocheometry.CompoundBezierPath,
    silhouette: bool = False,
    *,
    volume_radius: float | None = None,
    samples_per_segment: int = 13,
) -> tuple[float, dict[str, Any], float]:
    """Implicit normal displacement, independently verified by exact radial roots.

    All segment probes are checked against the exact level/LOS envelope.
    At parameter 0.371 on each envelope segment, an exact radius solve
    supplies an additional projected normal-distance check. Cut pieces use
    the exact constant-rho watershed fit and its analytic derivatives.
    """
    g = surface.component
    rv = surface.volume_radius() if volume_radius is None else volume_radius
    field = g.field
    env = {s for p in envelope.paths for s in p.segments}
    proj: ProjectedField | None = None
    if silhouette:
        assert isinstance(plane, rocheometry.CentrelinePlane)
        proj = _projected(surface, plane)
    worst = 0.0
    worst_case: dict[str, Any] = {}
    radial = 0.0
    knots = 0.0
    near = 0.0
    exact_points: list[Point2D] = []
    for pathpart in path.paths:
        for segment in pathpart.segments:
            cut = not silhouette and segment not in env
            for t in (
                0.0,
                0.01,
                *((k + 0.5) / samples_per_segment for k in range(samples_per_segment)),
                0.99,
                1.0,
            ):
                p = segment.evaluate(t)
                p3 = plane.lift(p)
                if p3 == g.l1().position and surface.potential == g.l1().potential:
                    near += 1
                    continue
                if cut:
                    value, gradient = sheet_value_gradient(surface, plane, p)
                elif proj is not None:
                    value = proj.value(*p)
                    maximum = proj.maximum(*p)
                    gradient = plane_gradient(plane, field.gradient(*maximum))
                else:
                    value = level_residual(surface, p3)
                    gradient = plane_gradient(plane, field.gradient(*p3))
                norm = hypot(*gradient)
                error = abs(value) / (norm * rv) if norm else 0.0
                if error > worst:
                    worst = error
                    worst_case = dict(point=p, parameter=t, cut=cut)
                if t in (0.0, 1.0):
                    knots = max(knots, error)
                # Independent points on the exact implicit curve supply the
                # reverse distance. Iterate the normal correction to floating
                # resolution; this is not a fit to the traced path.
                exact = p
                for _ in range(16):
                    point3 = plane.lift(exact)
                    if cut:
                        f, grad = sheet_value_gradient(surface, plane, exact)
                    elif proj is not None:
                        f = proj.value(*exact)
                        grad = plane_gradient(
                            plane, field.gradient(*proj.maximum(*exact))
                        )
                    else:
                        f = level_residual(surface, point3)
                        grad = plane_gradient(plane, field.gradient(*point3))
                    square = fsum(v * v for v in grad)
                    if not square:
                        break
                    updated = (
                        exact[0] - f * grad[0] / square,
                        exact[1] - f * grad[1] / square,
                    )
                    if updated == exact:
                        break
                    exact = updated
                exact_points.append(exact)
            # Independent exact level/limb radius at an off-grid parameter.
            if not cut:
                p = segment.evaluate(0.371)
                r = hypot(*p)
                if r and (
                    surface.potential != g.l1().potential
                    or plane.lift(p) != g.l1().position
                ):
                    direction = (p[0] / r, p[1] / r)
                    if proj is not None:
                        angle = atan2(abs(p[1]), p[0])
                        rr = proj.radius(angle)
                        exact = (rr * direction[0], rr * direction[1])
                        grad = plane_gradient(
                            plane, field.gradient(*proj.maximum(*exact))
                        )
                    elif isinstance(plane, rocheometry.CentrelinePlane):
                        rr = surface.radius(plane.lift(direction))
                        exact = (rr * direction[0], rr * direction[1])
                        grad = plane_gradient(plane, field.gradient(*plane.lift(exact)))
                    elif surface.potential.value >= g.l1().potential.value:
                        origin = plane.lift((0.0, 0.0))
                        lifted = plane.lift(direction)
                        v = (
                            lifted[0] - origin[0],
                            lifted[1] - origin[1],
                            lifted[2] - origin[2],
                        )
                        if origin == (0.0, 0.0, 0.0):
                            rr = surface.radius(v)
                        else:
                            # A generic expanding bracket can skip the thin
                            # negative-potential interval near a lobe. The
                            # returned curve supplies an excellent independent
                            # Newton seed for this offset-slice check.
                            rr = r
                            for _ in range(16):
                                p3 = (
                                    origin[0] + rr * v[0],
                                    origin[1] + rr * v[1],
                                    origin[2] + rr * v[2],
                                )
                                derivative = fsum(
                                    a * b for a, b in zip(field.gradient(*p3), v)
                                )
                                step = level_residual(surface, p3) / derivative
                                updated = rr - step
                                if updated == rr:
                                    break
                                rr = updated
                                if abs(step) <= 2e-15 * rr:
                                    break
                            else:
                                rr = precise_slice_radius(
                                    Level.of(surface), origin, v, r
                                )
                        exact = (rr * direction[0], rr * direction[1])
                        grad = plane_gradient(plane, field.gradient(*plane.lift(exact)))
                    else:
                        # Contact axis slices may have shifted/disconnected polar origins.
                        # Their exact implicit check above does not assume the origin.
                        continue
                    exact_points.append(exact)
                    radial = max(
                        radial,
                        abs(fsum((a - b) * v for a, b, v in zip(p, exact, grad)))
                        / (hypot(*grad) * rv),
                    )
    reverse = NearestCurve(path).maximum(exact_points)[0] / rv if exact_points else 0.0
    return (
        max(worst, radial, reverse),
        dict(
            implicit_worst=worst,
            exact_radius_worst=radial,
            exact_to_reference_worst=reverse,
            worst=worst_case,
            near_apex_probes=near,
        ),
        knots,
    )
