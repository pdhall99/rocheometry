"""Checks of every quadrature node against independent roots and running bounds."""

from __future__ import annotations

from math import atan2, dist, fsum, hypot
from typing import Any

import rocheometry
from crosschecks.distance.outlines import level_residual
from crosschecks.instrumented.node_running import capture_running_nodes, check_node
from crosschecks.ray_root import RayRoot


def inspect(surface: rocheometry.ComponentSurfaceGeometry) -> dict[str, Any]:
    g = surface.component
    rv = surface.volume_radius()
    # Measure afresh: empty the cached sample and quadrature first.
    for computer in (g._computers.sample, g._computers.quadrature):
        if hasattr(computer, "cache_clear"):
            computer.cache_clear()
    with capture_running_nodes() as records:
        q = surface.quadrature()
        sample = g._computers.sample(g.field, surface._depth)
    critical = surface._depth == 0
    ray_root = RayRoot(surface)
    backward = dict(level_backward=0.0, gradient_backward=0.0, gravity_backward=0.0)
    backward_worst = {}
    displacement = near_displacement = angle = near_angle = unit = 0.0
    worst_normal_node = 0
    minimum_dot = 1.0
    reflection = True
    for j in range(0, len(q), 4):
        p, n, gravity = q.positions[j], q.normals[j], q.gravities[j]
        direction = sample.rays[j // 4][0]
        node_check = check_node(
            surface, sample.radii[j // 4], direction, p, n, gravity, records[direction]
        )
        if not node_check["coordinate_rounding_passed"]:
            raise AssertionError("Cartesian node differs from single rounding of r*d")
        for name, source in (
            ("level_backward", "position_ratio"),
            ("gradient_backward", "gradient_ratio"),
            ("gravity_backward", "gravity_ratio"),
        ):
            if node_check[source] >= backward[name]:
                backward[name] = node_check[source]
                backward_worst[name] = dict(node_check, node=j // 4)
        expected = ray_root.normal(sample.rays[j // 4][0], sample.radii[j // 4])
        cross = (
            n[1] * expected[2] - n[2] * expected[1],
            n[2] * expected[0] - n[0] * expected[2],
            n[0] * expected[1] - n[1] * expected[0],
        )
        theta = atan2(hypot(*cross), fsum(a * b for a, b in zip(n, expected)))
        error = abs(level_residual(surface, p)) / (gravity * rv)
        if critical and dist(p, g.l1().position) < 1e-3 * rv:
            near_displacement = max(near_displacement, error)
            near_angle = max(near_angle, theta)
        else:
            displacement = max(displacement, error)
            if theta > angle:
                angle = theta
                worst_normal_node = j
        unit = max(unit, abs(hypot(*n) - 1))
        minimum_dot = min(minimum_dot, fsum(a * b for a, b in zip(n, p)) / hypot(*p))
        for k, (sy, sz) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            reflection &= (
                q.positions[j + k]
                == (
                    p[0],
                    sy * p[1],
                    sz * p[2],
                )
                and q.normals[j + k] == (n[0], sy * n[1], sz * n[2])
                and q.gravities[j + k] == gravity
            )
    area = fsum(q.weights)
    curvature = fsum(w * h for w, h in zip(q.weights, q.mean_curvatures))
    flux = fsum(w * h for w, h in zip(q.weights, q.gravities))
    smooth = dict(
        x2=fsum(w * p[0] ** 2 for w, p in zip(q.weights, q.positions)),
        z2=fsum(w * p[2] ** 2 for w, p in zip(q.weights, q.positions)),
        gravity4=fsum(w * v**4 for w, v in zip(q.weights, q.gravities)),
    )
    return dict(
        **backward,
        backward_worst=backward_worst,
        displacement=displacement,
        normal_angle=angle,
        worst_normal_node=worst_normal_node,
        normal_oracle="100-digit exact ray root",
        near_displacement=near_displacement,
        near_normal_angle=near_angle,
        unit=unit,
        outward=float(minimum_dot <= 0),
        reflections=float(not reflection),
        area_consistency=abs(area / surface.area() - 1),
        curvature_consistency=abs(curvature / surface.integrated_mean_curvature() - 1),
        gravity_consistency=abs(flux / area / surface.mean_gravity() - 1),
        smooth=smooth,
    )
