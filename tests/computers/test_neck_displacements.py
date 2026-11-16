"""Resolvable contact necks retain local coordinates through their readers."""

import math

import pytest

import rocheometry
from rocheometry._computers.contact import _geometry_depth
from rocheometry._computers.neck import (
    _neck_displacement,
    _neck_field,
    _unresolved_neck_ray,
)
from rocheometry._computers.silhouette.projected import ProjectedField
from tests._depth_precision import direction_angle, projected_at_depth, ray_at_depth

_QI = (1e-8, 1e-3, 1.0, 1e3, 1e8)
_DEPTHS = (1e-16, 1e-20, 1e-30, 1e-60)
_CONSTRUCTION_DEPTHS = (*_DEPTHS, 1e-200, math.nextafter(0.0, 1.0))


@pytest.mark.slow
@pytest.mark.parametrize("qi", _QI)
@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
def test_shallow_neck_grid_constructs_without_changing_caps(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    l1 = g.l1()
    for depth in _CONSTRUCTION_DEPTHS:
        s = g._surface_at_depth(depth)
        assert s.regime is rocheometry.LevelRegime.CONTACT
        if _geometry_depth(depth, l1) == 0:
            neck = math.sqrt(depth) * math.sqrt(2 / l1.hessian[0][0])
            assert l1.position[0] - neck == l1.position[0] + neck == l1.position[0]
        for inclination in (0.0, math.pi / 3, math.pi / 2):
            plane = rocheometry.CentrelinePlane(inclination)
            for outline in (s.section(plane), s.silhouette_envelope(plane)):
                for part in outline.paths:
                    for segment in part.segments:
                        assert all(
                            math.isfinite(v)
                            for point in (
                                segment.p0,
                                segment.p1,
                                segment.p2,
                                segment.p3,
                            )
                            for v in point
                        )
        assert math.isfinite(s.area())
        assert math.isfinite(s.integrated_mean_curvature())
        assert math.isfinite(s.mean_gravity())
        assert all(math.isfinite(v) for v in s.quadrature().gravities)


@pytest.mark.parametrize("qi", _QI)
@pytest.mark.parametrize("depth", _DEPTHS[:-1])
def test_neck_radii_and_tangents_match_independent_80_digit_roots(
    qi: float, depth: float
) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    l1 = g.l1()
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    second = plane.lift((0.0, 1.0))
    rim = g._computers.watershed.rim(g.field, depth, plane.inclination)
    start = math.atan2(math.hypot(rim[1], rim[2]), rim[0])
    projected = ProjectedField(
        g.field,
        depth,
        plane,
        g._computers.radius,
        l1,
        g._computers.outer,
        g._computers.watershed,
    )
    for factor in (1.01, 1.5, 2.0):
        angle = start * factor
        direction = (
            math.cos(angle),
            math.sin(angle) * second[1],
            math.sin(angle) * second[2],
        )
        assert _unresolved_neck_ray(depth, direction, l1.position[0], g.field)
        radius = g._computers.radius(g.field, depth, direction)
        truth, gradient = ray_at_depth(g.field, l1.position[0], depth, direction)
        actual_gradient = _neck_field(g.field, l1.position[0]).gradient(
            _neck_displacement(g.field, l1.position[0], depth, direction)
        )
        assert abs(radius / truth - 1) <= 1e-10
        assert direction_angle(actual_gradient, gradient) <= 1e-10
        angle = projected.start * factor
        radius = projected.radius(angle)
        truth, gradient = projected_at_depth(projected, angle)
        (gx, gt), _ = projected.differential(
            (radius * math.cos(angle), radius * math.sin(angle))
        )
        assert abs(radius / truth - 1) <= 1e-10
        assert direction_angle((gx, gt, 0.0), gradient) <= 1e-10


@pytest.mark.parametrize("qi", _QI)
def test_contact_neck_distances_approach_critical_at_fixed_sky_rays(qi: float) -> None:
    g = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    l1 = g.l1()
    rim = g._computers.watershed.rim(g.field, _DEPTHS[0], plane.inclination)
    start = math.atan2(math.hypot(rim[1], rim[2]), rim[0])
    second = plane.lift((0.0, 1.0))
    directions = tuple(
        (math.cos(a), math.sin(a) * second[1], math.sin(a) * second[2])
        for a in (start * 1.01, start * 1.5, start * 2, 1e-3)
    )
    critical = tuple(g._computers.radius(g.field, 0.0, d) for d in directions)
    errors = [
        max(
            abs(g._computers.radius(g.field, depth, d) - r)
            for d, r in zip(directions, critical, strict=True)
        )
        for depth in _DEPTHS
    ]
    assert all(
        b <= a + math.ulp(l1.position[0])
        for a, b in zip(errors[:-1], errors[1:], strict=True)
    )
    assert errors[-1] == 0


@pytest.mark.slow
def test_contact_limb_keeps_displacement_before_distances_stop_changing() -> None:
    from rocheometry._computers.silhouette.projected import ProjectedField
    from tests._depth_precision import direction_angle, projected_at_depth

    g = rocheometry.ComponentGeometry(1e-3, accuracy=rocheometry.Reference())
    surface = g.surface(fill_out_factor=1e-12)
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    p = ProjectedField(
        g.field,
        surface._depth,
        plane,
        g._computers.radius,
        g.l1(),
        g._computers.outer,
        g._computers.watershed,
    )
    for angle in (
        2.6459556903720454e-7,
        2.6459556904810416e-7,
        p.start * 1.01,
        p.start * 1.5,
    ):
        radius = p.radius(angle)
        truth, gradient = projected_at_depth(p, angle)
        actual, _ = p.differential((radius * math.cos(angle), radius * math.sin(angle)))
        assert radius == pytest.approx(truth, rel=1e-10)
        assert direction_angle((*actual, 0.0), gradient) <= 1e-10
    assert surface.silhouette_envelope(plane).paths
