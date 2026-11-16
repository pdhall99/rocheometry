"""Cosine rounding must not turn a shallow-depth ray into an axis."""

import math

import mpmath as mp
import pytest

import rocheometry
from rocheometry._computers.contact import _geometry_depth
from rocheometry._geometry import Point3D


def test_resolvable_contact_neck_does_not_alias_nonaxial_ray_to_l1() -> None:
    g = rocheometry.ComponentGeometry(1e-3)
    depth = 1e-20
    l1 = g.l1()
    neck = math.sqrt(depth) * math.sqrt(2 / l1.hessian[0][0])
    assert neck > 3e6 * math.ulp(l1.position[0])
    assert _geometry_depth(depth, l1) == depth
    direction = (1.0, 1.1242215360652175e-9, 2.4939394342663396e-12)
    radius = g._computers.radius(g.field, depth, direction)
    # The 60-digit first enclosing root is 20195 ULP below the false axis hit.
    assert radius < l1.position[0]
    gradient = g.field.gradient(*(radius * d for d in direction))
    assert -sum(a * b for a, b in zip(gradient, direction)) > 0


def _envelope_root(g: rocheometry.ComponentGeometry, direction: Point3D) -> float:
    """Bisect the first enclosing level root with independent 60-digit arithmetic."""
    with mp.workdps(60):
        m, c = mp.mpf(g.field.component_mass), mp.mpf(g.field.companion_mass)
        total = m + c
        x1 = mp.mpf(g.l1().position[0])
        target = m / x1 + c / (1 - x1) + total / 2 * (x1 - c / total) ** 2
        target -= mp.mpf(1e-20)
        d = tuple(map(mp.mpf, direction))

        def residual(radius: mp.mpf) -> mp.mpf:
            x, y, z = (radius * v for v in d)
            return (
                m / mp.sqrt(x * x + y * y + z * z)
                + c / mp.sqrt((1 - x) ** 2 + y * y + z * z)
                + total / 2 * ((x - c / total) ** 2 + y * y)
                - target
            )

        lower, upper = x1 * (1 - mp.mpf("1e-5")), x1
        assert residual(lower) > 0 > residual(upper)
        for _ in range(180):
            middle = (lower + upper) / 2
            if residual(middle) > 0:
                lower = middle
            else:
                upper = middle
        return float((lower + upper) / 2)


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
def test_shallow_nonaxial_rays_follow_the_cut_or_the_envelope(
    accuracy: rocheometry.Fitted | rocheometry.Reference,
) -> None:
    """Cosine is one on all three rays; only the last is outside the cut.

    The two rays inside the rim meet the watershed, not the selected level.
    Their transverse displacement changes its axial height by less than an ULP.
    Compare the envelope ray with an independent high-precision level root.
    """
    g = rocheometry.ComponentGeometry(1e-3, accuracy=accuracy)
    depth = 1e-20
    rim = g._computers.watershed.rim(g.field, depth, math.pi / 2)
    rim_angle = math.atan2(math.hypot(rim[1], rim[2]), rim[0])
    for angle in (1e-12, 1e-10, 1e-9):
        direction = (math.cos(angle), 0.0, math.sin(angle))
        assert direction[0] == 1.0 and direction[2] != 0.0
        radius = g._computers.radius(g.field, depth, direction)
        if angle < rim_angle:
            point = g._computers.watershed.sheet_point_on_ray(g.field, direction)
            assert radius == math.hypot(*point)
            # The compensated sheet includes the residual axial-gradient
            # correction at the computed L1 (a few ULPs in this case).
            assert radius == pytest.approx(g.l1().position[0], rel=1e-12)
        else:
            root = _envelope_root(g, direction)
            assert radius == pytest.approx(root, rel=1e-12)
            assert radius < g.l1().position[0]


@pytest.mark.parametrize("accuracy", (rocheometry.Fitted(), rocheometry.Reference()))
def test_cosine_rounding_boundary_has_no_radius_jump(
    accuracy: rocheometry.Fitted | rocheometry.Reference,
) -> None:
    g = rocheometry.ComponentGeometry(1e-3, accuracy=accuracy)
    lower, upper = 1e-9, 2e-8
    while math.nextafter(lower, upper) != upper:
        middle = lower + (upper - lower) / 2
        if math.cos(middle) == 1.0:
            lower = middle
        else:
            upper = middle
    second = rocheometry.CentrelinePlane(math.pi / 3).lift((0.0, 1.0))
    radii = []
    for angle in (lower, upper):
        direction = (
            math.cos(angle),
            math.sin(angle) * second[1],
            math.sin(angle) * second[2],
        )
        radius = g._computers.radius(g.field, 1e-20, direction)
        assert radius == pytest.approx(_envelope_root(g, direction), rel=1e-12)
        radii.append(radius)
    assert abs(radii[1] - radii[0]) <= 4 * math.ulp(g.l1().position[0])


@pytest.mark.slow
def test_reference_shallow_paths_construct_with_unchanged_caps() -> None:
    surface = rocheometry.ComponentGeometry(
        1e-3, accuracy=rocheometry.Reference()
    )._surface_at_depth(1e-20)
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    assert surface.regime is rocheometry.LevelRegime.CONTACT
    for path in (surface.section(plane), surface.silhouette_envelope(plane)):
        assert path.paths
        for part in path.paths:
            for segment in part.segments:
                for point in (segment.p0, segment.p1, segment.p2, segment.p3):
                    assert all(math.isfinite(value) for value in point)
