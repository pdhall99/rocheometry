"""Projected quadrature outlines agree with independent shadows and exact limbs."""

import pickle
from dataclasses import FrozenInstanceError, replace
from math import atan2, cos, hypot, pi, sin, sqrt

import pytest
from crosschecks.distance.bezier import _PathDistance

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.section_path.gauge_fixed import (
    NECK_SLOT_PLACEMENT_MULTIPLIER,
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.silhouette.paths import (
    GaugeFixedSilhouettePathComputer,
    TracedSilhouettePathComputer,
)
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point2D
from tests._helpers import _Wired

QI = (1e-3, 0.05, 1.0, 20.0, 1e3)
VIEWS = (pi / 6, pi / 3)


def surface(
    qi: float, kind: str, reference: bool
) -> rocheometry.ComponentSurfaceGeometry:
    selected = rocheometry.ComponentGeometry(qi)
    if kind == "critical":
        potential = selected.l1().potential
    elif kind == "detached":
        potential = selected.surface(volume_radius_fill=0.99).potential
    else:
        potential = selected.surface(fill_out_factor=0.5).potential
    graph = Reference()._computers() if reference else Fitted()._computers()
    return rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph)).surface(
        potential=potential
    )


def projected(
    s: rocheometry.ComponentSurfaceGeometry, inclination: float
) -> ProjectedField:
    geo = s.component
    g = geo._computers
    return ProjectedField(
        geo.field,
        s._depth,
        rocheometry.CentrelinePlane(inclination),
        g.radius,
        geo.l1(),
        g.outer,
        g.watershed,
    )


def radial_error(path: rocheometry.BezierPath, f: ProjectedField) -> float:
    worst = 0.0
    for segment in path.segments:
        if segment.p0 == segment.p1 == segment.p2 == segment.p3:
            continue
        for t in (0.071, 0.193, 0.413, 0.687, 0.919):
            x, y = segment.evaluate(t)
            radius = f.radius(atan2(abs(y), x))
            worst = max(worst, abs(hypot(x, y) - radius) / f.scale)
    return worst


def test_silhouette_plane_is_folded_frozen_picklable_and_has_only_inclination() -> None:
    view = rocheometry.CentrelinePlane(pi / 3)
    assert rocheometry.CentrelinePlane(pi - pi / 3).inclination == pytest.approx(
        view.inclination, rel=1e-15
    )
    assert pickle.loads(pickle.dumps(view)) == view
    assert "CentrelinePlane" in repr(view)
    with pytest.raises(FrozenInstanceError):
        setattr(view, "inclination", 0.0)
    with pytest.raises(TypeError, match="inclination"):
        rocheometry.CentrelinePlane(True)
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="inclination"):
            rocheometry.CentrelinePlane(value)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
@pytest.mark.parametrize("inclination", VIEWS)
def test_silhouette_accuracy_and_limb_property(
    reference: bool, qi: float, kind: str, inclination: float
) -> None:
    s = surface(qi, kind, reference)
    view = rocheometry.CentrelinePlane(inclination)
    outline = s.silhouette_envelope(view)
    assert len(outline.paths) == 1
    path = outline.paths[0]
    assert path.closed is (kind != "contact")
    if not reference:
        assert len(path.segments) == 88
    if kind == "contact":
        with pytest.raises(
            NotImplementedError, match="use silhouette_envelope at contact levels"
        ):
            s.silhouette(view)
    else:
        assert s.silhouette(view) == outline
    f = projected(s, inclination)
    assert radial_error(path, f) <= (1e-10 if reference else 3e-4)
    if reference:
        for segment in path.segments:
            x, t = segment.evaluate(0.413)
            p = f.maximum(x, t)
            gradient = f.field.gradient(*p)
            assert abs(gradient[1] * f.s + gradient[2] * f.c) <= 1e-10 * hypot(
                *gradient
            )


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
@pytest.mark.parametrize("inclination", (0.0, pi / 2))
def test_special_views_equal_the_corresponding_sections(
    reference: bool, kind: str, inclination: float
) -> None:
    s = surface(0.05, kind, reference)
    limb = s.silhouette_envelope(rocheometry.CentrelinePlane(inclination)).paths[0]
    section = s.section_envelope(rocheometry.CentrelinePlane(inclination)).paths[0]
    if reference:
        f = projected(s, inclination)
        assert radial_error(limb, f) <= 1e-10
        distance = _PathDistance(section)
        assert max(distance(p) for p in limb.sample(73)) / f.scale <= 1e-10
    else:
        # Match resolutions when checking algebraic identity. Packaged
        # sections retain 26 per half; silhouettes now use 44 per half.
        g = s.component._computers
        section = (
            GaugeFixedSectionPathComputer(
                radius_computer=g.radius,
                l1_computer=g.l1,
                outer_computer=g.outer,
                watershed_computer=g.watershed,
                segments_per_half=44,
            )
            .envelope(
                s.component.field,
                s._depth,
                rocheometry.CentrelinePlane(inclination),
            )
            .paths[0]
        )
        segments = limb.segments[1:-1] if kind == "contact" else limb.segments
        assert len(segments) == len(section.segments)
        for a, b in zip(segments, section.segments, strict=True):
            for p, q in zip(
                (a.p0, a.p1, a.p2, a.p3), (b.p0, b.p1, b.p2, b.p3), strict=True
            ):
                assert hypot(p[0] - q[0], p[1] - q[1]) <= 1e-12 * s.named_radius("pole")


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("inclination", (0.0, pi / 6, pi / 3, pi / 2))
def test_critical_l1_corner_uses_projected_hessian_cone(
    reference: bool, qi: float, inclination: float
) -> None:
    s = surface(qi, "critical", reference)
    l1 = s.component.l1()
    path = s.silhouette(rocheometry.CentrelinePlane(inclination)).paths[0]
    segment = next(a for a in path.segments if a.p0 == l1.position[:2] and a.p1 != a.p0)
    tx, tt = segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1]
    h = l1.hessian
    vv = h[1][1] * sin(inclination) ** 2 + h[2][2] * cos(inclination) ** 2
    expected = sqrt(h[0][0] / -(h[1][1] * h[2][2] / vv))
    assert abs(tt / tx) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("q", QI)
@pytest.mark.parametrize("inclination", (0.0, *VIEWS, pi / 2))
def test_binary_contact_limbs_meet_and_follow_frame_transforms(
    reference: bool, q: float, inclination: float
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    for frame in ("binary", "centre_of_mass"):
        b = rocheometry.BinaryGeometry(
            q, accuracy=_Wired(graph), separation=3.0, frame=frame
        )
        u = b.component(1).surface(fill_out_factor=0.5).potential
        view = rocheometry.CentrelinePlane(inclination)
        parts = [b.component(k).surface(potential=u) for k in (1, 2)]
        paths = [part.silhouette_envelope(view).paths[0] for part in parts]
        a, z = paths
        scale = min(part.named_radius("pole") for part in parts)
        bound = (1e-10 if reference else 1e-8) * scale
        assert (
            hypot(
                a.segments[0].p0[0] - z.segments[-1].p3[0],
                a.segments[0].p0[1] - z.segments[-1].p3[1],
            )
            <= bound
        )
        assert (
            hypot(
                a.segments[-1].p3[0] - z.segments[0].p0[0],
                a.segments[-1].p3[1] - z.segments[0].p0[1],
            )
            <= bound
        )
        for part, path in zip(parts, paths, strict=True):
            local = part.intrinsic_geometry.silhouette_envelope(view)
            assert (
                path == local.transformed(part.component._output_transform()).paths[0]
            )


def direct_shadow_value(
    field: rocheometry.ComponentGeometry,
    x: float,
    t: float,
    i: float,
    other_radius: float | None,
) -> float:
    """Independent sampled/golden maximization of the defining potential."""
    outer = abs(field.outer().position[0])
    square = outer * outer - x * x - t * t
    if other_radius is not None:
        square = max(square, other_radius**2 - (x - 1) ** 2 - t * t)
    if square <= 0:
        return -float("inf")
    bound = sqrt(square)
    t = abs(t)

    def value(w: float) -> float:
        if x == t == w == 0:
            return float("inf")
        return field.field.potential(
            x, t * cos(i) + w * sin(i), -t * sin(i) + w * cos(i)
        )

    nodes = [bound * k / 32 for k in range(33)]
    values = [value(w) for w in nodes]
    index = max(range(len(nodes)), key=values.__getitem__)
    if index == 32:
        return values[index]
    lo, hi = nodes[max(0, index - 1)], nodes[index + 1]
    golden = (sqrt(5) - 1) / 2
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = value(a), value(b)
    for _ in range(55):
        if fa > fb:
            hi = b
            b, fb = a, fa
            a = hi - golden * (hi - lo)
            fa = value(a)
        else:
            lo = a
            a, fa = b, fb
            b = lo + golden * (hi - lo)
            fb = value(b)
    return max(values[index], fa, fb)


def row_crossings(points: list[Point2D], y: float) -> list[float]:
    crossings = []
    for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1], strict=True):
        if (ay > y) != (by > y):
            crossings.append(ax + (bx - ax) * (y - ay) / (by - ay))
    return crossings


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
@pytest.mark.parametrize("inclination", VIEWS)
def test_reference_outline_matches_independent_brute_force_shadow(
    qi: float, kind: str, inclination: float
) -> None:
    s = surface(qi, kind, True)
    geo = s.component
    view = rocheometry.CentrelinePlane(inclination)
    paths = list(s.silhouette_envelope(view).paths)
    other_radius = None
    if kind == "contact":
        binary = rocheometry.BinaryGeometry(qi, accuracy=_Wired(geo._computers))
        paths = list(
            binary.component(1)
            .surface(potential=s.potential)
            .silhouette_envelope(view)
            .paths
        )
        paths += list(
            binary.component(2)
            .surface(potential=s.potential)
            .silhouette_envelope(view)
            .paths
        )
        other_radius = abs(binary.geometry2.outer().position[0])
    points = [p for path in paths for p in path.sample(4001)]
    bounds = rocheometry.CompoundBezierPath(tuple(paths)).bounds
    x0, y0, x1, y1 = bounds
    pole = s.named_radius("pole")
    distance = [_PathDistance(path) for path in paths]
    for j in range(81):
        y = y0 + (y1 - y0) * (j + 0.317) / 81
        crossings = row_crossings(points, y)
        for k in range(121):
            x = x0 + (x1 - x0) * (k + 0.193) / 121
            actual = bool(sum(c > x for c in crossings) % 2)
            expected = (
                direct_shadow_value(geo, x, y, inclination, other_radius)
                >= s.potential.value
            )
            if actual != expected:
                assert min(d((x, y)) for d in distance) <= 1e-6 * pole, (
                    qi,
                    kind,
                    inclination,
                    x,
                    y,
                )


@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3))
@pytest.mark.parametrize("kind", ("detached", "critical", "contact"))
def test_fixed_silhouette_controls_vary_smoothly_with_inclination(
    qi: float, kind: str
) -> None:
    s = surface(qi, kind, False)
    controls = []
    for i in (pi / 3 - 1e-6, pi / 3, pi / 3 + 1e-6):
        path = s.silhouette_envelope(rocheometry.CentrelinePlane(i)).paths[0]
        controls.append(
            [p for seg in path.segments for p in (seg.p0, seg.p1, seg.p2, seg.p3)]
        )
    for a, b, c in zip(*controls, strict=True):
        assert hypot(
            a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1]
        ) <= 1e-8 * s.named_radius("pole")


@pytest.mark.parametrize("qi", (1e-3, 0.05, 1.0, 20.0, 1e3, 1e6))
@pytest.mark.parametrize("inclination", (0.0, *VIEWS, pi / 2))
def test_silhouette_controls_continue_through_l1_at_quarter_power(
    qi: float, inclination: float
) -> None:
    """Check fixed-depth decrease, the final bound and transition-relative rate.

    The quarter-power depths are relative to the neck slot's transition.
    Stored depth resolves the transition-relative levels directly on both
    sides of L1, including light stars; no absolute-level rounding is used.
    """
    geo = rocheometry.ComponentGeometry(qi)

    def error(contact: rocheometry.ComponentSurfaceGeometry) -> float:
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        detached = geo._surface_at_depth(-contact._depth)
        paths = []
        for s in (contact, detached):
            paths.append(
                [
                    p
                    for seg in s.silhouette_envelope(
                        rocheometry.CentrelinePlane(inclination)
                    )
                    .paths[0]
                    .segments
                    for p in (seg.p0, seg.p1, seg.p2, seg.p3)
                ]
            )
        return max(
            hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(*paths, strict=True)
        ) / contact.named_radius("pole")

    errors = []
    for factor in (1e-6, 1e-8, 1e-10, 1e-12):
        contact = geo.surface(fill_out_factor=factor)
        assert contact.regime is rocheometry.LevelRegime.CONTACT
        errors.append(error(contact))
    assert len(errors) == 4
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] <= 2.25e-4

    strategy = geo._computers.silhouette
    assert isinstance(strategy, GaugeFixedSilhouettePathComputer)
    m = strategy.segments_per_half - 2
    phi1 = pi * (1 / m - sin(2 * pi / m) / (2 * pi))
    l1 = geo.l1()
    transition = (
        l1.hessian[0][0]
        * l1.position[0] ** 2
        * phi1**2
        / (32 * NECK_SLOT_PLACEMENT_MULTIPLIER**2)
    )
    pair = [geo._surface_at_depth(transition / divisor) for divisor in (100, 10000)]
    assert all(s.regime is rocheometry.LevelRegime.CONTACT for s in pair)
    assert error(pair[1]) <= 0.4 * error(pair[0])


def test_traced_strategy_can_be_used_in_a_custom_default_graph() -> None:
    g = Fitted()._computers()
    strategy = TracedSilhouettePathComputer(
        radius_computer=g.radius,
        l1_computer=g.l1,
        outer_computer=g.outer,
        watershed_computer=g.watershed,
        tolerance=1e-5,
    )
    custom = replace(g, silhouette=strategy)
    s = rocheometry.ComponentGeometry(1, accuracy=_Wired(custom)).lobe()
    assert (
        radial_error(
            s.silhouette(rocheometry.CentrelinePlane(pi / 3)).paths[0],
            projected(s, pi / 3),
        )
        <= 1e-5
    )
