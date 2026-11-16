"""Verification of the additive point API, independent of contract studies."""

from __future__ import annotations

import doctest
from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, hypot, inf, nextafter, pi, sin, sqrt, ulp
from random import Random
from sys import float_info
from types import ModuleType
from typing import Any, cast

import pytest
from crosschecks.instrumented.node_backward import copied
from crosschecks.instrumented.node_running import Bound, Replay, field_model
from crosschecks.problem import Binary, exact_values

import rocheometry
from rocheometry import _binary_geometry, _component_geometry
from rocheometry._frames import component_frame, transform
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._types import BinaryFrame, ComponentId


def residual_and_bound(
    geometry: rocheometry.ComponentGeometry, point: Point3D, depth: float | None = None
) -> Any:
    """Replay the actual factored Cartesian residual, with no tuned multiplier."""
    replay = Replay()
    model, namespace = field_model(geometry.field, replay)
    namespace["hypot"] = replay.hypot
    difference = copied(RocheField._axis_potential_difference, namespace)
    value: Any = difference(
        model, tuple(Bound(v) for v in point), Bound(geometry.l1().position[0])
    )
    assert float(value) == geometry.field.potential_difference(
        point, geometry.l1().position
    )
    if depth is None:
        return value
    residual: Any = replay.fsum((value, Bound(depth)))
    assert float(residual) == fsum((float(value), depth))
    return residual


@pytest.mark.parametrize("qi", [1e-8, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e8])
@pytest.mark.parametrize("accuracy", [rocheometry.Fitted(), rocheometry.Reference()])
def test_potential_at_100_digit_running_bound(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    rng = Random(2026102901)
    x1 = geometry.l1().position[0]
    points = [(x1, 0.0, 0.0), (nextafter(x1, 0.0), 1e-10 * x1, 0.0)]
    points += [
        (rng.uniform(-1, 2), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(20)
    ]
    for point in points:
        difference = residual_and_bound(geometry, point)
        report = geometry.potential_at(point)
        assert report.convention is rocheometry.PotentialConvention.JACOBI
        anchor = geometry.field._axis_potential_exact(x1)
        exact_sum = anchor + Fraction(float(difference))
        assert report.value == float(exact_sum)
        rounding = abs(Fraction(report.value) - exact_sum)
        bound = nextafter(difference.error + float(rounding), inf)
        with localcontext() as ctx:
            ctx.prec = 100
            truth, _ = exact_values(
                Binary.of(geometry.field), tuple(map(Fraction, point))
            )
            assert abs(Decimal(report.value) - truth) <= Decimal(bound)


@pytest.mark.parametrize("accuracy", [rocheometry.Fitted(), rocheometry.Reference()])
@pytest.mark.parametrize("kind", ["detached", "critical", "contact", "shallow"])
def test_every_balanced_quadrature_node_just_inside_and_outside(
    accuracy: rocheometry.Fitted | rocheometry.Reference, kind: str
) -> None:
    """Balanced-grid nominal labels agree at four epsilons."""
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=accuracy)
    surface = (
        geometry.surface(volume_radius_fill=0.8)
        if kind == "detached"
        else geometry.lobe()
        if kind == "critical"
        else geometry.surface(fill_out_factor=0.5)
        if kind == "contact"
        else geometry._surface_at_depth(1e-20)
    )
    for point in surface.quadrature().positions:
        for inside, scale in [
            (True, 1 - 4 * float_info.epsilon),
            (False, 1 + 4 * float_info.epsilon),
        ]:
            probe = cast(Point3D, tuple(v * scale for v in point))
            assert surface.contains(probe) is inside


@pytest.mark.parametrize("qi", [1e-8, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e8])
@pytest.mark.parametrize("accuracy", [rocheometry.Fitted(), rocheometry.Reference()])
def test_same_ray_radius_consistency_and_bounded_region(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    for surface in [
        geometry.surface(volume_radius_fill=0.8),
        geometry.lobe(),
        geometry._surface_at_depth(1e-20),
        geometry.surface(fill_out_factor=0.5),
    ]:
        assert surface.contains((0.0, 0.0, 0.0))
        assert not surface.contains((1.0, 0.0, 0.0))
        assert not surface.contains((100.0, 0.0, 0.0))
        for direction in [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (-2.0, 1.0, 1.0),
        ]:
            r = surface.radius(direction)
            unit = tuple(v / hypot(*direction) for v in direction)
            for scale, inside in [
                (0.5, True),
                (1 - 1e-6, True),
                (1 + 1e-6, False),
                (2.0, False),
            ]:
                point = cast(Point3D, tuple(scale * r * d for d in unit))
                assert surface.contains(point) is inside, (
                    qi,
                    surface._depth,
                    direction,
                    scale,
                )


@pytest.mark.parametrize("qi", [1e-3, 0.1, 1.0, 10.0, 1e3])
@pytest.mark.parametrize("accuracy", [rocheometry.Fitted(), rocheometry.Reference()])
@pytest.mark.parametrize("depth", [1e-12, 1e-16, 1e-20])
def test_shallow_neck_and_watershed(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference, depth: float
) -> None:
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    surface = geometry._surface_at_depth(depth)
    x1 = geometry.l1().position[0]
    h = geometry.l1().hessian
    for angle in [0.0, pi / 4, pi / 2]:
        cy, sz = sin(pi / 2 - angle), sin(angle)
        rho = sqrt(depth / (-h[1][1] * cy * cy - h[2][2] * sz * sz))
        height = geometry._computers.watershed.cut_height(geometry.field, rho, angle)
        # Several representable steps on either side avoid a rounded chart tie.
        step = 32 * ulp(height)
        own = (height - step, rho * cy, rho * sz)
        other = (height + step, rho * cy, rho * sz)
        assert surface.contains(own), (qi, depth, angle, x1)
        assert not surface.contains(other), (qi, depth, angle, x1)


@pytest.mark.parametrize("frame", ["binary", "centre_of_mass"])
@pytest.mark.parametrize("separation", [0.25, 1.0, 7.0])
@pytest.mark.parametrize("q", [0.1, 1.0, 10.0])
def test_binary_frames_and_both_component_charts(
    frame: BinaryFrame, separation: float, q: float
) -> None:
    binary = rocheometry.BinaryGeometry(q, separation=separation, frame=frame)
    for component_id in (cast(ComponentId, 1), cast(ComponentId, 2)):
        component = binary.component(component_id)
        for local_surface in [
            component.intrinsic_geometry.surface(volume_radius_fill=0.8),
            component.intrinsic_geometry._surface_at_depth(1e-20),
        ]:
            view_surface = rocheometry.BinaryComponentSurfaceView._from_depth(
                component, local_surface._depth
            )
            for direction in [(1.0, 0.0, 0.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 1.0)]:
                r = local_surface.radius(direction)
                for scale in [0.5, 1.5]:
                    local = cast(
                        Point3D,
                        tuple(v * r * scale / hypot(*direction) for v in direction),
                    )
                    view = component._output_position(local)
                    assert view_surface.contains(view) is local_surface.contains(local)
                    # Express that same spatial point first in the other star's
                    # chart, then transform to the queried component's chart.
                    other = transform(
                        local,
                        source=component_frame(component_id),
                        target=component_frame(2 if component_id == 1 else 1),
                        q=q,
                    )
                    restored = transform(
                        other,
                        source=component_frame(2 if component_id == 1 else 1),
                        target=component_frame(component_id),
                        q=q,
                    )
                    assert local_surface.contains(restored) is local_surface.contains(
                        local
                    )
                    actual = binary.potential_at(view).value
                    expected = component.intrinsic_geometry.potential_at(local).value
                    assert abs(actual - expected) <= 4 * ulp(expected)


@pytest.mark.parametrize(
    "geometry", [rocheometry.ComponentGeometry(1.0), rocheometry.BinaryGeometry(1.0)]
)
def test_point_input_validation(
    geometry: rocheometry.ComponentGeometry | rocheometry.BinaryGeometry,
) -> None:
    for point in [
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (1.0, 2.0),
        [0.0, 0.0, 0.0],
        (True, 0.0, 0.0),
    ]:
        with pytest.raises((TypeError, ValueError)):
            geometry.potential_at(cast(Point3D, point))
    for point in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]:
        with pytest.raises(ValueError, match="singular"):
            geometry.potential_at(point)
    surface = (
        geometry.lobe()
        if isinstance(geometry, rocheometry.ComponentGeometry)
        else geometry.component(1).lobe()
    )
    with pytest.raises(ValueError):
        surface.contains((float("inf"), 0.0, 0.0))


@pytest.mark.parametrize("module", [_component_geometry, _binary_geometry])
def test_point_query_doctests(module: ModuleType) -> None:
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
    assert result.attempted > 0


@pytest.mark.parametrize("frame", ["binary", "centre_of_mass"])
@pytest.mark.parametrize("separation", [0.25, 1.0, 7.0])
@pytest.mark.parametrize("q", [1e-8, 0.1, 1.0, 10.0, 1e8])
def test_binary_potential_100_digit_bound_includes_frame_arithmetic(
    frame: BinaryFrame, separation: float, q: float
) -> None:
    """Replay scale, transform construction/application and factored reporting."""
    from types import SimpleNamespace

    from rocheometry._frames import frame_transform
    from rocheometry._geometry import AffineTransform2D

    replay = Replay()
    translation: Any = Bound(0.0)
    if frame == "centre_of_mass":
        translation = Bound(1.0) / (Bound(1.0) + Bound(q))

    def matrix(tx: Any) -> Any:
        return SimpleNamespace(
            xx=Bound(1.0),
            xy=Bound(0.0),
            yx=Bound(0.0),
            yy=Bound(1.0),
            tx=tx,
            ty=Bound(0.0),
        )

    source = matrix(translation)
    target = matrix(Bound(0.0))
    namespace = dict(vars(__import__("rocheometry._geometry", fromlist=[""])))
    namespace["AffineTransform2D"] = SimpleNamespace
    composed: Any = copied(AffineTransform2D.then, namespace)(source, target)
    actual_transform = frame_transform(source=frame, target="component1", q=q)
    for name in ("xx", "xy", "yx", "yy", "tx", "ty"):
        assert float(getattr(composed, name)) == getattr(actual_transform, name)
    apply = copied(AffineTransform2D.apply, namespace)
    binary = rocheometry.BinaryGeometry(q, separation=separation, frame=frame)
    geometry = binary.geometry1
    rng = Random(2026102902)
    for _ in range(12):
        point = (rng.uniform(-0.4, 0.8), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6))
        scaled = tuple(Bound(v) / Bound(separation) for v in point)
        x, y = apply(composed, (scaled[0], scaled[1]))
        local = (x, y, scaled[2])
        model, field_namespace = field_model(geometry.field, replay)
        field_namespace["hypot"] = replay.hypot
        difference: Any = copied(
            RocheField._axis_potential_difference, field_namespace
        )(model, local, Bound(geometry.l1().position[0]))
        report = binary.potential_at(point)
        anchor = geometry.field._axis_potential_exact(geometry.l1().position[0])
        exact_sum = anchor + Fraction(float(difference))
        assert report.value == float(exact_sum)
        rounding = abs(Fraction(report.value) - exact_sum)
        bound = nextafter(difference.error + float(rounding), inf)
        exact_local = [Fraction(v) / Fraction(separation) for v in point]
        if frame == "centre_of_mass":
            exact_local[0] += Fraction(1) / (1 + Fraction(q))
        with localcontext() as ctx:
            ctx.prec = 100
            truth, _ = exact_values(Binary.of(geometry.field), exact_local)
            assert abs(Decimal(report.value) - truth) <= Decimal(bound)


def test_light_component_near_opening_reproducer() -> None:
    surface = rocheometry.ComponentGeometry(1e-8).surface(fill_out_factor=1 - 1e-6)
    assert (
        sum(
            not surface.contains(cast(Point3D, tuple(v * (1 - 1e-9) for v in p)))
            for p in surface.quadrature().positions
        )
        == 0
    )


def test_noncanonical_queries_keep_l1_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    from rocheometry import _component_geometry
    from rocheometry._computers.surface_radius import radial

    class AlternateField(RocheField):
        def potential(self, x: float, y: float = 0.0, z: float = 0.0) -> float:  # type: ignore[missing-override-decorator]
            return super().potential(x, y, z)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Canonical anchor helper used for a non-canonical field")

    monkeypatch.setattr(_component_geometry, "_residual_anchor", forbidden)
    monkeypatch.setattr(radial, "_residual_anchor", forbidden)
    geometry = rocheometry.ComponentGeometry(1.0)
    surface = geometry.surface(volume_radius_fill=0.8)
    field = AlternateField(1.0)
    object.__setattr__(geometry, "_field", field)
    assert surface.contains((0.1, 0.0, 0.0))
    assert geometry._computers.radius(field, surface._depth, (0.0, 0.0, 1.0)) > 0
