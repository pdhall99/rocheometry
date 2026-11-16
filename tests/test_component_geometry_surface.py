import math

import pytest

import rocheometry
from rocheometry import _component_geometry
from rocheometry._computers.packaged import (
    packaged_lobe_volume_computer,
    packaged_lobe_volume_radius_computer,
)
from rocheometry._computers.surface_potential import SurfacePotentialComputer
from rocheometry._computers.surface_radius import SurfaceRadiusComputer
from rocheometry._roche import RocheField
from rocheometry._roche.volume import sphere_volume
from rocheometry._types import BinaryFrame, ComponentId
from tests._helpers import (
    RadiusDelegate,
    _depth,
    _Wired,
    computer_graph,
    numerical_graph,
    single_path,
)


def test_detached_angular_grid_depends_on_the_level() -> None:
    graph = numerical_graph()
    field = RocheField(0.3)
    critical = graph.l1(field).potential
    detached = rocheometry.Potential.jacobi(critical.value + 0.1)
    critical_sample = graph.sample(field, _depth(field, critical))
    detached_sample = graph.sample(field, _depth(field, detached))
    assert critical_sample.rays != detached_sample.rays
    assert graph.sample(field, _depth(field, critical)) is critical_sample


@pytest.mark.parametrize(
    ("qi", "expected"),
    [(0.3, 0.2802700986393757), (1.0, 0.3798632409274403)],
)
def test_integrated_lobe_radius_matches_corrected_polar_quadrature(
    qi: float, expected: float
) -> None:
    assert rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(numerical_graph())
    ).lobe().volume_radius() == pytest.approx(expected, rel=1.0e-12)


def test_fast_lobe_radius_uses_packaged_volume_calibration() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    expected = packaged_lobe_volume_computer()(geometry.field)
    assert sphere_volume(geometry.lobe_volume_radius()) == pytest.approx(
        expected, rel=2e-15
    )
    assert geometry.lobe_volume_radius() == pytest.approx(
        packaged_lobe_volume_radius_computer()(geometry.field), rel=2e-15
    )


def test_surface_selection_returns_equal_values() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)

    first = geometry.surface(volume_radius_fill=0.8)
    second = geometry.surface(volume_radius_fill=0.8)

    assert first == second
    assert first is not second
    assert first.volume_radius_fill() == pytest.approx(0.8, rel=2e-8)
    assert first.component is geometry


def test_surface_fill_is_validated_once_before_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = rocheometry.ComponentGeometry(
        1.0, accuracy=_Wired(numerical_graph(polar_order=8, azimuth_order=8))
    )
    calls: list[float] = []
    original = _component_geometry._normalize_finite_positive_float

    def normalize(value: float, *, name: str = "volume_radius_fill") -> float:
        calls.append(value)
        return original(value, name=name)

    monkeypatch.setattr(
        _component_geometry, "_normalize_finite_positive_float", normalize
    )

    surface = geometry.surface(volume_radius_fill=0.8)
    surface.volume()
    surface.potential
    surface.radius((0.0, 1.0, 0.0))
    single_path(surface.section()).sample(17)

    assert calls == [0.8]


def test_surface_selection_always_validates_fill() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    geometry.lobe()

    with pytest.raises(TypeError, match="volume_radius_fill"):
        geometry.surface(volume_radius_fill=True)


def test_reference_surface_fill_is_validated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    original = _component_geometry._normalize_finite_positive_float

    def normalize(value: float, *, name: str = "volume_radius_fill") -> float:
        calls.append(value)
        return original(value, name=name)

    monkeypatch.setattr(
        _component_geometry, "_normalize_finite_positive_float", normalize
    )
    surface = rocheometry.ComponentGeometry(
        1.0, accuracy=_Wired(numerical_graph(polar_order=8, azimuth_order=8))
    ).surface(volume_radius_fill=0.8)
    surface.volume()
    surface.radius((0.0, 1.0, 0.0))

    assert calls == [0.8]


@pytest.mark.parametrize("potential", [0.0, None, object()])
def test_direct_surface_constructor_requires_typed_potential(potential: object) -> None:
    with pytest.raises(TypeError, match="potential"):
        rocheometry.ComponentSurfaceGeometry(
            rocheometry.ComponentGeometry(1.0), potential
        )  # type: ignore[arg-type]


def test_lobe_is_exact_critical_surface_alias() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    assert geometry.lobe() == geometry.surface(volume_radius_fill=1.0)


def test_surface_delegates_parent_computer_configuration() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    surface = geometry.surface(volume_radius_fill=0.8)

    assert surface.component._computers is geometry._computers


def test_volume_radius_fill_defines_volume_and_volume_radius_directly() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    fill = 0.8
    surface = geometry.surface(volume_radius_fill=fill)

    assert surface.volume() == pytest.approx(
        geometry.lobe().volume() * fill**3,
        rel=6.0e-8,
    )
    assert surface.volume_radius() == pytest.approx(
        geometry.lobe().volume_radius() * fill,
        rel=2.0e-8,
    )


def test_surface_potential_is_above_critical_for_detached_fill() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)

    critical = geometry.l1().potential
    detached = geometry.surface(volume_radius_fill=0.8).potential

    assert detached.value > critical.value
    assert detached.convention is rocheometry.PotentialConvention.JACOBI


def test_directional_radii_use_three_dimensional_field() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    surface = geometry.surface(volume_radius_fill=0.8)
    level = surface.potential.value

    point = surface.named_radius("point")
    back = surface.named_radius("back")
    side = surface.named_radius("side")
    pole = surface.named_radius("pole")

    field = RocheField(1.0)
    assert field.potential(point, 0.0, 0.0) == pytest.approx(level, abs=5.0e-14)
    assert field.potential(-back, 0.0, 0.0) == pytest.approx(level, abs=5.0e-14)
    assert field.potential(0.0, side, 0.0) == pytest.approx(level, abs=5.0e-14)
    assert field.potential(0.0, 0.0, pole) == pytest.approx(level, abs=5.0e-14)


def test_radius_accepts_arbitrary_direction() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    surface = geometry.surface(volume_radius_fill=0.8)
    direction = (1.0, 2.0, 3.0)

    radius = surface.radius(direction)
    norm = sum(value * value for value in direction) ** 0.5
    unit = tuple(value / norm for value in direction)
    level = surface.potential.value
    field = RocheField(1.0)

    assert field.potential(
        radius * unit[0],
        radius * unit[1],
        radius * unit[2],
    ) == pytest.approx(level, abs=5.0e-14)


def test_critical_point_radius_reuses_l1_without_radius_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = rocheometry.ComponentGeometry(1.0)

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("critical point radius should reuse L1")

    monkeypatch.setattr(type(geometry._computers.radius), "__call__", unexpected)
    assert geometry.lobe().named_radius("point") == geometry.l1().position[0]


def test_directional_radii_share_configured_surface_potential() -> None:
    calls = 0

    class CountedPotential:
        def __init__(self, computer: SurfacePotentialComputer) -> None:
            self.computer = computer

        def __call__(
            self,
            field: RocheField,
            volume_radius_fill: float,
        ) -> float:
            nonlocal calls
            calls += 1
            return self.computer(field, volume_radius_fill)

    base = numerical_graph(polar_order=8, azimuth_order=8)
    geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(
            computer_graph(depth_computer=CountedPotential(base.selection.depth))
        ),
    )
    surface = geometry.surface(volume_radius_fill=0.8)

    assert surface.named_radius("pole") > 0.0
    assert calls == 1

    assert surface.named_radius("side") > 0.0
    assert calls == 1


def test_directional_radii_are_uncached_by_default() -> None:
    calls: list[tuple[float, float, float]] = []

    class CountedRadius:
        def __init__(self, computer: SurfaceRadiusComputer) -> None:
            self.computer = computer

        def __call__(
            self,
            field: RocheField,
            potential: float,
            direction: tuple[float, float, float],
        ) -> float:
            calls.append(direction)
            return self.computer(field, potential, direction)

    base = numerical_graph(polar_order=8, azimuth_order=8)
    geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(
            computer_graph(
                radius_computer=RadiusDelegate(CountedRadius(base.radius)),
                depth_computer=base.selection.depth,
            )
        ),
    )
    surface = geometry.surface(volume_radius_fill=0.8)

    pole = surface.named_radius("pole")
    calls_after_pole = len(calls)
    assert pole > 0.0
    assert calls_after_pole == 1
    assert calls[-1] == (0.0, 0.0, 1.0)

    assert surface.named_radius("pole") == pole
    assert len(calls) == calls_after_pole + 1

    assert surface.named_radius("side") > 0.0
    assert len(calls) == calls_after_pole + 2
    assert calls[-1] == (0.0, 1.0, 0.0)


def test_lobe_volume_computer_is_configured_on_parent_geometry() -> None:
    calls = 0

    def volume(field: RocheField, /) -> float:
        nonlocal calls
        calls += 1
        assert field is geometry.field
        return 0.123

    geometry = rocheometry.ComponentGeometry(
        1.0, accuracy=_Wired(computer_graph(lobe_volume_computer=volume))
    )
    assert sphere_volume(geometry.lobe_volume_radius()) == pytest.approx(
        0.123, rel=2e-15
    )
    assert sphere_volume(geometry.lobe_volume_radius()) == pytest.approx(
        0.123, rel=2e-15
    )
    assert calls == 1
    assert rocheometry.ComponentGeometry(1.0).lobe().volume() != 0.123


def test_lobe_memoization_is_a_graph_wiring_choice() -> None:
    class FixedVolume:
        def __init__(self, value: float) -> None:
            self.value = value
            self.calls = 0

        def __call__(self, field: RocheField, /) -> float:
            self.calls += 1
            return self.value

    first = FixedVolume(0.2)
    second = FixedVolume(0.3)
    first_geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(computer_graph(lobe_volume_computer=first, memoize_lobe=False)),
    )
    second_geometry = rocheometry.ComponentGeometry(
        1.0, accuracy=_Wired(computer_graph(lobe_volume_computer=second))
    )
    assert sphere_volume(first_geometry.lobe_volume_radius()) == pytest.approx(
        0.2, rel=2e-15
    )
    assert sphere_volume(first_geometry.lobe_volume_radius()) == pytest.approx(
        0.2, rel=2e-15
    )
    assert first.calls == 2
    assert sphere_volume(second_geometry.lobe_volume_radius()) == pytest.approx(
        0.3, rel=2e-15
    )
    assert sphere_volume(second_geometry.lobe_volume_radius()) == pytest.approx(
        0.3, rel=2e-15
    )
    assert second.calls == 1


def test_binary_surface_view_only_scales_intrinsic_surface() -> None:
    unit = rocheometry.BinaryGeometry(q=1.0, separation=1.0, frame="binary")
    scaled = rocheometry.BinaryGeometry(q=1.0, separation=3.0, frame="centre_of_mass")
    fill = 0.8
    unit_surface = unit.component(1).surface(volume_radius_fill=fill)
    scaled_surface = scaled.component(1).surface(volume_radius_fill=fill)

    assert scaled_surface.potential == unit_surface.potential
    assert scaled_surface.volume() == pytest.approx(27.0 * unit_surface.volume())
    assert scaled_surface.volume_radius() == pytest.approx(
        3.0 * unit_surface.volume_radius()
    )
    assert scaled_surface.named_radius("pole") == pytest.approx(
        3.0 * unit_surface.named_radius("pole")
    )


@pytest.mark.parametrize("frame", ["binary", "centre_of_mass"])
@pytest.mark.parametrize("component_id", [1, 2])
@pytest.mark.parametrize("coordinate", [0, 1, 2])
@pytest.mark.parametrize("value", [True, False])
def test_binary_radius_rejects_boolean_direction_coordinates(
    frame: BinaryFrame, component_id: ComponentId, coordinate: int, value: bool
) -> None:
    surface = (
        rocheometry.BinaryGeometry(0.4, frame=frame).component(component_id).lobe()
    )
    direction = [1.0, 2.0, 3.0]
    direction[coordinate] = value

    with pytest.raises(
        TypeError, match=rf"direction\[{coordinate}\]: expected a real number, got bool"
    ):
        surface.radius((direction[0], direction[1], direction[2]))


@pytest.mark.parametrize("frame", ["binary", "centre_of_mass"])
@pytest.mark.parametrize("component_id", [1, 2])
@pytest.mark.parametrize("scale", [1.0, 1.0e308, 1.0e-308])
def test_binary_radius_transforms_arbitrary_finite_directions(
    frame: BinaryFrame, component_id: ComponentId, scale: float
) -> None:
    surface = (
        rocheometry.BinaryGeometry(0.4, frame=frame, separation=3.0)
        .component(component_id)
        .lobe()
    )
    direction = (scale, 0.5 * scale, 0.2 * scale)
    local_direction = (
        direction if component_id == 1 else (-direction[0], -direction[1], direction[2])
    )

    assert surface.radius(direction) == pytest.approx(
        3.0 * surface.intrinsic_geometry.radius(local_direction), rel=2.0e-13
    )


def test_component_exchange_preserves_intrinsic_radii() -> None:
    first = rocheometry.BinaryGeometry(q=0.4).geometry1.lobe()
    exchanged = rocheometry.BinaryGeometry(q=2.5).geometry2.lobe()

    assert first.qi == pytest.approx(exchanged.qi)
    for name in ("point", "back", "side", "pole"):
        assert exchanged.named_radius(name) == pytest.approx(
            first.named_radius(name),
            rel=2.0e-13,
        )


@pytest.mark.parametrize("fill", [0.0, -0.1])
def test_surface_selection_rejects_non_detached_fill(fill: float) -> None:
    geometry = rocheometry.ComponentGeometry(1.0)

    with pytest.raises(ValueError, match="volume_radius_fill"):
        geometry.surface(volume_radius_fill=fill)


@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3))
@pytest.mark.parametrize("method", ("section", "silhouette_envelope"))
def test_fitted_outline_controls_are_continuous_in_fill(qi: float, method: str) -> None:
    # Every regime uses one fixed segment layout, and each knot, tangent and arm
    # is a continuous function of the level, so control points move by O(df)
    # away from L1 and still converge, more slowly, at fill one.
    geometry = rocheometry.ComponentGeometry(qi)
    plane = rocheometry.CentrelinePlane(math.pi / 3)
    scale = geometry.lobe_volume_radius()

    def controls(fill: float) -> list[float]:
        outline = getattr(geometry.surface(volume_radius_fill=fill), method)(plane)
        return [
            v
            for path in outline.paths
            for s in path.segments
            for p in (s.p0, s.p1, s.p2, s.p3)
            for v in p
        ]

    def change(fill: float, step: float) -> float:
        a, b = controls(fill), controls(fill + step)
        assert len(a) == len(b)
        return max(abs(u - v) for u, v in zip(a, b, strict=True)) / scale

    contact = 1 + (geometry.max_volume_radius_fill() - 1) / 2
    for fill in (0.3, 0.9, 0.999, contact):
        coarse, fine = change(fill, 1e-5), change(fill, 1e-6)
        assert fine <= 0.2 * coarse + 1e-12, (fill, coarse, fine)
    steps = [change(1.0, step) for step in (1e-4, 1e-5, 1e-6)]
    assert steps[0] > steps[1] > steps[2], steps


DECADES = tuple(10.0**k for k in range(-8, 9))
SHAPE_LEVELS = (("fill", 0.8), ("fill", 1.0), ("F", 0.01), ("F", 0.99))
SHAPE_METHODS = ("section", "section_envelope", "silhouette_envelope")


@pytest.mark.slow
@pytest.mark.parametrize(
    "accuracy", (rocheometry.Fitted(), rocheometry.Reference()), ids=repr
)
@pytest.mark.parametrize("qi", DECADES)
def test_outline_shape_guarantees_at_every_decade(
    qi: float, accuracy: rocheometry.Fitted | rocheometry.Reference
) -> None:
    # Guarantees, not measurements: every outline is nonempty and mirror
    # symmetric, closed unless it is a contact envelope, and starts exactly at
    # L1 on the critical surface. The release run measures the errors.
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    x1 = geometry.l1().position[0]
    for kind, level in SHAPE_LEVELS:
        surface = (
            geometry.surface(volume_radius_fill=level)
            if kind == "fill"
            else geometry.surface(fill_out_factor=level)
        )
        for inclination in (0.0, math.pi / 6, math.pi / 3, math.pi / 2):
            plane = rocheometry.CentrelinePlane(inclination)
            for method in SHAPE_METHODS:
                outline = getattr(surface, method)(plane)
                case = (kind, level, inclination, method)
                assert outline.paths, case
                closed = method == "section" or kind == "fill"
                assert all(p.closed is closed for p in outline.paths), case
                controls = {
                    (s.p0, s.p1, s.p2, s.p3) for p in outline.paths for s in p.segments
                }
                mirror = {tuple((x, -y) for x, y in reversed(c)) for c in controls}
                assert mirror == controls, case
                if kind == "fill" and level == 1.0:
                    assert outline.paths[0].segments[0].p0 == (x1, 0.0), case
