import math
from dataclasses import replace

import pytest

import rocheometry
from rocheometry import Fitted
from tests._helpers import single_path


def _control_points(path: rocheometry.BezierPath) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for segment in path.segments:
        points.extend((segment.p0, segment.p1, segment.p2, segment.p3))
    return tuple(points)


def _assert_points_close(
    actual: tuple[tuple[float, float], ...],
    expected: tuple[tuple[float, float], ...],
    *,
    abs_tol: float = 1.0e-12,
) -> None:
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected, strict=True):
        assert actual_point == pytest.approx(expected_point, abs=abs_tol)


def test_geometry_hierarchy_types_are_public() -> None:
    geometry = rocheometry.ComponentGeometry(0.37)
    surface = geometry.surface(volume_radius_fill=0.8)
    binary = rocheometry.BinaryGeometry(q=0.37)
    component = binary.component(component_id=1)
    binary_surface = component.surface(volume_radius_fill=0.8)

    assert isinstance(surface, rocheometry.ComponentSurfaceGeometry)
    assert isinstance(component, rocheometry.BinaryComponentView)
    assert isinstance(binary_surface, rocheometry.BinaryComponentSurfaceView)


def test_binary_lobe_returns_surface_with_closed_fifty_two_segment_section() -> None:
    lobe = rocheometry.BinaryGeometry(q=0.37).component(1).lobe()
    path = single_path(lobe.section())

    assert isinstance(lobe, rocheometry.BinaryComponentSurfaceView)
    assert isinstance(path, rocheometry.BezierPath)
    assert path.closed
    assert len(path.segments) == 52
    assert path.segments[-1].p3 == path.segments[0].p0


def test_binary_lobe_matches_critical_surface_selection() -> None:
    component = rocheometry.BinaryGeometry(
        q=0.37,
        separation=2.5,
        frame="centre_of_mass",
    ).component(2)
    lobe = component.lobe()
    selected = component.surface(volume_radius_fill=1.0)

    assert lobe == selected
    assert lobe is not selected
    assert lobe.intrinsic_geometry == component.intrinsic_geometry.lobe()
    assert single_path(lobe.section()) == single_path(selected.section())


def test_binary_views_scale_every_intrinsic_measure() -> None:
    component = rocheometry.BinaryGeometry(q=0.37, separation=2.5).component(2)
    surface = component.surface(volume_radius_fill=0.8)
    intrinsic = surface.intrinsic_geometry
    lengths = ("volume_radius", "area_radius", "mean_curvature_radius")

    for name in lengths:
        assert getattr(surface, name)() == pytest.approx(
            2.5 * getattr(intrinsic, name)()
        )
    assert surface.integrated_mean_curvature() == pytest.approx(
        2.5 * intrinsic.integrated_mean_curvature()
    )
    assert component.lobe_volume_radius_slope() == (
        component.intrinsic_geometry.lobe_volume_radius_slope()
    )


def test_binary_views_offer_every_component_measure() -> None:
    def public(cls: type) -> set[str]:
        return {name for name in dir(cls) if not name.startswith("_")}

    # The binary holds the accuracy and point queries; field and qi are
    # normalized, component-local concepts. Both surfaces name their parent
    # `component`.
    assert public(rocheometry.ComponentGeometry) - public(
        rocheometry.BinaryComponentView
    ) == {
        "accuracy",
        "field",
        "potential_at",
        "qi",
    }
    assert public(rocheometry.ComponentSurfaceGeometry) - public(
        rocheometry.BinaryComponentSurfaceView
    ) == {"qi"}


def test_component_local_section_uses_exact_reflection_for_lower_half() -> None:
    path = single_path(
        rocheometry.ComponentGeometry(0.37).surface(volume_radius_fill=0.8).section()
    )
    upper = path.segments[:26]
    lower = path.segments[26:]

    for original, reflected in zip(reversed(upper), lower, strict=True):
        assert reflected.p0 == (original.p3[0], -original.p3[1])
        assert reflected.p1 == (original.p2[0], -original.p2[1])
        assert reflected.p2 == (original.p1[0], -original.p1[1])
        assert reflected.p3 == (original.p0[0], -original.p0[1])


def test_component_exchange_preserves_local_section() -> None:
    first = single_path(
        rocheometry.BinaryGeometry(q=0.03)
        .geometry1.surface(volume_radius_fill=0.7)
        .section()
    )
    swapped = single_path(
        rocheometry.BinaryGeometry(q=1.0 / 0.03)
        .geometry2.surface(volume_radius_fill=0.7)
        .section()
    )

    _assert_points_close(
        _control_points(first),
        _control_points(swapped),
        abs_tol=2.0e-13,
    )


def test_separation_scales_binary_section() -> None:
    unit = single_path(
        rocheometry.BinaryGeometry(q=0.4, separation=1.0)
        .component(2)
        .surface(volume_radius_fill=0.6)
        .section()
    )
    scaled = single_path(
        rocheometry.BinaryGeometry(q=0.4, separation=3.5)
        .component(2)
        .surface(volume_radius_fill=0.6)
        .section()
    )

    expected = tuple((3.5 * x, 3.5 * y) for x, y in _control_points(unit))
    _assert_points_close(_control_points(scaled), expected)


def test_component_two_binary_frame_is_rotated_and_translated() -> None:
    q = 0.4
    separation = 2.5
    local = single_path(
        rocheometry.ComponentGeometry(1.0 / q).surface(volume_radius_fill=0.6).section()
    )
    binary = single_path(
        rocheometry.BinaryGeometry(q=q, separation=separation)
        .component(2)
        .surface(volume_radius_fill=0.6)
        .section()
    )

    expected = tuple(
        (separation - separation * x, -separation * y)
        for x, y in _control_points(local)
    )
    _assert_points_close(_control_points(binary), expected)


def test_com_frame_is_binary_frame_shifted_by_barycentre() -> None:
    q = 0.4
    separation = 2.5
    binary = single_path(
        rocheometry.BinaryGeometry(q=q, separation=separation, frame="binary")
        .component(1)
        .lobe()
        .section()
    )
    com = single_path(
        rocheometry.BinaryGeometry(q=q, separation=separation, frame="centre_of_mass")
        .component(1)
        .lobe()
        .section()
    )
    offset = separation / (1.0 + q)

    expected = tuple((x - offset, y) for x, y in _control_points(binary))
    _assert_points_close(_control_points(com), expected)


def test_path_sampling_and_bounds() -> None:
    binary = rocheometry.BinaryGeometry(q=1.0)
    path = single_path(binary.component(1).lobe().section())
    samples = path.sample(401)
    xmin, ymin, xmax, ymax = binary.lobe_bounds()

    assert len(samples) == 401
    assert samples[0] == samples[-1]
    assert xmin <= path.bounds[0]
    assert ymin <= path.bounds[1]
    assert xmax >= path.bounds[2]
    assert ymax >= path.bounds[3]


def test_binary_geometry_constructs_reciprocal_component_geometries() -> None:
    binary = rocheometry.BinaryGeometry(
        0.25,
        separation=4.0,
        frame="centre_of_mass",
    )

    assert binary.q == 0.25
    assert binary.separation == 4.0
    assert binary.frame == "centre_of_mass"
    assert binary.geometry1.qi == 0.25
    assert binary.geometry2.qi == 4.0
    assert (
        binary.geometry1._computers is binary.geometry2._computers is binary._computers
    )
    assert binary.component(1).position == pytest.approx((-3.2, 0.0, 0.0))
    assert binary.component(2).position == pytest.approx((0.8, 0.0, 0.0))
    assert binary.component(1).intrinsic_geometry is binary.geometry1
    assert binary.component(2).intrinsic_geometry is binary.geometry2


def test_binary_geometry_is_the_only_public_binary_constructor() -> None:
    assert not hasattr(rocheometry, "binary")
    assert "binary" not in rocheometry.__all__


def test_binary_geometry_values_share_fitted_computers() -> None:
    first = rocheometry.BinaryGeometry(0.25)
    second = rocheometry.BinaryGeometry(0.25)

    assert first == second and first is not second
    assert hash(first) == hash(second)
    assert first.geometry1 == second.geometry1
    assert first.geometry1 is not second.geometry1
    assert first._computers is second._computers is Fitted()._computers()


def test_replacing_binary_mass_ratio_rebuilds_reciprocal_components() -> None:
    original = rocheometry.BinaryGeometry(0.25, separation=4.0, frame="centre_of_mass")
    changed = replace(original, q=2.0)

    assert changed.q == 2.0
    assert changed.geometry1.qi == 2.0
    assert changed.geometry2.qi == 0.5
    assert changed._computers is original._computers
    assert (
        changed.geometry1._computers
        is changed.geometry2._computers
        is changed._computers
    )
    assert changed.separation == original.separation
    assert changed.frame == original.frame
    assert original.geometry1.qi == 0.25
    assert original.geometry2.qi == 4.0


@pytest.mark.parametrize("q", [0.0, -1.0, math.nan, math.inf])
def test_binary_geometry_rejects_invalid_mass_ratio(q: float) -> None:
    with pytest.raises(ValueError, match="q"):
        rocheometry.BinaryGeometry(q)


@pytest.mark.parametrize("q", [True, "1.0"])
def test_binary_geometry_rejects_wrong_mass_ratio_type(q: object) -> None:
    with pytest.raises(TypeError, match="q: expected a real number"):
        rocheometry.BinaryGeometry(q)  # type: ignore[arg-type]


@pytest.mark.parametrize("accuracy", [None, object()])
def test_binary_geometry_rejects_wrong_accuracy_type(accuracy: object) -> None:
    with pytest.raises(TypeError, match="accuracy: expected _Accuracy"):
        rocheometry.BinaryGeometry(0.25, accuracy=accuracy)  # type: ignore[arg-type]


def test_binary_surface_derives_equal_intrinsic_surface() -> None:
    component = rocheometry.BinaryGeometry(q=0.4).component(1)
    surface = component.surface(volume_radius_fill=0.8)

    assert surface.volume_radius_fill() == pytest.approx(0.8, rel=2e-8)
    assert surface.intrinsic_geometry == component.intrinsic_geometry.surface(
        volume_radius_fill=0.8
    )


def test_binary_views_are_cheap_noncanonical_values() -> None:
    binary = rocheometry.BinaryGeometry(q=0.4)
    first_component = binary.component(1)
    second_component = binary.component(1)

    assert first_component == second_component
    assert first_component is not second_component

    first_surface = first_component.surface(volume_radius_fill=0.8)
    second_surface = first_component.surface(volume_radius_fill=0.8)

    assert first_surface == second_surface
    assert first_surface is not second_surface
    assert first_surface.intrinsic_geometry == second_surface.intrinsic_geometry


def test_binary_radius_uses_output_frame_direction() -> None:
    separation = 2.5
    surface = (
        rocheometry.BinaryGeometry(q=0.4, separation=separation, frame="binary")
        .component(2)
        .surface(volume_radius_fill=0.8)
    )

    assert surface.radius((1.0, 0.0, 0.0)) == pytest.approx(
        separation * surface.intrinsic_geometry.named_radius("back")
    )
    assert surface.radius((-1.0, 0.0, 0.0)) == pytest.approx(
        separation * surface.intrinsic_geometry.named_radius("point")
    )
    assert surface.named_radius(name="point") == surface.radius((-1.0, 0.0, 0.0))
    assert surface.named_radius(name="back") == surface.radius((1.0, 0.0, 0.0))


def test_equal_mass_binary_is_mirror_symmetric_about_midpoint() -> None:
    binary = rocheometry.BinaryGeometry(q=1.0, separation=1.0, frame="binary")
    first = single_path(binary.component(1).lobe().section()).sample(201)
    second = single_path(binary.component(2).lobe().section()).sample(201)
    reflected_first = tuple((1.0 - x, -y) for x, y in first)

    for point in reflected_first:
        assert min(math.hypot(point[0] - x, point[1] - y) for x, y in second) < 1.0e-12


def test_surface_operations_are_not_flattened_onto_parent_geometries() -> None:
    for name in (
        "surface_potential",
        "volume",
        "volume_radius",
        "radius",
        "named_radius",
        "section",
    ):
        assert not hasattr(rocheometry.ComponentGeometry, name)

    for name in (
        "lobe_volume",
        "lobe_volume_radius",
        "surface_potential",
        "volume",
        "volume_radius",
        "radius",
        "named_radius",
        "section",
        "lobe",
    ):
        assert not hasattr(rocheometry.BinaryGeometry, name)


@pytest.mark.parametrize("q", [1.0e-5, 1.0e5])
def test_default_section_traces_values_outside_fit_domain(q: float) -> None:
    geometry = rocheometry.ComponentGeometry(q)

    assert geometry.qi == q
    assert 0.0 < geometry.l1().position[0] < 1.0
    assert single_path(geometry.lobe().section()).closed


@pytest.mark.parametrize("separation", [0.0, -1.0, math.inf, math.nan])
def test_binary_rejects_invalid_separation(separation: float) -> None:
    with pytest.raises(ValueError, match="separation"):
        rocheometry.BinaryGeometry(q=1.0, separation=separation)


def test_binary_rejects_component_local_frame() -> None:
    with pytest.raises(ValueError, match="frame: expected"):
        rocheometry.BinaryGeometry(q=1.0, frame="component")  # type: ignore[arg-type]


@pytest.mark.parametrize("q", [0.0, -1.0, math.inf, math.nan])
def test_component_geometry_rejects_invalid_mass_ratio(q: float) -> None:
    with pytest.raises(ValueError, match="qi:"):
        rocheometry.ComponentGeometry(q)


@pytest.mark.parametrize("component_id", [0, 3])
def test_binary_geometry_rejects_invalid_component(component_id: int) -> None:
    binary = rocheometry.BinaryGeometry(q=1.0)

    with pytest.raises(ValueError, match="component_id: expected 1 or 2"):
        binary.component(component_id=component_id)  # type: ignore[arg-type]


def test_binary_geometry_rejects_bool_component() -> None:
    binary = rocheometry.BinaryGeometry(q=1.0)

    with pytest.raises(TypeError, match="component_id: expected int, got bool"):
        binary.component(True)  # type: ignore[arg-type]


def test_public_cubic_evaluates_endpoints() -> None:
    segment = rocheometry.CubicBezier(
        p0=(0.0, 0.0),
        p1=(1.0, 0.0),
        p2=(1.0, 1.0),
        p3=(2.0, 1.0),
    )

    assert segment.evaluate(0.0) == (0.0, 0.0)
    assert segment.evaluate(1.0) == (2.0, 1.0)


@pytest.mark.parametrize("value", [True, "1.0"])
def test_public_api_rejects_wrong_mass_ratio_type(value: object) -> None:
    with pytest.raises(TypeError, match="qi: expected a real number"):
        rocheometry.ComponentGeometry(value)  # type: ignore[arg-type]


def test_binary_geometry_rejects_bool_separation() -> None:
    with pytest.raises(TypeError, match="separation: expected a real number"):
        rocheometry.BinaryGeometry(
            1.0,
            separation=True,
        )
