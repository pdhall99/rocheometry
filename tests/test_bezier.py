from math import sqrt

import pytest

from rocheometry._bezier import BezierPath, CubicBezier
from rocheometry._geometry import AffineTransform2D


def _segment() -> CubicBezier:
    return CubicBezier(
        p0=(0.0, 0.0),
        p1=(0.3, 0.8),
        p2=(0.7, -0.2),
        p3=(1.0, 1.0),
    )


@pytest.mark.parametrize("scale", [1.0e-300, 1.0e-15, 1.0, 1.0e150, 1.0e308])
def test_bezier_bounds_preserve_linear_derivative_extrema_at_any_scale(
    scale: float,
) -> None:
    curve = CubicBezier((0.0, 0.0), (scale, 0.0), (scale, 0.0), (0.0, 0.0))
    bounds = BezierPath((curve,)).bounds
    assert bounds == pytest.approx((0.0, 0.0, 0.75 * scale, 0.0), rel=2.0e-15, abs=0.0)
    assert bounds[2] == curve.evaluate(0.5)[0]


@pytest.mark.parametrize("scale", [1.0e-300, 1.0e-15, 1.0, 1.0e150, 1.0e308])
def test_bezier_bounds_preserve_quadratic_derivative_extrema_at_any_scale(
    scale: float,
) -> None:
    curve = CubicBezier((0.0, 0.0), (0.0, scale), (scale, -scale), (0.0, 0.0))
    maximum_y = sqrt(3.0) / 6.0 * scale
    assert BezierPath((curve,)).bounds == pytest.approx(
        (0.0, -maximum_y, 4.0 / 9.0 * scale, maximum_y), rel=2.0e-15, abs=0.0
    )


def test_bezier_bounds_handle_overflowing_control_point_differences() -> None:
    curve = CubicBezier(
        (-1.0e308, 0.0), (1.0e308, 0.0), (1.0e308, 0.0), (-1.0e308, 0.0)
    )
    assert BezierPath((curve,)).bounds == pytest.approx(
        (-1.0e308, 0.0, 5.0e307, 0.0), rel=2.0e-15, abs=0.0
    )


def test_bezier_bounds_handle_constant_coordinates() -> None:
    curve = CubicBezier((3.0, -2.0), (3.0, -2.0), (3.0, -2.0), (3.0, -2.0))
    assert BezierPath((curve,)).bounds == (3.0, -2.0, 3.0, -2.0)


@pytest.mark.parametrize("operation", ["evaluate", "split"])
@pytest.mark.parametrize("t", [-0.01, 2.0, float("nan"), float("inf"), -float("inf")])
def test_bezier_parameter_methods_reject_values_outside_the_segment(
    operation: str, t: float
) -> None:
    segment = _segment()
    method = segment.evaluate if operation == "evaluate" else segment.split

    with pytest.raises(ValueError, match=r"t: expected a value in \[0, 1\]"):
        method(t)


@pytest.mark.parametrize("t", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_affine_transform_commutes_with_bezier_evaluation(t: float) -> None:
    segment = _segment()
    transform = AffineTransform2D(
        xx=-2.0,
        xy=0.25,
        yx=0.5,
        yy=1.5,
        tx=4.0,
        ty=-3.0,
    )

    assert segment.transformed(transform).evaluate(t) == pytest.approx(
        transform.apply(segment.evaluate(t))
    )


@pytest.mark.parametrize("t", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_reversed_bezier_reverses_parametrization(t: float) -> None:
    segment = _segment()

    assert segment.reversed().evaluate(t) == pytest.approx(segment.evaluate(1.0 - t))


def test_split_bezier_preserves_curve() -> None:
    segment = _segment()
    split = 0.3
    left, right = segment.split(split)

    assert left.p0 == segment.p0
    assert left.p3 == right.p0
    assert right.p3 == segment.p3
    for local_t in (0.0, 0.25, 0.5, 1.0):
        assert left.evaluate(local_t) == pytest.approx(
            segment.evaluate(split * local_t)
        )
        assert right.evaluate(local_t) == pytest.approx(
            segment.evaluate(split + (1.0 - split) * local_t)
        )


def test_bezier_path_transforms_and_reverses_as_one_object() -> None:
    first = _segment()
    second = CubicBezier(
        p0=first.p3,
        p1=(1.2, 1.3),
        p2=(1.7, 0.6),
        p3=(2.0, 0.0),
    )
    path = BezierPath((first, second), closed=False)
    transform = AffineTransform2D.translation(3.0, -1.0)

    transformed = path.transformed(transform)
    reversed_path = path.reversed()

    assert transformed.segments[0].p0 == transform.apply(first.p0)
    assert transformed.segments[-1].p3 == transform.apply(second.p3)
    assert reversed_path.segments[0] == second.reversed()
    assert reversed_path.segments[-1] == first.reversed()


def test_cubic_bezier_canonicalizes_control_points() -> None:
    segment = CubicBezier(
        p0=(0, 0),
        p1=(1, 0),
        p2=(1, 1),
        p3=(2, 1),
    )

    assert segment.p0 == (0.0, 0.0)
    assert all(
        isinstance(value, float)
        for point in (segment.p0, segment.p1, segment.p2, segment.p3)
        for value in point
    )


def test_cubic_bezier_rejects_wrong_control_point_type() -> None:
    with pytest.raises(TypeError, match="p0: expected tuple"):
        CubicBezier(
            p0=[0.0, 0.0],  # type: ignore[arg-type]
            p1=(1.0, 0.0),
            p2=(1.0, 1.0),
            p3=(2.0, 1.0),
        )


def test_bezier_path_rejects_bool_sample_count() -> None:
    path = BezierPath((_segment(),), closed=False)

    with pytest.raises(TypeError, match="n: expected int, got bool"):
        path.sample(True)


def test_bezier_path_rejects_wrong_closed_type() -> None:
    with pytest.raises(TypeError, match="closed: expected bool"):
        BezierPath((_segment(),), closed=1)  # type: ignore[arg-type]
