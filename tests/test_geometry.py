import pytest

import rocheometry
from rocheometry._geometry import AffineTransform2D


def test_affine_transform_is_public_geometry_primitive() -> None:
    assert rocheometry.AffineTransform2D is AffineTransform2D


def test_affine_transform_composes_in_reading_order() -> None:
    transform = AffineTransform2D.translation(1.0, -2.0).then(
        AffineTransform2D.scale(2.0, 3.0)
    )

    assert transform.apply((4.0, 5.0)) == pytest.approx((10.0, 9.0))
    assert transform.apply_vector((4.0, 5.0)) == pytest.approx((8.0, 15.0))


def test_affine_transform_inverse_round_trips() -> None:
    transform = AffineTransform2D(
        xx=2.0,
        xy=0.5,
        yx=-0.25,
        yy=1.5,
        tx=3.0,
        ty=-4.0,
    )
    point = (0.37, -1.2)

    assert transform.inverse().apply(transform.apply(point)) == pytest.approx(point)
    assert transform.then(transform.inverse()).apply(point) == pytest.approx(point)


def test_affine_transform_rejects_nonfinite_coefficients() -> None:
    with pytest.raises(ValueError, match="finite"):
        AffineTransform2D(xx=float("inf"))


@pytest.mark.parametrize("operation", ["point", "vector"])
@pytest.mark.parametrize("coordinate", [0, 1])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_affine_application_rejects_nonfinite_coordinates(
    operation: str, coordinate: int, value: float
) -> None:
    transform = AffineTransform2D.identity()
    point = (value, 1.0) if coordinate == 0 else (1.0, value)
    method = transform.apply if operation == "point" else transform.apply_vector

    with pytest.raises(
        ValueError, match=rf"{operation}\[{coordinate}\]: expected a finite"
    ):
        method(point)


def test_singular_affine_transform_has_no_inverse() -> None:
    with pytest.raises(ValueError, match="singular"):
        AffineTransform2D.scale(0.0, 1.0).inverse()


def test_affine_transform_canonicalizes_numeric_coefficients() -> None:
    transform = AffineTransform2D(xx=2, tx=-3)

    assert transform.xx == 2.0
    assert transform.tx == -3.0
    assert isinstance(transform.xx, float)
    assert isinstance(transform.tx, float)


def test_affine_transform_rejects_bool_coefficient() -> None:
    with pytest.raises(TypeError, match="xx: expected a real number"):
        AffineTransform2D(xx=True)
