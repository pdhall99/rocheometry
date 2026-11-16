import pytest

from rocheometry._numerics.polynomial import (
    chebyshev_value,
    chebyshev_value_2d,
    power_value,
    power_value_2d,
)


@pytest.mark.parametrize("x", [-1.0, -0.3, 0.0, 0.7, 1.0])
def test_power_and_chebyshev_series_evaluate_the_same_cubic(x: float) -> None:
    expected = 1.0 + 2.0 * x - 3.0 * x**2 + 4.0 * x**3

    assert power_value(x, (1.0, 2.0, -3.0, 4.0)) == pytest.approx(expected)
    assert chebyshev_value(x, (-0.5, 5.0, -1.5, 1.0)) == pytest.approx(expected)


def test_tensor_polynomials_preserve_x_and_y_degree_order() -> None:
    x, y = 0.2, -0.3

    assert power_value_2d(x, y, ((1.0, 2.0), (3.0, 4.0))) == pytest.approx(
        1.0 + 2.0 * y + 3.0 * x + 4.0 * x * y
    )
    assert chebyshev_value_2d(
        x, y, ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (3.0, 0.0, 4.0))
    ) == pytest.approx(
        2.0 * (2.0 * y**2 - 1.0)
        + 3.0 * (2.0 * x**2 - 1.0)
        + 4.0 * (2.0 * x**2 - 1.0) * (2.0 * y**2 - 1.0)
    )
