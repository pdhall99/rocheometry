from math import sqrt

import pytest

from rocheometry._numerics.quadrature import gauss_legendre


def test_gauss_legendre_two_point_rule() -> None:
    nodes, weights = gauss_legendre(2)

    assert nodes == pytest.approx((-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)))
    assert weights == pytest.approx((1.0, 1.0))


def test_gauss_legendre_integrates_quartic_with_three_points() -> None:
    nodes, weights = gauss_legendre(3)

    integral = sum(
        weight * node**4 for node, weight in zip(nodes, weights, strict=True)
    )
    assert integral == pytest.approx(2.0 / 5.0)


def test_gauss_legendre_is_cached() -> None:
    first = gauss_legendre(8)
    second = gauss_legendre(8)

    assert second is first


def test_gauss_legendre_cache_evicts_least_recently_used_orders() -> None:
    gauss_legendre.cache_clear()
    try:
        capacity = gauss_legendre.cache_info().maxsize
        assert capacity is not None and capacity >= 2
        rules = [gauss_legendre(order) for order in range(1, capacity + 1)]
        assert gauss_legendre(1) is rules[0]
        gauss_legendre(capacity + 1)
        assert gauss_legendre.cache_info().currsize == capacity
        assert gauss_legendre(1) is rules[0]
        assert gauss_legendre(2) is not rules[1]
        assert gauss_legendre.cache_info().currsize == capacity
    finally:
        gauss_legendre.cache_clear()


@pytest.mark.parametrize(("order", "invalid"), [(1, True), (8, 8.0)])
def test_gauss_legendre_rejects_wrong_order_type_with_a_warm_cache(
    order: int, invalid: object
) -> None:
    gauss_legendre(order=order)
    with pytest.raises(TypeError, match="order: expected int"):
        gauss_legendre(order=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("order", [0, -1])
def test_gauss_legendre_rejects_invalid_order(order: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        gauss_legendre(order)


@pytest.mark.parametrize("order", [True, 2.5])
def test_gauss_legendre_rejects_wrong_order_type(order: object) -> None:
    with pytest.raises(TypeError, match="order: expected int"):
        gauss_legendre(order)  # type: ignore[arg-type]
