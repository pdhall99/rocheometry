"""Dependency-free quadrature rules."""

import reprlib
from functools import lru_cache
from math import cos, pi
from sys import float_info

from rocheometry._validation import require_int


@lru_cache(maxsize=64, typed=True)
def gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return nodes and weights on [-1, 1], retaining 64 recent orders.

    Args:
        order: Number of quadrature nodes; must be an positive integer.

    Returns:
        A `(nodes, weights)` pair of ordered immutable tuples. The rule
        integrates polynomials of degree up to `2*order - 1` on `[-1, 1]`.

    Raises:
        RuntimeError: A quadrature node fails to converge.
    """
    order = require_int(order, name="order")
    if order < 1:
        raise ValueError(
            f"order: expected a positive integer, got {reprlib.repr(order)}"
        )

    nodes = [0.0] * order
    weights = [0.0] * order
    midpoint_count = (order + 1) // 2
    tolerance = 8.0 * float_info.epsilon

    for index in range(midpoint_count):
        root = cos(pi * (index + 0.75) / (order + 0.5))
        derivative = 0.0
        for _ in range(64):
            p1 = 1.0
            p2 = 0.0
            for degree in range(1, order + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * degree - 1) * root * p2 - (degree - 1) * p3) / degree
            derivative = order * (root * p1 - p2) / (root * root - 1.0)
            updated = root - p1 / derivative
            if abs(updated - root) <= tolerance:
                root = updated
                break
            root = updated
        else:
            raise RuntimeError("Gauss-Legendre node solve did not converge")

        p1 = 1.0
        p2 = 0.0
        for degree in range(1, order + 1):
            p3 = p2
            p2 = p1
            p1 = ((2 * degree - 1) * root * p2 - (degree - 1) * p3) / degree
        derivative = order * (root * p1 - p2) / (root * root - 1.0)
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        nodes[index] = -root
        nodes[order - 1 - index] = root
        weights[index] = weight
        weights[order - 1 - index] = weight

    return tuple(nodes), tuple(weights)
