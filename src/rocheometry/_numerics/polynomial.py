"""Dependency-free power and Chebyshev polynomial evaluation."""

CoefficientMatrix = tuple[tuple[float, ...], ...]


def power_value(x: float, coefficients: tuple[float, ...]) -> float:
    """Evaluate a power series in ascending coefficient order using Horner's rule.

    Args:
        x: Evaluation coordinate.
        coefficients: Nonempty power coefficients in ascending degree order.
    """
    if not coefficients:
        raise ValueError("power coefficient series must not be empty")
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def chebyshev_value(x: float, coefficients: tuple[float, ...]) -> float:
    """Evaluate a Chebyshev series in ascending degree order.

    Args:
        x: Evaluation coordinate; Chebyshev coordinates usually lie in `[-1, 1]`.
        coefficients: Nonempty Chebyshev coefficients in ascending degree order.
    """
    if not coefficients:
        raise ValueError("Chebyshev coefficient series must not be empty")
    value = coefficients[0]
    if len(coefficients) == 1:
        return value

    previous = 1.0
    current = x
    value += coefficients[1] * current
    for coefficient in coefficients[2:]:
        following = 2.0 * x * current - previous
        value += coefficient * following
        previous, current = current, following
    return value


def power_value_2d(x: float, y: float, coefficients: CoefficientMatrix) -> float:
    """Evaluate a tensor power polynomial with rows ordered by x degree.

    Args:
        x: First evaluation coordinate.
        y: Second evaluation coordinate.
        coefficients: Nonempty power coefficient rows; row index is x degree
            and column index is y degree. Every row must be nonempty.
    """
    if not coefficients:
        raise ValueError("power coefficient tensor must not be empty")
    return power_value(x, tuple(power_value(y, row) for row in coefficients))


def chebyshev_value_2d(x: float, y: float, coefficients: CoefficientMatrix) -> float:
    """Evaluate a tensor Chebyshev polynomial with rows ordered by x degree.

    Args:
        x: First evaluation coordinate.
        y: Second evaluation coordinate.
        coefficients: Nonempty Chebyshev coefficient rows; row index is x degree
            and column index is y degree. Every row must be nonempty.
    """
    if not coefficients:
        raise ValueError("Chebyshev coefficient tensor must not be empty")
    return chebyshev_value(x, tuple(chebyshev_value(y, row) for row in coefficients))
