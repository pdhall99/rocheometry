"""Dependency-free scalar root finding."""

from collections.abc import Callable
from math import isfinite, nextafter
from sys import float_info

from rocheometry._validation import _normalize_finite_float, require_finite, require_int

# Binary64 position tolerances for scalar roots.

# Resolution budget: default root absolute tolerance.
DEFAULT_ROOT_ABSOLUTE_TOLERANCE = 1e-14
"""The absolute scalar-root position tolerance."""
# Resolution budget: default root relative tolerance.
DEFAULT_ROOT_RELATIVE_TOLERANCE = 16.0 * float_info.epsilon
"""The scalar-root relative tolerance with a unit floor in its scale."""
# Work guard or cache capacity: root max iterations.
ROOT_MAX_ITERATIONS = 256
"""The maximum scalar-root solver iteration count."""


def _adjacent_float_root(
    lower: float, upper: float, f_lower: float, f_upper: float, tolerance: float
) -> float | None:
    """Choose the closer adjacent endpoint within tolerance or the measured step.

    Residuals must use the same units as tolerance. None means the bracket is
    not adjacent or neither endpoint meets the measured representation floor.
    Callers retain responsibility for physical bounds and failure messages.
    """
    if not lower < upper or nextafter(lower, upper) != upper:
        return None
    chosen, error = (
        (lower, abs(f_lower)) if abs(f_lower) <= abs(f_upper) else (upper, abs(f_upper))
    )
    return chosen if error <= max(tolerance, abs(f_upper - f_lower)) else None


def bisect_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float = DEFAULT_ROOT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ROOT_RELATIVE_TOLERANCE,
    max_iterations: int = ROOT_MAX_ITERATIONS,
) -> float:
    """Find a bracketed scalar root by bisection.

    Return an exact endpoint or midpoint root immediately. Otherwise accept
    when the bracket half-width is no larger than
    `max(absolute_tolerance, relative_tolerance * max(1, abs(left), abs(right)))`.
    The relative scale therefore has a unit floor; rescale small roots when
    relative accuracy is needed below that floor.

    Args:
        function: Continuous scalar residual. Endpoint values must have
            opposite signs or include an exact zero; evaluations must be finite.
        left: Finite lower bracket bound, strictly below `right`.
        right: Finite upper bracket bound.
        absolute_tolerance: Finite non-negative absolute position tolerance.
        relative_tolerance: Finite non-negative scaled position tolerance.
            At least one position tolerance must be positive.
        max_iterations: Positive integer limit on midpoint evaluations.

    Returns:
        A root estimate inside the original bracket.

    Raises:
        RuntimeError: The iteration limit is reached before convergence.
    """
    left, right, absolute_tolerance, relative_tolerance, max_iterations = (
        _root_controls(
            left, right, absolute_tolerance, relative_tolerance, max_iterations
        )
    )

    f_left = _require_finite_result(function(left), left)
    f_right = _require_finite_result(function(right), right)
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if same_sign(f_left, f_right):
        raise ValueError("root is not bracketed")

    for _ in range(max_iterations):
        midpoint = 0.5 * left + 0.5 * right
        f_midpoint = _require_finite_result(function(midpoint), midpoint)
        if f_midpoint == 0.0:
            return midpoint

        tolerance = max(
            absolute_tolerance,
            relative_tolerance * max(1.0, abs(left), abs(right)),
        )
        if 0.5 * right - 0.5 * left <= tolerance:
            return midpoint

        if same_sign(f_left, f_midpoint):
            left = midpoint
            f_left = f_midpoint
        else:
            right = midpoint

    raise RuntimeError("bisection did not converge")


def same_sign(left: float, right: float) -> bool:
    """Return whether two nonzero values have the same sign."""
    return (left < 0.0 and right < 0.0) or (left > 0.0 and right > 0.0)


def _require_finite_result(value: float, x: float) -> float:
    if isfinite(value):
        return value
    # Format the diagnostic only on failure: root evaluations are a hot path.
    return require_finite(value, name=f"function result at x={x!r}")


def _root_controls(
    left: float,
    right: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[float, float, float, float, int]:
    if type(left) is not float or not isfinite(left):
        left = _normalize_finite_float(left, name="left")
    if type(right) is not float or not isfinite(right):
        right = _normalize_finite_float(right, name="right")
    if not left < right:
        raise ValueError("root bracket: expected left < right")
    absolute_tolerance = _root_float(absolute_tolerance, name="absolute_tolerance")
    relative_tolerance = _root_float(relative_tolerance, name="relative_tolerance")
    if absolute_tolerance < 0.0:
        raise ValueError(
            f"absolute_tolerance: expected a non-negative number, got {absolute_tolerance!r}"
        )
    if relative_tolerance < 0.0:
        raise ValueError(
            f"relative_tolerance: expected a non-negative number, got {relative_tolerance!r}"
        )
    if absolute_tolerance == relative_tolerance == 0.0:
        raise ValueError("at least one root tolerance must be positive")
    if type(max_iterations) is not int:
        max_iterations = require_int(max_iterations, name="max_iterations")
    if max_iterations < 1:
        raise ValueError("max_iterations: expected a positive integer")

    return left, right, absolute_tolerance, relative_tolerance, max_iterations


def brent_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float = DEFAULT_ROOT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ROOT_RELATIVE_TOLERANCE,
    max_iterations: int = ROOT_MAX_ITERATIONS,
    residual_tolerance: float = 0.0,
) -> float:
    """Find a bracketed root using position/residual tolerance acceptance."""
    return _brent_core(
        function,
        left,
        right,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
        adjacent=False,
    )


def brent_root_adjacent(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float = DEFAULT_ROOT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ROOT_RELATIVE_TOLERANCE,
    max_iterations: int = ROOT_MAX_ITERATIONS,
) -> float:
    """Stop at a residual zero or the closer adjacent-float bracket endpoint.

    Position tolerances validate inputs only. The interpolation arithmetic and
    iteration work guard are shared with the tolerance-stopping entry point.
    """
    return _brent_core(
        function,
        left,
        right,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        residual_tolerance=0.0,
        adjacent=True,
    )


def _brent_core(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float = DEFAULT_ROOT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ROOT_RELATIVE_TOLERANCE,
    max_iterations: int = ROOT_MAX_ITERATIONS,
    residual_tolerance: float = 0.0,
    adjacent: bool = False,
) -> float:
    """Find a bracketed root with safeguarded Brent interpolation.

    Interpolation falls back to bisection when progress is unsafe. Position
    acceptance uses the same unit-floor tolerance as `bisect_root`; an exact
    zero or an evaluated residual within `residual_tolerance` is also accepted.
    Each iteration evaluates at most one new point in addition to the two
    initial endpoint evaluations.

    Args:
        function: Continuous scalar residual. Endpoint values must have
            opposite signs or include an exact zero; evaluations must be finite.
        left: Finite lower bracket bound, strictly below `right`.
        right: Finite upper bracket bound.
        absolute_tolerance: Finite non-negative absolute position tolerance.
        relative_tolerance: Finite non-negative scaled position tolerance.
            At least one position tolerance must be positive.
        max_iterations: Positive integer limit on solver iterations.
        residual_tolerance: Finite non-negative absolute residual tolerance;
            zero requires an exact residual zero for residual-only acceptance.
        adjacent: Internal node mode: positional tolerances validate
            inputs but cannot terminate the solve. Accept an exact residual
            zero or the closer endpoint of an adjacent-float sign bracket.
            This encloses the evaluated residual's zero, not necessarily an
            exact-field root when residual arithmetic loses precision.

    Returns:
        A root estimate inside the original bracket.

    Raises:
        RuntimeError: The iteration limit is reached before convergence.
    """
    left, right, absolute_tolerance, relative_tolerance, max_iterations = (
        _root_controls(
            left, right, absolute_tolerance, relative_tolerance, max_iterations
        )
    )
    residual_tolerance = _root_float(residual_tolerance, name="residual_tolerance")
    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance: expected a non-negative number")
    a, b = left, right
    fa = _require_finite_result(function(a), a)
    fb = _require_finite_result(function(b), b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if same_sign(fa, fb):
        raise ValueError("root is not bracketed")
    if abs(fa) <= residual_tolerance:
        return a
    if abs(fb) <= residual_tolerance:
        return b
    c, fc = a, fa
    step = previous_step = b - a
    magnitude_a, magnitude_b, magnitude_c = abs(fa), abs(fb), abs(fc)
    for _ in range(max_iterations):
        if (fa < 0.0) != (fb < 0.0):
            c, fc, magnitude_c = a, fa, magnitude_a
            step = previous_step = b - a
        if magnitude_c < magnitude_b:
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
            magnitude_a, magnitude_b, magnitude_c = (
                magnitude_b,
                magnitude_c,
                magnitude_b,
            )
        tolerance = absolute_tolerance
        if relative_tolerance:
            scale = max(1.0, abs(b), abs(c))
            tolerance = max(tolerance, relative_tolerance * scale)
        half_width = 0.5 * c - 0.5 * b
        width = abs(half_width)
        if adjacent:
            # Node radii have no positional-tolerance acceptance. Preserve
            # the interpolation rules with the smallest positive float and
            # stop at the closest representable bracket endpoint instead.
            tolerance = nextafter(0.0, 1.0)
            if nextafter(b, c) == c:
                return b if magnitude_b <= magnitude_c else c
        if magnitude_b <= residual_tolerance or width <= tolerance:
            return b

        interpolation = half_width
        previous_size = abs(previous_step)
        if previous_size > tolerance and magnitude_b < magnitude_a:
            if a == c:
                interpolation = -(fb / (fb - fa)) * (b - a)
            else:
                # Divided differences give the inverse quadratic step without
                # forming products of three residuals. Degenerate arithmetic
                # (including extreme scales) falls back to a finite midpoint.
                slope_a = (fa - fb) / (a - b)
                slope_c = (fc - fb) / (c - b)
                denominator = slope_a * slope_c * (fc - fa)
                if denominator != 0.0:
                    interpolation = -fb * (fc * slope_c - fa * slope_a) / denominator
            size = 2.0 * abs(interpolation)
            if (
                isfinite(interpolation)
                and size < previous_size
                and size < 3.0 * width - tolerance
            ):
                previous_step, step = step, interpolation
            else:
                previous_step = step = half_width
        else:
            previous_step = step = half_width

        a, fa, magnitude_a = b, fb, magnitude_b
        if abs(step) > tolerance:
            candidate = b + step
        else:
            candidate = b + (tolerance if half_width > 0.0 else -tolerance)
        if not isfinite(candidate):
            candidate = 0.5 * b + 0.5 * c
        if adjacent and candidate == b:
            candidate = nextafter(b, c)
        b = candidate
        fb = function(b)
        if not isfinite(fb):
            _require_finite_result(fb, b)
        magnitude_b = abs(fb)
        if magnitude_b <= residual_tolerance:
            return b

    raise RuntimeError("Brent root solve did not converge")


def _root_float(value: float, *, name: str) -> float:
    """Validate common float controls without numeric-ABC dispatch."""
    if type(value) is float and isfinite(value):
        return value
    return _normalize_finite_float(value, name=name)
