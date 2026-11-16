from collections.abc import Callable
from math import inf, nextafter, sqrt
from sys import float_info
from typing import Protocol, cast

import pytest

from rocheometry._numerics import bisect_root, brent_root, brent_root_adjacent
from rocheometry._numerics.roots import _adjacent_float_root, same_sign


def test_adjacent_float_root_requires_adjacency_and_a_measured_floor() -> None:
    lower = 1.5
    upper = nextafter(lower, inf)
    assert _adjacent_float_root(lower, upper, -3e-12, 2e-12, 1e-13) == upper
    assert _adjacent_float_root(lower, upper, -1e-12, 2e-12, 1e-13) == lower
    assert _adjacent_float_root(lower, upper, 3e-12, 2e-12, 1e-13) is None
    assert _adjacent_float_root(lower, upper, 3e-14, 2e-14, 1e-13) == upper
    assert (
        _adjacent_float_root(lower, nextafter(upper, inf), -1e-12, 2e-12, 1e-13) is None
    )
    assert _adjacent_float_root(lower, lower, 0.0, 0.0, 1e-13) is None


@pytest.mark.parametrize("tolerance", (1e-3, 1e-14, 2e-14))
def test_node_brent_stops_at_adjacent_floats_independent_of_position_tolerance(
    tolerance: float,
) -> None:
    lower, upper = 1.0, 2.0

    def residual(x: float) -> float:
        nonlocal lower, upper
        value = x * x - 2.0
        if value < 0:
            lower = max(lower, x)
        else:
            upper = min(upper, x)
        return value

    root = brent_root_adjacent(
        residual,
        lower,
        upper,
        absolute_tolerance=tolerance,
        relative_tolerance=0.0,
    )
    assert nextafter(lower, upper) == upper
    assert root in (lower, upper)
    assert abs(root * root - 2) == min(abs(x * x - 2) for x in (lower, upper))


def test_node_brent_retains_the_iteration_work_guard() -> None:
    calls = 0

    def residual(x: float) -> float:
        nonlocal calls
        calls += 1
        return x * x - 2

    with pytest.raises(RuntimeError, match="Brent root solve did not converge"):
        brent_root_adjacent(residual, 1.0, 2.0, max_iterations=3)
    assert calls == 5


class _RootSolver(Protocol):
    def __call__(
        self,
        function: Callable[[float], float],
        left: float,
        right: float,
        *,
        absolute_tolerance: float = ...,
        relative_tolerance: float = ...,
        max_iterations: int = ...,
    ) -> float: ...


@pytest.fixture(params=[bisect_root, brent_root], ids=["bisection", "brent"])
def root(request: pytest.FixtureRequest) -> _RootSolver:
    return cast(_RootSolver, request.param)


def test_bracketed_root_finds_bracketed_scalar_root(root: _RootSolver) -> None:
    actual = root(lambda x: x * x - 2.0, 0.0, 2.0)

    assert actual == pytest.approx(sqrt(2.0), rel=0.0, abs=1.0e-14)


@pytest.mark.parametrize("scale", [1.0, 1.0e6])
@pytest.mark.parametrize("relative", [False, True])
def test_bracketed_root_tolerances_control_accuracy_and_work(
    root: _RootSolver, scale: float, relative: bool
) -> None:
    calls = 0

    def function(x: float) -> float:
        nonlocal calls
        calls += 1
        return (x / scale) ** 2 - 2.0

    def solve(tolerance: float) -> float:
        return root(
            function,
            scale,
            2.0 * scale,
            absolute_tolerance=0.0 if relative else tolerance * scale,
            relative_tolerance=tolerance if relative else 0.0,
        )

    coarse = solve(1.0e-3)
    coarse_calls = calls
    calls = 0
    fine = solve(1.0e-10)
    expected = sqrt(2.0) * scale

    assert abs(coarse - expected) <= 2.0e-3 * scale
    assert abs(fine - expected) <= 2.0e-10 * scale
    assert abs(fine - expected) < abs(coarse - expected)
    assert calls > coarse_calls


def test_bracketed_root_iteration_limit_bounds_midpoint_evaluations(
    root: _RootSolver,
) -> None:
    calls = 0

    def function(x: float) -> float:
        nonlocal calls
        calls += 1
        return x * x - 2.0

    with pytest.raises(RuntimeError, match="did not converge"):
        root(function, 0.0, 2.0, max_iterations=3)
    assert calls == 5  # Two endpoints and three midpoints.


@pytest.mark.parametrize("bounds", [(3.0, 8.0), (-2.0, 3.0), (2.0, 4.0)])
def test_bracketed_root_returns_exact_roots_with_one_iteration(
    root: _RootSolver,
    bounds: tuple[float, float],
) -> None:
    assert root(lambda x: x - 3.0, *bounds, max_iterations=1) == 3.0


@pytest.mark.parametrize("name", ["absolute_tolerance", "relative_tolerance"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_bracketed_root_rejects_invalid_tolerances(
    root: _RootSolver, name: str, value: float
) -> None:
    controls = {name: value}
    with pytest.raises(ValueError, match=name):
        root(lambda x: x, -1.0, 1.0, **controls)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["absolute_tolerance", "relative_tolerance"])
@pytest.mark.parametrize("value", [True, "1e-6"])
def test_bracketed_root_rejects_wrong_tolerance_types(
    root: _RootSolver, name: str, value: object
) -> None:
    controls = {name: value}
    with pytest.raises(TypeError, match=name):
        root(lambda x: x, -1.0, 1.0, **controls)  # type: ignore[arg-type]


def test_bracketed_root_rejects_zero_tolerances(root: _RootSolver) -> None:
    with pytest.raises(
        ValueError, match="at least one root tolerance must be positive"
    ):
        root(lambda x: x, -1.0, 1.0, absolute_tolerance=0.0, relative_tolerance=0.0)


@pytest.mark.parametrize("value", [0, -1])
def test_bracketed_root_rejects_nonpositive_iteration_limits(
    root: _RootSolver, value: int
) -> None:
    with pytest.raises(ValueError, match="max_iterations: expected a positive integer"):
        root(lambda x: x, -1.0, 1.0, max_iterations=value)


@pytest.mark.parametrize("value", [True, 2.5])
def test_bracketed_root_rejects_wrong_iteration_limit_types(
    root: _RootSolver, value: object
) -> None:
    with pytest.raises(TypeError, match="max_iterations: expected int"):
        root(lambda x: x, -1.0, 1.0, max_iterations=value)  # type: ignore[arg-type]


def test_bracketed_root_rejects_unbracketed_root(root: _RootSolver) -> None:
    with pytest.raises(ValueError, match="not bracketed"):
        root(lambda x: x * x + 1.0, -1.0, 1.0)


@pytest.mark.parametrize("bounds", [(1.0, 0.0), (1.0, 1.0)])
def test_bracketed_root_rejects_unordered_bounds_before_evaluating(
    root: _RootSolver,
    bounds: tuple[float, float],
) -> None:
    def unexpected(x: float) -> float:
        raise AssertionError("invalid bounds must be rejected before evaluation")

    with pytest.raises(ValueError, match="expected left < right"):
        root(unexpected, *bounds)


@pytest.mark.parametrize("name", ["left", "right"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_bracketed_root_rejects_nonfinite_bounds(
    root: _RootSolver, name: str, value: float
) -> None:
    left = value if name == "left" else 0.0
    right = value if name == "right" else 1.0
    with pytest.raises(ValueError, match=f"{name}: expected a finite number"):
        root(lambda x: x - 0.3, left, right)


@pytest.mark.parametrize("at", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_bracketed_root_rejects_nonfinite_endpoint_and_midpoint_results(
    root: _RootSolver, at: float, value: float
) -> None:
    def function(x: float) -> float:
        return value if x == at or (at == 0.5 and 0.0 < x < 1.0) else x - 0.3

    with pytest.raises(
        ValueError, match="function result at x=.*expected a finite number"
    ):
        root(function, 0.0, 1.0)


@pytest.mark.parametrize("left", [1.0e308, -float_info.max])
def test_bracketed_root_keeps_midpoints_finite_for_large_bounds(
    root: _RootSolver, left: float
) -> None:
    actual = root(lambda x: x / float_info.max - 0.75, left, float_info.max)
    assert actual == pytest.approx(0.75 * float_info.max, rel=1.0e-14)


def test_same_sign_handles_nonzero_values() -> None:
    assert same_sign(-2.0, -1.0)
    assert same_sign(1.0, 2.0)
    assert not same_sign(-1.0, 1.0)
    assert not same_sign(0.0, 1.0)


@pytest.mark.parametrize("power", [1, 3, 7])
def test_brent_handles_flat_and_asymmetric_roots(power: int) -> None:
    actual = brent_root(lambda x: (x - 0.125) ** power, -1.0, 2.0)
    assert actual == pytest.approx(0.125, abs=2.0e-14)


def test_brent_uses_fewer_evaluations_than_bisection() -> None:
    counts = []
    for solver in (bisect_root, brent_root):
        calls = 0

        def function(x: float) -> float:
            nonlocal calls
            calls += 1
            return x * x - 2.0

        actual = solver(function, 0.0, 2.0)
        assert actual == pytest.approx(sqrt(2.0), abs=1.0e-14)
        counts.append(calls)
    assert counts[1] <= 10
    assert counts[1] < counts[0] / 3


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_brent_validates_residual_tolerance(value: float) -> None:
    error = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(error, match="residual_tolerance"):
        brent_root(lambda x: x, -1.0, 1.0, residual_tolerance=value)


def test_brent_residual_stopping_uses_an_evaluated_point() -> None:
    calls: list[float] = []

    def function(x: float) -> float:
        calls.append(x)
        return x * x - 2.0

    result = brent_root(function, 0.0, 2.0, residual_tolerance=1.0e-6)
    assert result in calls
    assert abs(function(result)) <= 1.0e-6
