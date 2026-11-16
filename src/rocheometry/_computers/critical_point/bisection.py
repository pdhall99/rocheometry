"""Packaged binary-centreline critical-point computers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import nextafter, sqrt
from sys import float_info
from typing import Literal, get_args

from rocheometry._numerics import bisect_root
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint, Potential
from rocheometry._validation import (
    _normalize_finite_float,
    require_choice,
    require_finite,
    require_instance,
)

CriticalBranch = Literal["l1", "outer"]
# Insensitive control: critical point distance relative tolerance.
CRITICAL_POINT_DISTANCE_RELATIVE_TOLERANCE = 8.0 * float_info.epsilon
# Field evaluation squares distances; keep even those squares representable.
# Derived: critical point minimum distance.
CRITICAL_POINT_MINIMUM_DISTANCE = sqrt(nextafter(0.0, 1.0))
# Derived: critical point companion coordinate gap.
CRITICAL_POINT_COMPANION_COORDINATE_GAP = 1.0 - nextafter(1.0, 0.0)


@dataclass(frozen=True, slots=True)
class BisectionCriticalPointComputer:
    """A bracketed bisection strategy for component-local critical points.

    Use `l1()` or `outer()` for adaptive mass-ratio brackets. Alternatively,
    supply finite ordered `lower` and `upper` bounds for fixed-bracket
    experiments. Adaptive and fixed-bracket construction are mutually exclusive.

    Attributes:
        lower: Fixed lower x bound, or `None` for adaptive construction.
        upper: Fixed upper x bound, or `None` for adaptive construction.
        branch: Adaptive branch (`l1` or `outer`), or `None` for fixed bounds.
    """

    lower: float | None = None
    upper: float | None = None
    branch: CriticalBranch | None = field(default=None, kw_only=True)

    @classmethod
    def l1(cls) -> BisectionCriticalPointComputer:
        """Return an adaptive solver for L1, at 0 < x < 1."""
        return cls(branch="l1")

    @classmethod
    def outer(cls) -> BisectionCriticalPointComputer:
        """Return an adaptive solver for the outer stationary point, at x < 0."""
        return cls(branch="outer")

    def __post_init__(self) -> None:
        if self.branch is not None:
            require_choice(
                require_instance(self.branch, str, name="branch"),
                get_args(CriticalBranch),
                name="branch",
            )
            if self.lower is not None or self.upper is not None:
                raise ValueError("branch: adaptive solves do not accept fixed bounds")
            return
        if self.lower is None or self.upper is None:
            raise ValueError(
                "critical-point bracket: both lower and upper are required"
            )
        lower = _normalize_finite_float(self.lower, name="lower")
        upper = _normalize_finite_float(self.upper, name="upper")
        if not lower < upper:
            raise ValueError(
                "critical-point bracket: expected lower < upper, "
                f"got ({lower!r}, {upper!r})"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def __call__(self, field: RocheField, /) -> CriticalPoint:
        """Return the stationary point, Jacobi potential and local Hessian.

        Positions use separation units; Hessian entries use intrinsic
        G*(M1+M2)/a**3 units in component-local axes.

        Args:
            field: Normalized component-local field whose gradient defines the root.

        Raises:
            ValueError: The stationary point cannot be bracketed or represented as
                a distinct coordinate, or a gradient evaluation is non-finite.
            RuntimeError: The root solve reaches its iteration limit.
        """
        if self.branch is not None:
            x = _adaptive_position(field, self.branch)
        else:
            if self.lower is None or self.upper is None:  # pragma: no cover
                raise AssertionError("fixed critical-point solve is missing its bounds")

            def derivative_x(x: float) -> float:
                return require_finite(
                    field.gradient(x)[0], name="critical-point gradient"
                )

            x = bisect_root(derivative_x, self.lower, self.upper)
        return CriticalPoint(
            position=(x, 0.0, 0.0),
            potential=Potential.jacobi(
                float(field._axis_potential_exact(x))
                if type(field).potential is RocheField.potential
                else field.potential(x)
            ),
            hessian=field.hessian(x),
        )


def _adaptive_position(field: RocheField, branch: CriticalBranch) -> float:
    component_mass, companion_mass = field.component_mass, field.companion_mass
    reflected = branch == "l1" and field.qi > 1.0
    mass = companion_mass if reflected else component_mass
    scale = mass ** (1.0 / 3.0)
    minimum = (
        CRITICAL_POINT_COMPANION_COORDINATE_GAP
        if reflected
        else CRITICAL_POINT_MINIMUM_DISTANCE
    )
    maximum = nextafter(1.0, 0.0) if branch == "l1" else float_info.max
    lower = max(0.25 * scale, minimum)
    upper = min(max(scale, 2.0 * lower), 0.5) if branch == "l1" else scale

    def resolution_error() -> ValueError:
        return ValueError(
            f"{branch} critical-point solve: numerical resolution cannot bracket "
            f"a distinct stationary point for qi={field.qi!r}"
        )

    def position(distance: float) -> float:
        if branch == "outer":
            return -distance
        return 1.0 - distance if reflected else distance

    def residual(distance: float) -> float:
        x = position(distance)
        if x == 0.0 or (branch == "l1" and x == 1.0):
            raise resolution_error()
        derivative = require_finite(
            field.gradient(x)[0], name=f"{branch} critical-point gradient"
        )
        return -derivative if reflected or branch == "outer" else derivative

    lower_value = residual(lower)
    while lower_value > 0.0:
        upper = lower
        lower = max(0.5 * lower, minimum)
        if lower == upper:
            raise resolution_error()
        lower_value = residual(lower)
    if lower_value == 0.0:
        return position(lower)

    upper_value = residual(upper)
    while upper_value < 0.0:
        lower = upper
        upper = min(2.0 * upper, maximum)
        if lower == upper:
            raise resolution_error()
        upper_value = residual(upper)
    if upper_value == 0.0:
        return position(upper)

    # The final sign bracket keeps the root's scaled coordinate O(1), even
    # after contractions or expansions. This avoids a unit-sized error floor
    # for roots close to a component while respecting custom field gradients.
    scale = upper
    coordinate = bisect_root(
        lambda value: residual(scale * value),
        lower / scale,
        1.0,
        absolute_tolerance=0.0,
        relative_tolerance=CRITICAL_POINT_DISTANCE_RELATIVE_TOLERANCE,
    )
    return position(scale * coordinate)
