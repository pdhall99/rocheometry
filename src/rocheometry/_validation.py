"""Internal validation, requirement, and normalization helpers.

Conventions:
    - require_* checks one value and returns the identical object.
    - validate_* checks relationships or numeric ABCs; returns None.
    - normalize_* canonicalizes a value within its type.
    - parse_* converts text; coerce_* accepts multiple representations.
    - is_*/has_*/can_* are predicates; ensure_* establishes external state.
    - Generic helpers require `name`; domain helpers default it.
    - Check types before values: TypeError for invalid types,
      ValueError for invalid values, with consistent messages.
    - Reject None, inappropriate subclasses, and non-finite numbers.
    - Validate at public boundaries; use canonical values internally.
"""

from __future__ import annotations

import reprlib
from collections.abc import Collection, Iterable, Mapping
from math import isfinite, pi
from numbers import Real
from typing import TypeAlias, TypeVar, cast, get_args

from rocheometry._types import BinaryFrame, ComponentId, Frame

T = TypeVar("T")

TypeSpec: TypeAlias = type[T] | tuple[type[T], ...]


def require_instance(
    value: object,
    expected: TypeSpec[T],
    *,
    name: str,
) -> T:
    """Return the original value after checking its runtime type.

    Args:
        value: Object to check without conversion.
        expected: Accepted runtime class or tuple of classes; subclasses are accepted.
        name: Caller-facing field name used in error messages.
    """
    types = expected if isinstance(expected, tuple) else (expected,)
    if not isinstance(value, expected):
        expected_names = " or ".join(type_.__name__ for type_ in types)
        raise TypeError(
            f"{name}: expected {expected_names}, got {type(value).__name__}"
        )
    return value


def require_bool(value: object, *, name: str) -> bool:
    """Return a boolean without coercing truthy values."""
    return require_instance(value, bool, name=name)


def require_int(value: object, *, name: str) -> int:
    """Return an integer, rejecting booleans and numeric coercion."""
    if isinstance(value, bool):
        raise TypeError(f"{name}: expected int, got bool")
    return require_instance(value, int, name=name)


def validate_real(value: object, *, name: str) -> None:
    """Check for a real numeric value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name}: expected a real number, got {type(value).__name__}")


def require_finite(value: float, *, name: str) -> float:
    """Return the original numeric value after checking finiteness."""
    if not isfinite(value):
        raise ValueError(f"{name}: expected a finite number, got {reprlib.repr(value)}")
    return value


def require_positive(value: float, *, name: str) -> float:
    """Return the original numeric value after checking positivity."""
    if not value > 0.0:
        raise ValueError(
            f"{name}: expected a positive number, got {reprlib.repr(value)}"
        )
    return value


def require_choice(value: object, choices: Collection[T], *, name: str) -> T:
    """Return the identical value after requiring membership in a closed set.

    Args:
        value: Object to check without conversion or a separate type check.
        choices: Collection of accepted values, compared by equality.
        name: Caller-facing field name used in error messages.
    """
    if value not in choices:
        expected = " or ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name}: expected {expected}, got {reprlib.repr(value)}")
    return cast(T, value)


def _normalize_finite_float(value: float, *, name: str) -> float:
    if type(value) is float and isfinite(value):
        return value

    validate_real(value, name=name)
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{name}: expected a finite number, got {reprlib.repr(value)}"
        ) from exc
    return require_finite(normalized, name=name)


def _normalize_finite_positive_float(value: float, *, name: str) -> float:
    normalized = _normalize_finite_float(value, name=name)
    return require_positive(normalized, name=name)


def normalize_mass_ratio(q: float, *, name: str = "q") -> float:
    """Return a canonical finite positive system mass ratio."""
    return _normalize_finite_positive_float(q, name=name)


def normalize_component_mass_ratio(qi: float, *, name: str = "qi") -> float:
    """Return a canonical finite positive component mass ratio."""
    return _normalize_finite_positive_float(qi, name=name)


def require_component_id(
    component_id: object,
    *,
    name: str = "component_id",
) -> ComponentId:
    """Return a validated component ID, 1 or 2."""
    value = require_int(component_id, name=name)
    return cast(ComponentId, require_choice(value, get_args(ComponentId), name=name))


def normalize_volume_radius(
    volume_radius: float,
    *,
    name: str = "volume_radius",
) -> float:
    """Return a canonical finite positive volume-equivalent radius."""
    return _normalize_finite_positive_float(volume_radius, name=name)


def normalize_volume_radius_fill(
    volume_radius_fill: float,
    *,
    name: str = "volume_radius_fill",
) -> float:
    """Return a canonical volume-radius filling factor in (0, 1]."""
    normalized = _normalize_finite_positive_float(volume_radius_fill, name=name)
    if not normalized <= 1.0:
        raise ValueError(
            f"{name}: expected a value in (0, 1], got {reprlib.repr(normalized)}"
        )
    return normalized


def normalize_mass(mass: float, *, name: str) -> float:
    """Return a canonical finite positive component mass."""
    return _normalize_finite_positive_float(mass, name=name)


def normalize_separation(
    separation: float,
    *,
    name: str = "separation",
) -> float:
    """Return a canonical finite positive binary separation."""
    return _normalize_finite_positive_float(separation, name=name)


def require_binary_frame(
    frame: object,
    *,
    name: str = "frame",
) -> BinaryFrame:
    """Return the identical validated binary-geometry frame."""
    value = require_instance(frame, str, name=name)
    return cast(BinaryFrame, require_choice(value, get_args(BinaryFrame), name=name))


def require_frame(frame: object, *, name: str = "frame") -> Frame:
    """Return the identical validated coordinate frame."""
    value = require_instance(frame, str, name=name)
    return cast(Frame, require_choice(value, get_args(Frame), name=name))


def require_quadrature_order(
    order: int,
    *,
    name: str = "order",
) -> int:
    """Return the identical validated angular quadrature order."""
    value = require_int(order, name=name)
    if value < 4:
        raise ValueError(
            f"{name}: expected an integer greater than or equal to 4, "
            f"got {reprlib.repr(value)}"
        )
    return value


def normalize_volume_radius_fill_tolerance(
    volume_radius_fill_tolerance: float,
    *,
    name: str = "volume_radius_fill_tolerance",
) -> float:
    """Return a canonical finite positive volume-radius-fill tolerance."""
    return _normalize_finite_positive_float(volume_radius_fill_tolerance, name=name)


def require_num_upper_points(
    num_upper_points: int,
    *,
    name: str = "num_upper_points",
) -> int:
    """Return the identical validated upper-section point count."""
    value = require_int(num_upper_points, name=name)
    if value < 2:
        raise ValueError(
            f"{name}: expected an integer greater than or equal to 2, "
            f"got {reprlib.repr(value)}"
        )
    return value


def normalize_inclination(inclination: float, *, name: str = "inclination") -> float:
    """Validate finite radians in `[0, pi]`, then fold into `[0, pi/2]`."""
    value = _normalize_finite_float(inclination, name=name)
    if not 0.0 <= value <= pi:
        raise ValueError(
            f"{name}: expected radians in [0, pi], got {reprlib.repr(inclination)}; "
            "use math.radians() for degrees"
        )
    value = abs(value) % pi
    return min(value, pi - value)


def require_precision(
    precision: int,
    *,
    name: str = "precision",
) -> int:
    """Return the identical validated decimal formatting precision."""
    value = require_int(precision, name=name)
    if not 1 <= value <= 17:
        raise ValueError(
            f"{name}: expected an integer in [1, 17], got {reprlib.repr(value)}"
        )
    return value


def normalize_float_tuple(
    value: tuple[float, ...],
    *,
    name: str,
) -> tuple[float, ...]:
    """Return a canonical tuple of finite floats."""
    require_instance(value, tuple, name=name)
    return tuple(
        _normalize_finite_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def normalize_point2d(
    point: tuple[float, float],
    *,
    name: str = "point",
) -> tuple[float, float]:
    """Return one canonical finite two-dimensional point."""
    require_instance(point, tuple, name=name)
    if len(point) != 2:
        raise ValueError(
            f"{name}: expected a pair of finite numbers, got {reprlib.repr(point)}"
        )
    x = _normalize_finite_float(point[0], name=f"{name}[0]")
    y = _normalize_finite_float(point[1], name=f"{name}[1]")
    return (x, y)


def normalize_point3d(
    point: tuple[float, float, float],
    *,
    name: str = "point",
) -> tuple[float, float, float]:
    """Return one canonical finite three-dimensional point."""
    require_instance(point, tuple, name=name)
    if len(point) != 3:
        raise ValueError(
            f"{name}: expected three Cartesian coordinates, got {reprlib.repr(point)}"
        )
    x = _normalize_finite_float(point[0], name=f"{name}[0]")
    y = _normalize_finite_float(point[1], name=f"{name}[1]")
    z = _normalize_finite_float(point[2], name=f"{name}[2]")
    return (x, y, z)


def normalize_query_point(
    point: object,
    *,
    name: str = "point",
) -> tuple[float, float, float]:
    """Return a queried point as canonical floats.

    Queries accept any sequence or array of three numbers, such as a list or a
    NumPy array, where stored values require tuples.
    """
    if not isinstance(point, tuple):
        if isinstance(point, (str, bytes)):
            raise TypeError(
                f"{name}: expected three numbers, got {type(point).__name__}"
            )
        try:
            point = tuple(cast(Iterable[object], point))
        except TypeError:
            raise TypeError(
                f"{name}: expected three numbers, got {type(point).__name__}"
            ) from None
    return normalize_point3d(cast(tuple[float, float, float], point), name=name)


def normalize_margin(
    margin: float,
    *,
    name: str = "margin",
) -> float:
    """Return a canonical finite non-negative margin."""
    value = _normalize_finite_float(margin, name=name)
    if not value >= 0.0:
        raise ValueError(
            f"{name}: expected a non-negative number, got {reprlib.repr(value)}"
        )
    return value


def validate_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    """Validate the exact key set of one mapping."""
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"{name}: missing keys {reprlib.repr(sorted(missing))}")
    if extra:
        raise ValueError(f"{name}: unknown keys {reprlib.repr(sorted(extra))}")


def normalize_period(period: float, *, name: str = "period") -> float:
    """Return a canonical finite positive orbital period."""
    return _normalize_finite_positive_float(period, name=name)


def require_exactly_one(name: str, **selectors: object) -> None:
    """Require one non-None selector before validation or numerical work."""
    if sum(value is not None for value in selectors.values()) != 1:
        names = list(selectors)
        choices = ", ".join(names[:-1]) + " or " + names[-1]
        raise TypeError(f"{name}: pass exactly one of {choices}")
