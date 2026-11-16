"""Checker-only first-order running bounds for the actual node arithmetic.

Each rounded operation contributes u*abs(result), u=2**-53, in addition to
first-order propagated input errors. Positive bound arithmetic rounds outward
with nextafter; this is arithmetic enclosure, not a fitted safety factor.
Values reproduce binary64 production operations; exact ray/point truth uses
100 digits. No production function or initializer is edited.
"""

from __future__ import annotations

import inspect
import math
import textwrap
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from types import CodeType, SimpleNamespace
from typing import Any, Self
from unittest.mock import patch

import rocheometry
from crosschecks.instrumented.node_backward import (
    NodeRecords,
    capture_nodes,
    copied,
)
from crosschecks.problem import Level, exact_values
from rocheometry._computers import neck, neck_ray
from rocheometry._computers.surface_radius import brent
from rocheometry._geometry import Point3D
from rocheometry._roche import field as field_module
from rocheometry._roche.field import RocheField

DD = tuple[float, float]

U = 2.0**-53


def add_up(a: float, b: float) -> float:
    if not a:
        return b
    if not b:
        return a
    return math.nextafter(a + b, math.inf)


def mul_up(a: float, b: float) -> float:
    if not a or not b:
        return 0.0
    return math.nextafter(a * b, math.inf)


def div_up(a: float, b: float) -> float:
    if not a:
        return 0.0
    return math.nextafter(a / b, math.inf)


def exact_error(value: Fraction) -> float:
    """Round a nonnegative rational displacement upward, including underflow."""
    value = abs(value)
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


class Bound(float):
    error: float

    def __new__(cls, value: float, error: float = 0.0) -> Self:
        result = float.__new__(cls, value)
        result.error = error
        return result

    @staticmethod
    def error_of(value: float) -> float:
        return value.error if isinstance(value, Bound) else 0.0

    def __add__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        value = float(self) + float(other)
        return Bound(
            value,
            add_up(add_up(self.error, self.error_of(other)), mul_up(U, abs(value))),
        )

    __radd__ = __add__

    def __sub__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        value = float(self) - float(other)
        return Bound(
            value,
            add_up(add_up(self.error, self.error_of(other)), mul_up(U, abs(value))),
        )

    def __rsub__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        return Bound(other, self.error_of(other)) - self

    def __mul__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        value = float(self) * float(other)
        propagated = add_up(
            mul_up(abs(float(other)), self.error),
            mul_up(abs(float(self)), self.error_of(other)),
        )
        return Bound(value, add_up(propagated, mul_up(U, abs(value))))

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        a, b = float(self), float(other)
        value = a / b
        propagated = div_up(
            add_up(self.error, mul_up(div_up(abs(a), abs(b)), self.error_of(other))),
            abs(b),
        )
        return Bound(value, add_up(propagated, mul_up(U, abs(value))))

    def __rtruediv__(self, other: float) -> Bound:  # type: ignore[missing-override-decorator]
        return Bound(other, self.error_of(other)) / self

    def __pow__(self, exponent: float) -> Bound:  # type: ignore[bad-override, missing-override-decorator]
        if self.error_of(exponent) or exponent != int(exponent):
            raise NotImplementedError(
                "Only exact integer powers occur in the residual/gradient"
            )
        power = int(exponent)
        value = float(self) ** power
        weight = abs(power) * abs(float(self)) ** (power - 1) if power else 0.0
        return Bound(value, add_up(mul_up(weight, self.error), mul_up(U, abs(value))))

    def __neg__(self) -> Bound:  # type: ignore[missing-override-decorator]
        return Bound(-float(self), self.error)

    def __abs__(self) -> Bound:  # type: ignore[missing-override-decorator]
        return Bound(abs(float(self)), self.error)


class Replay:
    """Faithful scalar values with first-order absolute error propagation.

    +/-: Ea+Eb+u|z|; multiply: |b|Ea+|a|Eb+u|z|;
    divide: (Ea+|a/b|Eb)/|b|+u|z|; sqrt: Ea/(2|z|)+u|z|;
    hypot: sum(|ai|Ei)/|z|+u|z|; fsum: sum(Ei)+u|z|.
    fsum forms an exact sum and rounds once. Negation and copying are exact.
    This is the requested first-order model; no second-order term or factor
    is introduced. Cached exact rational reports carry their one-round error.
    """

    def __init__(self) -> None:
        self.last_terms: tuple[float, ...] | None = None

    def sqrt(self, value: float) -> Bound:
        result = math.sqrt(value)
        if not result and Bound.error_of(value):
            raise ArithmeticError("First-order sqrt derivative is undefined at zero")
        propagated = div_up(Bound.error_of(value), 2 * result) if result else 0.0
        return Bound(result, add_up(propagated, mul_up(U, abs(result))))

    def hypot(self, *values: float) -> Bound:
        result = math.hypot(*values)
        if not result and any(Bound.error_of(v) for v in values):
            raise ArithmeticError("First-order hypot derivative is undefined at zero")
        propagated = 0.0
        for value in values:
            propagated = add_up(
                propagated, mul_up(abs(float(value)), Bound.error_of(value))
            )
        propagated = div_up(propagated, result) if result else 0.0
        return Bound(result, add_up(propagated, mul_up(U, abs(result))))

    def fsum(self, values: Iterable[float]) -> Bound:
        values = tuple(values)
        result = math.fsum(values)
        propagated = 0.0
        for value in values:
            propagated = add_up(propagated, Bound.error_of(value))
        self.last_terms = tuple(float(v) for v in values)
        return Bound(result, add_up(propagated, mul_up(U, abs(result))))


@contextmanager
def capture_running_nodes() -> Iterator[NodeRecords]:
    """Capture scale/conversion inputs and the final retained DD radius too."""
    with capture_nodes() as records:
        stack: list[dict[str, Any]] = []
        base_node = brent.BrentSurfaceRadiusComputer.node_radius
        base_root = brent._node_root
        base_ray = RocheField._ray_residual
        base_dd = neck_ray._ExactNeckRay.evaluate

        def ray(
            field: RocheField,
            direction: Point3D,
            depth: float,
            reference_x: float | None = None,
            *,
            inverse_radius: bool = False,
            l1_x: float,
        ) -> Callable[[float], float]:
            function = base_ray(
                field,
                direction,
                depth,
                reference_x,
                inverse_radius=inverse_radius,
                l1_x=l1_x,
            )
            if stack:
                stack[-1]["last_function"] = function
            return function

        def node(
            computer: brent.BrentSurfaceRadiusComputer,
            field: RocheField,
            depth: float,
            direction: Point3D,
        ) -> float:
            frame: dict[str, Any] = dict(direction=direction)
            stack.append(frame)
            try:
                result = base_node(computer, field, depth, direction)
                record = records.setdefault(direction, {})
                if "spec" not in record and "dd_terms" not in record:
                    function = frame["last_function"]
                    record.update(
                        spec=function._backward_spec,
                        argument=result,
                        bypass=True,
                        residual_value=function(result),
                    )
                return result
            finally:
                stack.pop()

        def root(
            function: Callable[[float], float],
            left: float,
            right: float,
            **kwargs: float,
        ) -> float:
            caller = inspect.currentframe()
            assert caller is not None and caller.f_back is not None
            scale = caller.f_back.f_locals["outer_radius"]
            coordinate = base_root(function, left, right, **kwargs)
            record = records[stack[-1]["direction"]]
            record.update(
                outer_radius=scale,
                root_coordinate=coordinate,
                residual_value=function(coordinate),
            )
            return coordinate

        def dd(
            model: neck_ray._ExactNeckRay, radius: DD, direction: Point3D, depth: float
        ) -> tuple[DD, tuple[DD, DD, DD]]:
            value, gradient = base_dd(model, radius, direction, depth)
            records.setdefault(direction, {}).update(
                dd_radius=tuple(radius),
                dd_depth=depth,
                dd_value=tuple(value),
                dd_gradient=tuple(tuple(v) for v in gradient),
            )
            return value, gradient

        with (
            patch.object(RocheField, "_ray_residual", ray),
            patch.object(brent.BrentSurfaceRadiusComputer, "node_radius", node),
            patch.object(brent, "_node_root", root),
            patch.object(neck_ray._ExactNeckRay, "evaluate", dd),
        ):
            yield records


def field_model(
    field: RocheField, replay: Replay
) -> tuple[SimpleNamespace, dict[str, Any]]:
    """Seed stored exact masses and exact-rational cached level conversions."""

    def level(l1: float, depth: float) -> Bound:
        value = field._level_value(float(l1), float(depth))
        return Bound(value, mul_up(U, abs(value)))

    def parts(l1: float, anchor: float) -> tuple[Bound, Bound]:
        high, low = field._axis_depth_parts(float(l1), float(anchor))
        # Fixed high + rounded exact rational remainder: the sum's only
        # approximation is the final low report. Do not invent a cache budget.
        return Bound(high), Bound(low, mul_up(U, abs(low)))

    namespace = dict(vars(field_module), fsum=replay.fsum, sqrt=replay.sqrt)
    model = SimpleNamespace(
        component_mass=Bound(field.component_mass),
        companion_mass=Bound(field.companion_mass),
        _mass_error=Bound(field._mass_error),
        _half_mass_error=Bound(field._half_mass_error),
        _potential_constant=Bound(
            field._potential_constant, mul_up(U, abs(field._potential_constant))
        ),
        _level_value=level,
        _axis_depth_parts=parts,
    )
    depth_offset = copied(RocheField._depth_offset, namespace)
    model._depth_offset = lambda depth, l1, anchor: depth_offset(
        model, depth, l1, anchor
    )
    return model, namespace


def residual_replay(
    field: RocheField, radius: float, direction: Point3D, record: dict[str, Any]
) -> Bound:
    """Replay coefficient binding and the residual at the solved coordinate.

    Detached coefficients include the actual outer_radius*d products. The
    returned-radius division contributes its first-order coordinate uncertainty
    through c=B/r (Ec=|c/r|Er). Contact replays the actual wrapper division.
    Cartesian products are separate exact-rounding checks, never part of E_ray.
    """
    replay = Replay()
    model, namespace = field_model(field, replay)
    spec = record["spec"]
    if record.get("bypass"):
        coefficients = tuple(Bound(v) for v in spec["direction"])
        argument = Bound(record["argument"])
    elif spec["inverse_radius"]:
        scale = Bound(record["outer_radius"])
        coordinate = Bound(record["root_coordinate"])
        report = scale / coordinate
        assert float(report) == radius
        # Transfer the one reported-radius rounding to the coordinate at which
        # the bound inverse residual is evaluated. This is derived propagation.
        coordinate_error = mul_up(
            div_up(abs(float(coordinate)), abs(radius)), report.error
        )
        argument = Bound(float(coordinate), coordinate_error)
        coefficients = tuple(scale * Bound(v) for v in direction)
    else:
        coefficients = tuple(Bound(v) for v in direction)
        argument = Bound(record["outer_radius"]) / Bound(record["root_coordinate"])
        assert float(argument) == radius
    assert tuple(float(v) for v in coefficients) == tuple(spec["direction"])
    function = copied(RocheField._ray_residual, namespace)(
        model,
        coefficients,
        Bound(spec["depth"]),
        None if spec["reference_x"] is None else Bound(spec["reference_x"]),
        inverse_radius=spec["inverse_radius"],
        l1_x=Bound(spec["l1_x"]),
    )
    result = function(argument)
    assert float(result) == record["residual_value"], (
        "Residual replay differs from production"
    )
    return result


@cache
def dd_code() -> CodeType:
    source = textwrap.dedent(inspect.getsource(neck_ray._ExactNeckRay.evaluate))
    old = "    displacement = _sub(x, self.anchor)"
    assert source.count(old) == 1
    source = source.replace(old, "    x, y, z = bridge((x, y, z))\n" + old)
    namespace = dict(vars(neck_ray))
    exec(compile(source, __file__ + ":dd-running", "exec"), namespace)
    return namespace["evaluate"].__code__


def dd_replay(
    field: RocheField,
    origin: float,
    radius: float,
    direction: Point3D,
    target: tuple[Fraction, ...],
    record: dict[str, Any],
) -> tuple[Replay, Bound, tuple[Bound, ...]]:
    """Replay every existing DD primitive, without assuming correlations.

    The captured DD radius is an exact pair of floats. Its represented point
    differs from the requested exact-ray or returned-float point; seed that
    exact rational displacement on the DD Cartesian inputs before propagation.
    This bridge changes bounds only. All reproduced high/low values must match.
    """
    from types import FunctionType

    replay = Replay()
    namespace: dict[str, Any] = dict(vars(neck_ray), sqrt=replay.sqrt, fsum=replay.fsum)
    for name, function in (
        ("_add", neck._add),
        ("_mul", neck._mul),
        ("_scale", neck._scale),
        ("_neg", neck_ray._neg),
        ("_sub", neck_ray._sub),
        ("_div", neck_ray._div),
        ("_sqrt", neck_ray._sqrt),
    ):
        namespace[name] = copied(function, namespace)
    retained = record["dd_radius"]
    exact_radius = Fraction(retained[0]) + Fraction(retained[1])
    displacement = [
        exact_error(Fraction(p) - exact_radius * Fraction(d))
        for p, d in zip(target, direction, strict=True)
    ]

    def bridge(
        coordinates: tuple[tuple[Bound, Bound], ...],
    ) -> tuple[tuple[Bound, Bound], ...]:
        return tuple(
            (Bound(float(pair[0]), add_up(pair[0].error, e)), pair[1])
            for pair, e in zip(coordinates, displacement, strict=True)
        )

    namespace["bridge"] = bridge
    original = neck_ray._exact_neck_ray(field, origin)
    model = SimpleNamespace(
        **{
            name: tuple(Bound(v) for v in getattr(original, name))
            for name in ("m", "c", "total", "anchor", "companion")
        }
    )
    function = FunctionType(dd_code(), namespace)
    value, gradient = function(
        model,
        tuple(Bound(v) for v in retained),
        tuple(Bound(v) for v in direction),
        Bound(record["dd_depth"]),
    )
    assert tuple(map(float, value)) == record["dd_value"], "DD residual replay differs"
    assert tuple(tuple(map(float, v)) for v in gradient) == record["dd_gradient"], (
        "DD gradient replay differs"
    )
    return replay, replay.fsum(value), tuple(replay.fsum(v) for v in gradient)


def gradient_replay(
    field: RocheField,
    point: Point3D,
    normal: Point3D,
    gravity: float,
    record: dict[str, Any],
    *,
    origin: float,
    radius: float,
    direction: Point3D,
) -> tuple[tuple[Bound, ...], Bound]:
    if "dd_radius" in record:
        replay, _, gradient = dd_replay(
            field, origin, radius, direction, tuple(Fraction(v) for v in point), record
        )
    else:
        replay = Replay()
        model, namespace = field_model(field, replay)
        gradient = copied(RocheField.gradient, namespace)(
            model, *(Bound(v) for v in point)
        )
    magnitude = replay.hypot(*gradient)
    normals = tuple(-v / magnitude for v in gradient)
    assert float(magnitude) == gravity, "Gravity replay differs from production"
    assert tuple(map(float, normals)) == tuple(normal), (
        "Normal replay differs from production"
    )
    represented = tuple(-magnitude * v for v in normals)
    return represented, magnitude


def check_node(
    surface: rocheometry.ComponentSurfaceGeometry,
    radius: float,
    direction: Point3D,
    point: Point3D,
    normal: Point3D,
    gravity: float,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Measure exact-ray level and float-position gradient against running E."""
    with localcontext() as context:
        context.prec = 100
        field = surface.component.field
        level = Level.of(surface)
        origin = level.l1_x
        ray = tuple(Fraction(radius) * Fraction(d) for d in direction)
        cartesian = tuple(Fraction(v) for v in point)
        coordinate_matches = [
            p.hex() == (float(v) if v else math.copysign(0.0, d)).hex()
            for p, v, d in zip(point, ray, direction, strict=True)
        ]
        value = level.value()
        target = Decimal(value.numerator) / Decimal(value.denominator)
        ray_value, _ = exact_values(level.binary, ray)
        point_value, point_gradient = exact_values(level.binary, cartesian)
        if "dd_radius" in record:
            _, residual, _ = dd_replay(field, origin, radius, direction, ray, record)
            branch = "double-double factored"
        else:
            residual = residual_replay(field, radius, direction, record)
            branch = (
                "factored" if record["spec"]["reference_x"] is not None else "absolute"
            )
        represented, magnitude = gradient_replay(
            field,
            point,
            normal,
            gravity,
            record,
            origin=origin,
            radius=radius,
            direction=direction,
        )
        errors = [
            abs(Decimal.from_float(float(a)) - b)
            for a, b in zip(represented, point_gradient, strict=True)
        ]
        bounds = [Decimal.from_float(v.error) for v in represented]
        exact_magnitude = sum((v * v for v in point_gradient), Decimal(0)).sqrt()
        magnitude_error = abs(Decimal.from_float(gravity) - exact_magnitude)
        level_error = ray_value - target
        bound = Decimal.from_float(residual.error)
        ratios = [
            float(error / b) if b else 0.0 if not error else math.inf
            for error, b in zip(errors, bounds, strict=True)
        ]
        level_ratio = (
            float(abs(level_error) / bound)
            if bound
            else 0.0
            if not level_error
            else math.inf
        )
        gravity_ratio = (
            float(magnitude_error / Decimal.from_float(magnitude.error))
            if magnitude.error
            else 0.0
            if not magnitude_error
            else math.inf
        )
        if not all(math.isfinite(v.error) for v in (*represented, magnitude, residual)):
            raise ArithmeticError("Non-finite running error bound")
        return dict(
            branch=branch,
            ray_level_offset=str(level_error),
            exact_ray_level_offset=str(level_error),
            ray_bound=str(bound),
            ray_ratio=level_ratio,
            position_ratio=level_ratio,
            cartesian_level_offset=str(point_value - target),
            coordinate_level_shift=str(point_value - ray_value),
            coordinate_rounding_diagnostic=str(
                Decimal(2) ** -53
                * sum(
                    abs(g) * abs(Decimal.from_float(p))
                    for g, p in zip(point_gradient, point, strict=True)
                )
            ),
            coordinate_matches=coordinate_matches,
            coordinate_rounding_passed=all(coordinate_matches),
            evaluated_residual=float(residual),
            gradient_ratio=max(ratios),
            gravity_ratio=gravity_ratio,
            ray_passed=level_ratio <= 1,
            gradient_passed=max(ratios) <= 1,
            gravity_passed=gravity_ratio <= 1,
            gradient_errors=[str(v) for v in errors],
            gradient_bounds=[str(v) for v in bounds],
            gravity_error=str(magnitude_error),
            gravity_bound=str(Decimal.from_float(magnitude.error)),
            passed=level_ratio <= 1
            and max(ratios) <= 1
            and gravity_ratio <= 1
            and all(coordinate_matches),
        )
