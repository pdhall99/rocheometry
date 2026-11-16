"""Independent node backward-error measurements, with executed operation counts.

These are measured acceptance envelopes, not an analytic claim that counting
operations controls arbitrary cancellation inside a compound term. Epsilon is
binary64 machine epsilon (2**-52). A faithfully rounded sqrt, hypot or fsum is
one primitive; negation, comparisons and exact copies have no rounding charge.
Term-formation arithmetic is counted, including binding cached coefficients.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FunctionType
from typing import Any
from unittest.mock import patch

from rocheometry._computers import neck_ray
from rocheometry._computers.surface_radius import brent
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField

#: What was solved at each quadrature node, keyed by ray direction.
NodeRecords = dict[Point3D, dict[str, Any]]


def copied(function: Callable[..., Any], namespace: dict[str, Any]) -> FunctionType:
    return FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )


@contextmanager
def capture_nodes() -> Iterator[NodeRecords]:
    """Record the residual actually solved, including DD refinement terms.

    All modifications are in-memory study instrumentation. The copied DD
    function executes the original statements and records existing locals.
    """
    records: NodeRecords = {}
    current: list[Point3D] = []
    bind = RocheField._ray_residual
    solve = brent._node_root
    node = brent.BrentSurfaceRadiusComputer.node_radius

    def ray(
        field: RocheField,
        direction: Point3D,
        depth: float,
        reference_x: float | None = None,
        *,
        inverse_radius: bool = False,
        l1_x: float,
    ) -> Callable[[float], float]:
        function = bind(
            field,
            direction,
            depth,
            reference_x,
            inverse_radius=inverse_radius,
            l1_x=l1_x,
        )
        function._backward_spec = dict(  # type: ignore[missing-attribute]
            direction=direction,
            depth=depth,
            reference_x=reference_x,
            inverse_radius=inverse_radius,
            l1_x=l1_x,
        )
        return function

    def radius(
        computer: brent.BrentSurfaceRadiusComputer,
        field: RocheField,
        depth: float,
        direction: Point3D,
    ) -> float:
        current.append(direction)
        try:
            return node(computer, field, depth, direction)
        finally:
            current.pop()

    def root(
        function: Callable[[float], float], left: float, right: float, **kwargs: float
    ) -> float:
        coordinate = solve(function, left, right, **kwargs)
        if current:
            values = inspect.getclosurevars(function).nonlocals
            wrapped = "residual" in values
            base = values["residual"] if wrapped else function
            if wrapped:
                argument = values["outer_radius"] / coordinate
            else:
                argument = coordinate
            records[current[-1]] = dict(
                spec=base._backward_spec,  # type: ignore[missing-attribute]
                argument=argument,
                # Detached rebinding: three coefficient products and final
                # radius division. Contact: wrapper and report divisions.
                coordinate_operations=2 if wrapped else 4,
            )
        return coordinate

    def record_dd(
        direction: Point3D,
        radius: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
        centrifugal: tuple[float, float],
        depth: float,
    ) -> None:
        record = records.setdefault(direction, {})
        record["dd_terms"] = [
            tuple(first),
            tuple(second),
            tuple(centrifugal),
            (depth, 0.0),
        ]

    source = textwrap.dedent(inspect.getsource(neck_ray._ExactNeckRay.evaluate))
    old = "    return value, gradient"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "    record_dd(direction, radius, first, second, centrifugal, depth)\n" + old,
    )
    namespace = dict(vars(neck_ray), record_dd=record_dd)
    exec(compile(source, __file__ + ":dd-capture", "exec"), namespace)
    with (
        patch.object(RocheField, "_ray_residual", ray),
        patch.object(brent, "_node_root", root),
        patch.object(brent.BrentSurfaceRadiusComputer, "node_radius", radius),
        patch.object(neck_ray._ExactNeckRay, "evaluate", namespace["evaluate"]),
    ):
        yield records
