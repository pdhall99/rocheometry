"""Bracketed bisection for directional surface radii."""

from __future__ import annotations

import reprlib
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from sys import float_info
from typing import TYPE_CHECKING

from rocheometry._computers.contact import default_outer, default_watershed
from rocheometry._computers.surface_radius.radial import _surface_radius
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics import bisect_root
from rocheometry._roche.field import RocheField
from rocheometry._validation import _normalize_finite_positive_float, require_int

if TYPE_CHECKING:
    from rocheometry._computers.critical_point import CriticalPointComputer

# Resolution budget: radius solve relative tolerance.
RADIUS_SOLVE_RELATIVE_TOLERANCE = 256.0 * float_info.epsilon


@dataclass(frozen=True, slots=True)
class BisectionSurfaceRadiusComputer:
    """A bisection strategy for component-enclosing surface radii.

    Solve the first boundary intersection in separation units. Detached and
    critical brackets use the L1 distance. Contact rays stop at the watershed
    or meet the envelope within the outer-point and rim bounds. Brackets are
    scaled to retain accuracy for small surfaces.

    Attributes:
        outer_computer: Critical-point strategy defining the outer opening.
        watershed_computer: Divider used for contact parts.
        l1_computer: Component-local L1 point computer.
        relative_tolerance: Finite positive tolerance in scaled root coordinates.
        max_inner_contractions: Positive maximum number of inner-bracket
            attempts; each unsuccessful attempt reduces radius tenfold.
        exact_anchor: Whether to extend exact-anchor factoring beyond the
            existing contact/front-ray branches, or a cached surface decision.
            Reference wiring always extends it to every ray.
    """

    l1_computer: CriticalPointComputer
    relative_tolerance: float = RADIUS_SOLVE_RELATIVE_TOLERANCE
    max_inner_contractions: int = 32
    outer_computer: CriticalPointComputer = dataclass_field(
        default_factory=default_outer, kw_only=True
    )
    watershed_computer: WatershedComputer = dataclass_field(
        default_factory=default_watershed, kw_only=True
    )
    exact_anchor: bool | Callable[[RocheField, float], bool] = dataclass_field(
        default=True, kw_only=True
    )

    def __post_init__(self) -> None:
        if not callable(self.l1_computer):
            raise TypeError("l1_computer: expected a callable computer")
        if not isinstance(self.exact_anchor, bool) and not callable(self.exact_anchor):
            raise TypeError("exact_anchor: expected a bool or callable")
        object.__setattr__(
            self,
            "relative_tolerance",
            _normalize_finite_positive_float(
                self.relative_tolerance,
                name="relative_tolerance",
            ),
        )
        max_inner_contractions = require_int(
            self.max_inner_contractions,
            name="max_inner_contractions",
        )
        if max_inner_contractions < 1:
            raise ValueError(
                "max_inner_contractions: expected a positive integer, "
                f"got {reprlib.repr(max_inner_contractions)}"
            )

    def node_radius(
        self, field: RocheField, depth: float, direction: Point3D, /
    ) -> float:
        """Use this strategy's directional solve for a quadrature node."""
        return self(field, depth, direction)

    def __call__(
        self,
        field: RocheField,
        depth: float,
        direction: Point3D,
        /,
    ) -> float:
        """Return the first radial intersection with the component-enclosing branch.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1 in canonical Jacobi units.
            direction: Unit direction vector in component-local coordinates.

        Raises:
            RuntimeError: A closed intersection cannot be bracketed or the root
                solver fails to converge.
        """
        return _surface_radius(
            field,
            depth,
            direction,
            self.l1_computer,
            self.relative_tolerance,
            self.max_inner_contractions,
            bisect_root,
            outer_computer=self.outer_computer,
            watershed_computer=self.watershed_computer,
            exact_anchor=self.exact_anchor(field, depth)
            if callable(self.exact_anchor)
            else self.exact_anchor,
        )
