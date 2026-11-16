"""Brent's method for directional surface radii."""

from collections.abc import Callable
from dataclasses import dataclass

from rocheometry._computers.surface_radius.bisection import (
    BisectionSurfaceRadiusComputer,
)
from rocheometry._computers.surface_radius.radial import _RootSolver, _surface_radius
from rocheometry._geometry import Point3D
from rocheometry._numerics import brent_root, brent_root_adjacent
from rocheometry._roche.field import RocheField


def _node_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Use the shared 256-iteration Brent guard with adjacent-float stopping."""
    return brent_root_adjacent(
        function,
        left,
        right,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


@dataclass(frozen=True, slots=True)
class BrentSurfaceRadiusComputer(BisectionSurfaceRadiusComputer):
    """A Brent strategy for component-enclosing surface radii.

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

    def node_radius(  # type: ignore[missing-override-decorator]
        self, field: RocheField, depth: float, direction: Point3D, /
    ) -> float:
        """Solve a quadrature ray to adjacent floats, retaining thin-neck DD."""
        return self._solve(field, depth, direction, _node_root)

    def __call__(self, field: RocheField, depth: float, direction: Point3D, /) -> float:
        return self._solve(field, depth, direction, brent_root)

    def _solve(
        self,
        field: RocheField,
        depth: float,
        direction: Point3D,
        root_solver: _RootSolver,
        /,
    ) -> float:
        return _surface_radius(
            field,
            depth,
            direction,
            self.l1_computer,
            self.relative_tolerance,
            self.max_inner_contractions,
            root_solver,
            inverse_radius=True,
            outer_computer=self.outer_computer,
            watershed_computer=self.watershed_computer,
            exact_anchor=self.exact_anchor(field, depth)
            if callable(self.exact_anchor)
            else self.exact_anchor,
        )
