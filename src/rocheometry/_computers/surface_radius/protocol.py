"""Protocol for interchangeable surface-radius computations."""

from typing import Protocol

from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


class SurfaceRadiusComputer(Protocol):
    """A callable strategy for a component surface or watershed part radius.

    A computer may use its own dependency instead of the graph's node,
    provided it computes the same quantity within the consumer's tolerance.
    A radius computer's L1 must not lie inside surfaces selected near fill 1,
    and the selection's L1 defines fill 1.
    """

    def __call__(
        self,
        field: RocheField,
        depth: float,
        direction: Point3D,
        /,
    ) -> float:
        """Compute the first outward intersection along a unit direction.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.
            direction: Unit component-local direction vector.

        Returns:
            The first outward intersection distance in separation units.
        """
        ...

    def node_radius(
        self, field: RocheField, depth: float, direction: Point3D, /
    ) -> float:
        """Return a node radius; Brent uses adjacent-float acceptance."""
        ...
