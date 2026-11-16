"""Protocol for centreline critical-point computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint


class CriticalPointComputer(Protocol):
    """A callable strategy for a component-local centreline stationary point.

    Computers need not be hashable. Their configuration must remain unchanged
    while shared caches retain results computed with them.

    A computer may use its own dependency instead of the graph's node,
    provided it computes the same quantity within the consumer's tolerance.
    A radius computer's L1 must not lie inside surfaces selected near fill 1,
    and the selection's L1 defines fill 1.

    Producers return `CriticalPoint`, including the Jacobi Hessian in local
    axes and intrinsic G*(M1+M2)/a**3 units, where a is the separation.
    """

    def __call__(self, field: RocheField, /) -> CriticalPoint:
        """Compute a centreline critical point in the supplied field.

        Args:
            field: Normalized component-local field whose stationary point is sought.

        Returns:
            A point in separation units with a canonical Jacobi potential and
            its intrinsic Hessian. Other result types are not accepted.
        """
        ...
