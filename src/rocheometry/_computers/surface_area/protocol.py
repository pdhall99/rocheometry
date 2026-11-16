"""Protocol for interchangeable surface area computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class SurfaceAreaComputer(Protocol):
    """A callable computing surface area of a closed equipotential.

    Results use separation units squared.
    """

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return the selected surface's surface area.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.
        """
        ...
