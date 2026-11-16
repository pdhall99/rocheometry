"""Protocol for interchangeable surface-volume computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class SurfaceVolumeComputer(Protocol):
    """A callable strategy for component-enclosing equipotential volume.

    A computer may use its own dependency instead of the graph's node,
    provided it computes the same quantity within the consumer's tolerance.
    A radius computer's L1 must not lie inside surfaces selected near fill 1,
    and the selection's L1 defines fill 1.
    """

    def __call__(
        self,
        field: RocheField,
        depth: float,
        /,
    ) -> float:
        """Compute the enclosed volume of the selected equipotential.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.

        Returns:
            Enclosed volume in separation units cubed.
        """
        ...
