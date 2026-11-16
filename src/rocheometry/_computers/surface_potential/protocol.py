"""Protocol for interchangeable surface-potential computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class SurfacePotentialComputer(Protocol):
    """A callable strategy selecting depth below exact L1 from volume-radius fill.

    A computer may use its own dependency instead of the graph's node,
    provided it computes the same quantity within the consumer's tolerance.
    A radius computer's L1 must not lie inside surfaces selected near fill 1,
    and the selection's L1 defines fill 1.
    """

    def __call__(
        self,
        field: RocheField,
        volume_radius_fill: float,
        /,
    ) -> float:
        """Compute the depth for the requested volume-radius fill.

        Args:
            field: Normalized component-local field.
            volume_radius_fill: Positive surface-to-lobe radius ratio, below the maximum part fill.

        Returns:
            A depth in canonical Jacobi units; fill `1` gives exactly zero.
        """
        ...
