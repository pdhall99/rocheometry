"""Protocol for interchangeable surface sampling strategies."""

from typing import Protocol

from rocheometry._computers.surface_sample.sample import SurfaceSample
from rocheometry._roche.field import RocheField


class SurfaceSampler(Protocol):
    """A callable producing radii on reflection-symmetric quadrature rays.

    Readers may integrate only quantities symmetric under both y and z
    reflection; the weights include the fourfold reflection factor.
    """

    def __call__(self, field: RocheField, depth: float, /) -> SurfaceSample:
        """Sample a closed component-enclosing equipotential.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.

        Returns:
            Weighted unit rays and matching finite positive surface radii.
        """
        ...
