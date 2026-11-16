"""Protocol for interchangeable integrated mean curvature computations."""

from typing import Protocol

from rocheometry._roche.field import RocheField


class IntegratedMeanCurvatureComputer(Protocol):
    """A callable computing integrated mean curvature of a closed equipotential.

    Results use separation units. The mean curvature convention is
    the mean of the principal curvatures, positive for a sphere.
    """

    def __call__(self, field: RocheField, depth: float, /) -> float:
        """Return the selected surface's integrated mean curvature.

        Args:
            field: Normalized component-local field.
            depth: Depth below exact L1, bounded by the outer-opening gap.
        """
        ...
