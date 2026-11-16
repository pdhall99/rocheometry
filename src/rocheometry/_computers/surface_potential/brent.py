"""Surface-potential selection by Brent volume inversion."""

from dataclasses import dataclass

from rocheometry._computers.surface_potential.bisection import (
    BisectionSurfacePotentialComputer,
)
from rocheometry._roche.field import RocheField


@dataclass(frozen=True, slots=True, kw_only=True)
class BrentSurfacePotentialComputer(BisectionSurfacePotentialComputer):
    """A Brent strategy for selecting surfaces by enclosed volume.

    Accept the canonical Jacobi level when the recovered volume radius differs
    from the target by at most the configured relative fill tolerance. The L1
    level is the lower bracket and is returned directly at fill `1`.
    Dependencies must use compatible lobe normalization and field conventions.

    Attributes:
        volume_computer: Trial equipotential-volume computer in separation cubed.
        l1_computer: Critical point defining fill `1` and the lower potential.
        lobe_volume_computer: Lobe volume defining the target radius scale.
        volume_radius_fill_tolerance: Finite positive relative radius tolerance.
        max_iterations: Positive maximum number of root-solve iterations.
        max_bracket_expansions: Positive maximum number of upper-bracket expansions.
    """

    def __call__(self, field: RocheField, volume_radius_fill: float, /) -> float:
        """Select a depth directly, without an intermediate absolute level."""
        return self._select_depth(field, volume_radius_fill, use_brent=True)
