"""Contracts for Bézier section paths."""

from typing import Protocol

from rocheometry._bezier import CompoundBezierPath
from rocheometry._planes import SectionPlane
from rocheometry._roche.field import RocheField


class SectionPathComputer(Protocol):
    """A callable strategy returning a compound component-local section outline.

    Strategies own their dependencies, including any critical-point solver or
    lobe-volume calibration. `depth` is U_L1(exact)-u, strictly below the
    first outer-opening gap.
    """

    def __call__(
        self,
        field: RocheField,
        depth: float,
        plane: SectionPlane,
        /,
    ) -> CompoundBezierPath:
        """Compute the selected equipotential's planar Bézier section.

        Args:
            field: Normalized component-local field; strategies may use only its mass
                ratio.
            depth: Depth below exact L1, strictly below the outer-opening gap.
            plane: Component-local plane and its in-plane coordinate convention.

        Returns:
            All closed pieces in the plane's coordinates, in separation units.
        """
        ...

    def envelope(
        self, field: RocheField, depth: float, plane: SectionPlane, /
    ) -> CompoundBezierPath:
        """Return every envelope run with exactly the matching outline segments.

        Detached and critical outlines are wholly envelope. Contact parts may
        have open runs, closed stellar loops, or no envelope boundary at all.
        """
        ...
