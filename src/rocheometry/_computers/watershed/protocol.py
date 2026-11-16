"""Contracts for the gravity-line divider of a common envelope."""

from __future__ import annotations

from typing import Protocol

from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


class WatershedComputer(Protocol):
    """A level-independent sheet represented by downhill lines from L1."""

    def rim_at_depth(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        """Return a rim point at continuous depth U_L1-U, with exact L1 at zero.

        Keeping depth independent of a rounded potential retains sub-ulp level
        differences during section-cut inversions. Implementations need not
        represent gravity lines, nor be callable themselves.
        """
        ...

    def rim(self, field: RocheField, depth: float, psi: float, /) -> Point3D:
        """Return the sheet's point at the given level and transverse azimuth."""
        ...

    def sheet_point_on_ray(self, field: RocheField, direction: Point3D, /) -> Point3D:
        """Return the point where a component-centred forward ray meets the sheet."""
        ...

    def cut_height(self, field: RocheField, rho: float, psi: float, /) -> float:
        """Return the sheet's axial height at transverse radius rho and azimuth psi."""
        ...
