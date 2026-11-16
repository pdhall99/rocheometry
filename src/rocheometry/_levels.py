"""Boundary conversions between public potential reports and stored depth."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from rocheometry._computers.composition import ComputerSet
from rocheometry._roche.field import RocheField
from rocheometry._roche.opening import _admissibility_gap, _coverage_gap
from rocheometry._roche.potential import (
    Potential,
    PotentialConvention,
    require_potential,
)


@dataclass(frozen=True, slots=True)
class _Levels:
    l1: Fraction
    opening: Fraction
    admissibility_gap: float
    coverage_gap: float
    critical_report: Potential
    opening_report: Potential
    masses: tuple[Fraction, Fraction]

    def depth(self, potential: Potential) -> float:
        potential = require_potential(potential)
        if potential.convention is PotentialConvention.JACOBI:
            critical = self.critical_report.value
            opening = self.opening_report.value
            exact = Fraction(potential.value)
        else:
            m, c = self.masses if potential.component_id == 1 else self.masses[::-1]
            offset = c * c / (2 * (m + c))
            critical = float((Fraction(self.critical_report.value) - offset) / m)
            opening = float((Fraction(self.opening_report.value) - offset) / m)
            exact = Fraction(potential.value) * m + offset
        if potential.value == critical:
            return 0.0
        if potential.value == opening:
            return self.admissibility_gap
        return float(self.l1 - exact)

    def potential(self, depth: float) -> Potential:
        """Round the reported level once, without changing surface identity."""
        return Potential.jacobi(float(self.l1 - Fraction(depth)))


@lru_cache(maxsize=4096)
def _levels(field: RocheField, computers: ComputerSet) -> _Levels:
    l1 = computers.l1(field)
    opposite = field.reciprocal()
    first = computers.outer(field)
    second = computers.outer(opposite)
    candidates = (
        (field._axis_potential_exact(first.position[0]), first.potential),
        (opposite._axis_potential_exact(second.position[0]), second.potential),
    )
    opening, report = max(candidates, key=lambda pair: pair[0])
    exact_l1 = field._axis_potential_exact(l1.position[0])
    return _Levels(
        exact_l1,
        opening,
        _admissibility_gap(
            field, l1.position[0], first.position[0], second.position[0]
        ),
        _coverage_gap(field, l1.position[0], first.position[0], second.position[0]),
        l1.potential,
        report,
        (Fraction(field.component_mass), Fraction(field.companion_mass)),
    )
