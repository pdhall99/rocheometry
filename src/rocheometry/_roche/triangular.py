"""Closed-form equilateral (triangular) points of the Jacobi field."""

from fractions import Fraction
from math import sqrt

from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint, Potential

# Derived: triangular point y coordinate.
TRIANGULAR_POINT_Y_COORDINATE = sqrt(3.0) / 2.0


def triangular_point(field: RocheField) -> CriticalPoint:
    """Return the triangular point at +y in component-local coordinates.

    It lies at unit distance from both masses for every qi. For an orbit
    counterclockwise about +z it leads the companion and trails this component.
    Its Jacobi Hessian uses local axes and intrinsic G*(M1+M2)/a**3 units.
    """
    m1, m2 = Fraction(field.component_mass), Fraction(field.companion_mass)
    total = m1 + m2
    potential = Potential.jacobi(float(Fraction(3, 2) * total - m1 * m2 / (2 * total)))
    position = (0.5, TRIANGULAR_POINT_Y_COORDINATE, 0.0)
    return CriticalPoint(position, potential, field.hessian(*position))
