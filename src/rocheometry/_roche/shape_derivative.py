"""Implicit radial motion of a Roche equipotential as its mass ratio changes."""

from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


def ray_log_qi_velocity(
    field: RocheField, level_slope: float, direction: Point3D, radius: float
) -> float:
    """Return `dr/d ln(qi)` along a regular ray intersection.

    Differentiating `U(r*d, qi) = level(qi)` gives
    `(level_slope - partial U/partial ln(qi)) / (gradient(U) dot d)`.
    The ray and component-local coordinate chart remain fixed.

    Args:
        field: Component-local Jacobi field at the current mass ratio.
        level_slope: Derivative of the equipotential level against `ln(qi)`.
        direction: Unit ray from the component centre.
        radius: Surface radius in separation units, at a regular intersection
            with nonzero radial gradient.

    Returns:
        Radial velocity per unit logarithmic mass ratio, in separation units.
    """
    x, y, z = radius * direction[0], radius * direction[1], radius * direction[2]
    gradient = field.gradient(x, y, z)
    radial_gradient = (
        gradient[0] * direction[0]
        + gradient[1] * direction[1]
        + gradient[2] * direction[2]
    )
    return (level_slope - field.log_qi_derivative(x, y, z)) / radial_gradient
