"""Conversions between sphere volumes and volume-equivalent radii."""

from math import pi


def volume_radius_from_volume(volume: float) -> float:
    """Return the radius of a sphere enclosing the supplied volume.

    Args:
        volume: Enclosed volume in length units cubed; must be non-negative.

    Returns:
        The equivalent spherical radius in the corresponding length units.
    """
    return (3.0 * volume / (4.0 * pi)) ** (1.0 / 3.0)


def sphere_volume(radius: float) -> float:
    """Return the volume of a sphere with the supplied radius.

    Args:
        radius: Spherical radius in caller-chosen length units.

    Returns:
        The volume in those length units cubed.
    """
    return (4.0 * pi / 3.0) * radius**3
