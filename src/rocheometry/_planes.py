"""Immutable planes with explicit two-dimensional coordinate frames."""

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Literal, TypeAlias, get_args

from rocheometry._geometry import Point2D, Point3D
from rocheometry._validation import (
    _normalize_finite_float,
    normalize_inclination,
    require_choice,
    require_instance,
)

SectionAxis: TypeAlias = Literal["x", "y", "z"]
"""The coordinate axis normal to a section plane."""


@dataclass(frozen=True, slots=True)
class CentrelinePlane:
    """A plane containing the line of centres at a folded inclination.

    The 2D axes are u along x and v along
    `(0, cos(inclination), -sin(inclination))`, with the 3D origin as origin.
    The plane is frame-agnostic: coordinates use the caller's frame and units.
    At quadrature, this is the sky plane for an observer at inclination i.
    The line of sight is the plane normal, `(0, sin i, cos i)`.
    A section intersects this plane; a silhouette projects onto it.

    Args:
        inclination: Finite radians in `[0, pi]`. Supplementary angles fold
            into `[0, pi/2]` using Roche symmetry. Use `math.radians()` for degrees.

    Attributes:
        inclination: Normalized inclination in radians, in `[0, pi/2]`.
    """

    inclination: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "inclination", normalize_inclination(self.inclination))

    def project(self, point: Point3D) -> Point2D:
        """Orthogonally project a 3D point into the documented 2D axes."""
        x, y, z = point
        _, c, s = _transverse_direction(self)
        return x, y * c + z * s

    def lift(self, point: Point2D) -> Point3D:
        """Lift a point in the documented 2D axes onto this plane."""
        u, v = point
        _, c, s = _transverse_direction(self)
        return u, v * c, v * s


@dataclass(frozen=True, slots=True)
class AxisNormalPlane:
    """An axis-normal plane in the caller's frame and length units.

    Planar coordinates are `(y, z)` for an x section, `(x, z)` for a y section
    and `(x, y)` for a z section. The 2D origin is the foot of the
    perpendicular from the frame's origin onto the plane. Zero-offset y and z
    planes give the same geometry as the matching centreline planes.

    Attributes:
        axis: Normal axis, `x`, `y` or `z`.
        offset: Signed normal coordinate in the caller's length units, as a float.
    """

    axis: SectionAxis
    offset: float

    def __post_init__(self) -> None:
        require_choice(
            require_instance(self.axis, str, name="axis"),
            get_args(SectionAxis),
            name="axis",
        )
        object.__setattr__(
            self, "offset", _normalize_finite_float(self.offset, name="offset")
        )

    def project(self, point: Point3D) -> Point2D:
        """Orthogonally project a 3D point into the documented 2D axes."""
        x, y, z = point
        if self.axis == "x":
            return y, z
        if self.axis == "y":
            return x, z
        return x, y

    def lift(self, point: Point2D) -> Point3D:
        """Lift a point in the documented 2D axes onto this plane."""
        u, v = point
        if self.axis == "x":
            return self.offset, u, v
        if self.axis == "y":
            return u, self.offset, v
        return u, v, self.offset


SectionPlane: TypeAlias = CentrelinePlane | AxisNormalPlane
"""A supported centreline or axis-normal equipotential section plane."""

# Derived: orbital plane.
ORBITAL_PLANE = CentrelinePlane(0.0)
"""The centreline plane coinciding with the orbital plane."""


def _require_silhouette_plane(plane: object) -> CentrelinePlane:
    if isinstance(plane, AxisNormalPlane):
        raise TypeError(
            "silhouettes project along the line of sight, normal to a CentrelinePlane"
        )
    return require_instance(plane, CentrelinePlane, name="plane")


def _canonical_section_plane(plane: SectionPlane) -> tuple[SectionPlane, bool]:
    """Return the geometric plane and whether its second coordinate reverses."""
    if isinstance(plane, AxisNormalPlane) and plane.offset == 0.0:
        if plane.axis == "z":
            return ORBITAL_PLANE, False
        if plane.axis == "y":
            return CentrelinePlane(0.5 * pi), True
    return plane, False


def _transverse_direction(plane: CentrelinePlane) -> Point3D:
    inclination = plane.inclination
    if inclination == 0.0:
        return (0.0, 1.0, 0.0)
    if inclination == 0.5 * pi:
        return (0.0, 0.0, -1.0)
    return (0.0, cos(inclination), -sin(inclination))
