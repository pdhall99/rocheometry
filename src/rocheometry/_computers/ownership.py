"""Shared certificates for membership of a component's watershed region."""

from math import atan2, hypot, sqrt
from sys import float_info

from rocheometry._computers.watershed import WatershedComputer
from rocheometry._computers.watershed.traced import TracedWatershedComputer
from rocheometry._errors import ComponentOwnershipError
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import CriticalPoint

# Insensitive control: outward allowances in field and cylinder certificates.
FIELD_CERTIFICATE_BOUND_ULPS = 32
# Insensitive control: rounding allowance at the watershed side test.
OWNERSHIP_SIDE_ULPS = 64


def invariant_cylinder(field: RocheField, x: float, lower: float, upper: float) -> bool:
    """Certify rho decreases and U_x<0 on a cylinder's right face."""
    m, mu = field.component_mass, field.companion_mass
    if (
        not 0 < x < 1
        or m / hypot(x, upper) ** 3
        <= 1 + FIELD_CERTIFICATE_BOUND_ULPS * float_info.epsilon
    ):
        return False
    nodes = [lower, upper]
    # U_x on this face has at most one interior extremum. Equating its
    # two inverse fifth-power terms gives the analytic exponent 2/5.
    k = (m * x / (mu * (1 - x))) ** (2.0 / 5.0)
    if k != 1:
        square = (k * (1 - x) ** 2 - x * x) / (1 - k)
        if lower * lower < square < upper * upper:
            nodes.append(sqrt(square))
    for rho in nodes:
        first = m * x / hypot(x, rho) ** 3
        second = mu * (1 - x) / hypot(1 - x, rho) ** 3
        value = -first + second + x - mu
        allowance = (
            FIELD_CERTIFICATE_BOUND_ULPS
            * float_info.epsilon
            * (first + second + abs(x) + mu)
        )
        if value >= -allowance:
            return False
    return True


def certify_part(
    field: RocheField,
    l1: CriticalPoint,
    watershed: WatershedComputer,
    point: Point3D,
    *,
    contact: bool = True,
) -> None:
    """Certify the relaxed winner belongs to this watershed part.

    Rho decreases under ascent wherever T>1. An invariant cylinder with
    U_x<0 on its right face therefore carries a point into the star, or
    into a known interior circle of the sheet. Bounds on that circle use
    the triangle inequality on the existing cosine coefficients; the
    certificate uses the public watershed range and integration depth.
    """
    if not contact or point[0] <= 0:
        return
    x, y, z = point
    rho = hypot(y, z)
    if x < l1.position[0] and invariant_cylinder(field, x, 0.0, rho):
        return
    tolerance = (
        max(
            OWNERSHIP_SIDE_ULPS * float_info.epsilon,
            getattr(watershed, "tolerance", 0.0),
        )
        * l1.position[0]
    )
    try:
        height = watershed.cut_height(field, rho, atan2(z, y))
    except ValueError:
        if not isinstance(watershed, TracedWatershedComputer):
            raise
        end = watershed.cut_height_computer.outer_section(field)
        inner = end.radius[0] - sum(abs(v) for v in end.radius[1:]) - tolerance
        if not 0 < inner < rho:
            raise
        cut = watershed.cut_section_computer(field, inner)
        middle = (
            1 - cut.x_l1 - cut.height[0] if cut.mirrored else cut.x_l1 + cut.height[0]
        )
        right = middle - sum(abs(v) for v in cut.height[1:]) - tolerance
        if x <= right and invariant_cylinder(field, right, inner, rho):
            return
        raise
    if x > height + tolerance:
        raise ComponentOwnershipError(
            "global line-of-sight maximum lies beyond the watershed"
        )


def belongs_to_component(
    field: RocheField, l1: CriticalPoint, watershed: WatershedComputer, point: Point3D
) -> bool:
    x, y, z = point
    if x <= 0:
        return True
    if x >= 1:
        return False
    rho = hypot(y, z)
    if x < l1.position[0] and invariant_cylinder(field, x, 0.0, rho):
        return True
    try:
        height = watershed.cut_height(field, rho, atan2(z, y))
    except ValueError:
        # Outside the public cut chart, the existing invariant-cylinder
        # continuation can still certify membership using an interior cut.
        try:
            certify_part(field, l1, watershed, point)
        except ValueError:
            return False
        except ComponentOwnershipError:
            return False
        return True
    # Use the actual side of the sheet. The certificate's rounding
    # allowance alone must not enlarge a component across its watershed.
    if x > height:
        return False
    certify_part(field, l1, watershed, point)
    return True
