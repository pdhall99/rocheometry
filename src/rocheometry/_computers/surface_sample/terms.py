"""Local area, mean-curvature, and support terms on a quadrature ray."""

from dataclasses import dataclass
from math import hypot

from rocheometry._computers.neck_ray import _refined_neck_ray
from rocheometry._computers.surface_sample.sample import SurfaceSample, _NeckSample
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField


def ray_terms(
    field: RocheField,
    direction: Point3D,
    radius: float,
    *,
    sample: SurfaceSample | None = None,
) -> tuple[float, float, float]:
    """Return `(a, H*a, p)` at a regular star-shaped surface intersection.

    For the outward normal `n = -gradient(U)/|gradient(U)|`, `a` is area per
    unit solid angle, `H` is the mean of the principal curvatures (a sphere
    has `H = 1/R`), and `p = x dot n` is the support function. The Jacobi
    field has Laplacian two away from its point masses, so
    `H = (U_nn - 2)/(2*|gradient(U)|)`. Lengths use separation units.

    Args:
        field: Component-local Jacobi field.
        direction: Unit ray from the selected component centre.
        radius: Positive surface radius in separation units.
        sample: Optional shared sample retaining contact-neck displacements.

    Raises:
        ValueError: The intersection has non-positive outward radial gravity.
    """
    local = ray_geometry(field, direction, radius, sample=sample)
    return local.area, local.mean_curvature * local.area, local.support


@dataclass(frozen=True, slots=True)
class _RayGeometry:
    position: Point3D
    normal: Point3D
    gravity: float
    area: float
    mean_curvature: float
    support: float
    hessian: tuple[Point3D, Point3D, Point3D]


def _gradient(
    field: RocheField, direction: Point3D, radius: float, sample: SurfaceSample | None
) -> Point3D:
    if isinstance(sample, _NeckSample):
        # Keep the low root component until after evaluating the gradient.
        # Absolute Cartesian vertices alone cannot encode this information.
        return _refined_neck_ray(field, sample.origin, sample.depth, direction, radius)[
            1
        ]
    return field.gradient(*(radius * d for d in direction))


def ray_geometry(
    field: RocheField,
    direction: Point3D,
    radius: float,
    *,
    sample: SurfaceSample | None = None,
) -> _RayGeometry:
    """Compute the shared local geometry used by readers and public quadrature."""
    x, y, z = (radius * direction[0], radius * direction[1], radius * direction[2])
    gx, gy, gz = _gradient(field, direction, radius, sample)
    magnitude = hypot(gx, gy, gz)
    radial_gravity = -(gx * direction[0] + gy * direction[1] + gz * direction[2])
    if radial_gravity <= 0.0:
        raise ValueError("ray must meet a regular star-shaped surface")
    a = radius * radius * magnitude / radial_gravity
    nx, ny, nz = gx / magnitude, gy / magnitude, gz / magnitude
    h0, h1, h2 = field.hessian(x, y, z)
    u_nn = (
        nx * (h0[0] * nx + h0[1] * ny + h0[2] * nz)
        + ny * (h1[0] * nx + h1[1] * ny + h1[2] * nz)
        + nz * (h2[0] * nx + h2[1] * ny + h2[2] * nz)
    )
    mean_curvature = (u_nn - 2.0) / (2.0 * magnitude)
    support = radius * radial_gravity / magnitude
    return _RayGeometry(
        (x, y, z), (-nx, -ny, -nz), magnitude, a, mean_curvature, support, (h0, h1, h2)
    )


def ray_gravity_flux(
    field: RocheField,
    direction: Point3D,
    radius: float,
    *,
    sample: SurfaceSample | None = None,
) -> float:
    """Return |gradient(U)| dA per unit solid angle on an envelope ray."""
    gradient = _gradient(field, direction, radius, sample)
    radial = -sum(g * d for g, d in zip(gradient, direction, strict=True))
    if radial <= 0:
        raise ValueError("ray must meet a regular star-shaped surface")
    return radius * radius * sum(g * g for g in gradient) / radial
