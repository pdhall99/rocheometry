"""Public surface nodes, differential geometry and area weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isclose
from typing import TYPE_CHECKING

from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from rocheometry._validation import (
    _normalize_finite_float,
    normalize_point3d,
    require_instance,
    require_positive,
)

# Resolution budget: quadrature normal length tolerance.
QUADRATURE_NORMAL_LENGTH_TOLERANCE = 1e-12

if TYPE_CHECKING:
    from rocheometry._computers.surface_sample.sample import SurfaceSample


@dataclass(frozen=True, slots=True)
class SurfaceQuadrature:
    """Surface-integral nodes in the producer's coordinate frame and length units.

    All fields are equal-length tuples. Normals are unit outward vectors.
    Gravities are in units G*(M1+M2)/separation**2, including in binary views.
    Area weights have units of length squared; mean and Gaussian curvatures
    have units of inverse length and inverse length squared. A sphere has
    H = 1/R and K = 1/R**2.

    Standard samples are in polar-major, azimuth-minor order, with the
    reflections (y, z), (y, -z), (-y, z), (-y, -z) consecutive for each node.
    Contact nodes cover only the envelope share, excluding the watershed cut,
    like `area()`.

    Smooth integrands converge like `area()`. Integrands with kinks, such as
    max(0, n·v) for projected area, converge more slowly; at quadrature,
    `silhouette(CentrelinePlane(inclination))` gives the projected boundary
    directly.

    Near saddles gravity vanishes and the normal is undefined, so nodes carry a
    backward-error contract rather than a forward bound; see
    [node accuracy](https://pdhall99.github.io/rocheometry/guide/accuracy/#quadrature-node-accuracy).
    At exactly critical fill, local curvatures near the L1 apex are amplified by
    the radius precision; integrated quantities are unaffected.

    Attributes:
        positions: Surface positions in the producer's axes and length units.
        normals: Unit outward normals, -gradient(U)/abs(gradient(U)).
        gravities: Local effective-gravity magnitudes, abs(gradient(U)).
        weights: Positive area weights for sums approximating surface integrals.
        mean_curvatures: Mean principal curvature H, positive on a sphere.
        gaussian_curvatures: Gaussian curvature K, the product of principal curvatures.
    """

    positions: tuple[Point3D, ...]
    normals: tuple[Point3D, ...]
    gravities: tuple[float, ...]
    weights: tuple[float, ...]
    mean_curvatures: tuple[float, ...]
    gaussian_curvatures: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(require_instance(self.positions, tuple, name="positions"))
        for name in (
            "positions",
            "normals",
            "gravities",
            "weights",
            "mean_curvatures",
            "gaussian_curvatures",
        ):
            values = require_instance(getattr(self, name), tuple, name=name)
            if len(values) != count:
                raise ValueError(f"{name}: expected {count} entries to match positions")
            if name in ("positions", "normals"):
                normalized = tuple(
                    normalize_point3d(value, name=f"{name}[{i}]")
                    for i, value in enumerate(values)
                )
                if name == "normals" and any(
                    not isclose(
                        hypot(*v),
                        1.0,
                        rel_tol=QUADRATURE_NORMAL_LENGTH_TOLERANCE,
                        abs_tol=0,
                    )
                    for v in normalized
                ):
                    raise ValueError("normals: expected unit vectors")
            else:
                normalized = tuple(
                    _normalize_finite_float(value, name=f"{name}[{i}]")
                    for i, value in enumerate(values)
                )
                if name in ("gravities", "weights"):
                    for value in normalized:
                        require_positive(value, name=name)
            object.__setattr__(self, name, normalized)

    def __len__(self) -> int:
        """Return the number of surface nodes, including all reflections."""
        return len(self.positions)


def _quadrature_from_sample(
    field: RocheField, sample: SurfaceSample
) -> SurfaceQuadrature:
    from rocheometry._computers.surface_sample.terms import ray_geometry

    positions: list[Point3D] = []
    normals: list[Point3D] = []
    gravities: list[float] = []
    weights: list[float] = []
    means: list[float] = []
    gaussians: list[float] = []
    for (direction, weight), radius in zip(sample.rays, sample.radii, strict=True):
        local = ray_geometry(field, direction, radius, sample=sample)
        # adj(Hess U / |g|) gives adj(Hess U)/|g|^2. Contracting with
        # the unit normal avoids forming |g|^4 and unnecessarily large cofactors.
        h = tuple(
            tuple(value / local.gravity for value in row) for row in local.hessian
        )
        a, b, c = h[0]
        _, d, e = h[1]
        _, _, f = h[2]
        nx, ny, nz = local.normal
        gaussian = (
            nx * nx * (d * f - e * e)
            + ny * ny * (a * f - c * c)
            + nz * nz * (a * d - b * b)
            + 2 * nx * ny * (c * e - b * f)
            + 2 * nx * nz * (b * e - c * d)
            + 2 * ny * nz * (b * c - a * e)
        )
        x, y, z = local.position
        for sy, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            positions.append((x, sy * y, sz * z))
            normals.append((nx, sy * ny, sz * nz))
            gravities.append(local.gravity)
            weights.append(weight * local.area / 4)
            means.append(local.mean_curvature)
            gaussians.append(gaussian)
    return SurfaceQuadrature(
        tuple(positions),
        tuple(normals),
        tuple(gravities),
        tuple(weights),
        tuple(means),
        tuple(gaussians),
    )
