"""Independent geometric probes shared by the guarantee and join studies."""

from math import cos, pi, sin

import rocheometry
from rocheometry._geometry import Point2D
from rocheometry._numerics.roots import brent_root


def crossing_at_depth(
    geometry: rocheometry.ComponentGeometry,
    depth: float,
    plane: rocheometry.CentrelinePlane,
) -> Point2D:
    """Locate the LOS-stationary rim at the stored surface depth."""
    field = geometry.field
    sine, cosine = sin(plane.inclination), cos(plane.inclination)

    def point(psi: float) -> tuple[float, float, float]:
        return geometry._computers.watershed.rim_at_depth(field, depth, psi)

    def residual(psi: float) -> float:
        _, gy, gz = field.gradient(*point(psi))
        return gy * sine + gz * cosine

    psi = (
        -pi / 2
        if plane.inclination == pi / 2
        else brent_root(
            residual,
            -pi / 2,
            0.0,
            absolute_tolerance=0.0,
            relative_tolerance=1e-15,
        )
    )
    return plane.project(point(psi))
