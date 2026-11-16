"""Extended inclination and mass-ratio accuracy checks for projected limbs."""

from math import atan2, cos, hypot, pi, sin

import pytest

import rocheometry
from rocheometry import Reference
from rocheometry._numerics.roots import brent_root
from tests.computers.silhouette.test_projected_outlines import projected


@pytest.mark.slow
@pytest.mark.parametrize(
    "qi,inclination",
    (
        (1e-6, pi / 3),
        (1e-3, pi / 6),
        (20.0, pi / 6),
        (1e3, pi / 6),
        (1e3, pi / 3),
        (1e6, pi / 6),
    ),
)
def test_critical_reference_limb_geometry_after_shared_third_derivatives(
    qi: float, inclination: float
) -> None:
    surface = rocheometry.ComponentGeometry(qi, accuracy=Reference()).lobe()
    path = surface.silhouette(rocheometry.CentrelinePlane(inclination)).paths[0]
    field = projected(surface, inclination)
    for segment in path.segments:
        # Fold the lower half into the upper field, preserving segment order.
        start = atan2(abs(segment.p0[1]), segment.p0[0])
        end = atan2(abs(segment.p3[1]), segment.p3[0])
        for index in range(200):
            angle = start + (end - start) * (index + 0.5) / 200
            c, s = cos(angle), sin(angle)

            def cross(t: float) -> float:
                x, y = segment.evaluate(t)
                return c * abs(y) - s * x

            t = brent_root(
                cross, 0.0, 1.0, absolute_tolerance=1e-15, relative_tolerance=1e-15
            )
            point = segment.evaluate(t)
            assert abs(hypot(*point) - field.radius(angle)) <= 1e-10 * field.scale
