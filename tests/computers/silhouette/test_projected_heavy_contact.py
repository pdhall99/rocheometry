"""Heavy contact limbs, rounding-aware start points and projected envelopes."""

from math import cos, dist, hypot, pi, sin

import pytest

import rocheometry
from rocheometry._computers.silhouette.projected import ProjectedField
from rocheometry._geometry import Point2D
from rocheometry._types import BinaryFrame
from tests._helpers import _depth
from tests.computers.silhouette._projection import ContactProjection, crossing_at_depth
from tests.computers.silhouette.test_projected_outlines import projected

QI = (1e6, 1e7, 1e8)
FACTORS = (1e-6, 0.01, 0.5, 0.99)
INCLINATIONS = (pi / 6, pi / 3, pi / 2)


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_heavy_contact_reference_and_fitted_containment(
    qi: float, factor: float, inclination: float
) -> None:
    binary = rocheometry.BinaryGeometry(qi, accuracy=rocheometry.Reference())
    surface = binary.component(1).surface(fill_out_factor=factor)
    plane = rocheometry.CentrelinePlane(inclination)
    points = tuple(plane.project(p) for p in surface.quadrature().positions)
    pole = surface.named_radius("pole")
    field = projected(surface.intrinsic_geometry, inclination)
    assert field.start_point is not None
    assert abs(field.value(*field.start_point)) <= field._value_roundoff(
        *field.start_point
    )
    path = surface.silhouette_envelope(plane).paths[0]
    assert path.segments[0].p0 == field.start_point
    assert path.segments[-1].p3 == (field.start_point[0], -field.start_point[1])
    for accuracy, tolerance in (
        (rocheometry.Reference(), 1e-8),
        (rocheometry.Fitted(), 3e-4),
    ):
        geometry = rocheometry.BinaryGeometry(qi, accuracy=accuracy)
        paths = tuple(
            geometry.component(k)
            .surface(potential=surface.potential)
            .silhouette_envelope(plane)
            .paths[0]
            for k in (1, 2)
        )
        outline = ContactProjection(*paths)
        assert max(outline.excess(p) for p in points) <= tolerance * pole


def test_cap_roundoff_is_accepted_but_a_genuine_positive_value_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = rocheometry.ComponentGeometry(
        1, accuracy=rocheometry.Reference()
    ).surface(fill_out_factor=0.5)
    field = projected(surface, pi / 3)
    angle, cap = 1.0, 0.4
    floor = field._value_roundoff(cap * cos(angle), cap * sin(angle))
    monkeypatch.setattr(field, "value", lambda x, t: floor / 2)
    assert field._limb_radius(angle, cap, cap) == cap
    monkeypatch.setattr(field, "value", lambda x, t: 2 * floor)
    with pytest.raises(RuntimeError, match="projected limb is not bracketed"):
        field._limb_radius(angle, cap, cap)


def test_start_polish_uses_the_limb_root_for_an_unresolved_crossing() -> None:
    surface = rocheometry.ComponentGeometry(
        1, accuracy=rocheometry.Reference()
    ).surface(fill_out_factor=0.5)
    geometry, potential = surface.component, surface.potential
    graph = geometry._computers
    exact = projected(surface, pi / 3)

    class DisplacedCrossing(ProjectedField):
        def _crossing(self) -> Point2D:  # type: ignore[missing-override-decorator]
            x, t = super()._crossing()
            return (x * (1 + 1e-8), t * (1 + 1e-8))

    corrected = DisplacedCrossing(
        geometry.field,
        _depth(geometry.field, potential),
        rocheometry.CentrelinePlane(pi / 3),
        graph.radius,
        geometry.l1(),
        graph.outer,
        graph.watershed,
    )
    assert corrected.start_point is not None and exact.start_point is not None
    assert (
        hypot(
            corrected.start_point[0] - exact.start_point[0],
            corrected.start_point[1] - exact.start_point[1],
        )
        < 1e-14
    )
    assert abs(corrected.value(*corrected.start_point)) <= corrected._value_roundoff(
        *corrected.start_point
    )
    assert corrected.radius(corrected.start) == corrected.start_radius


@pytest.mark.slow
@pytest.mark.parametrize("qi", QI)
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("inclination", INCLINATIONS)
@pytest.mark.parametrize("frame", ("binary", "centre_of_mass"))
def test_binary_limb_crossings_join_in_both_frames(
    qi: float, factor: float, inclination: float, frame: BinaryFrame
) -> None:
    binary = rocheometry.BinaryGeometry(
        qi, separation=3.7, frame=frame, accuracy=rocheometry.Reference()
    )
    light = binary.geometry2
    depth = light.surface(fill_out_factor=factor)._depth
    plane = rocheometry.CentrelinePlane(inclination)
    points = []
    for k in (1, 2):
        component = binary.component(k)
        crossing = crossing_at_depth(component.intrinsic_geometry, depth, plane)
        if k == 2:
            # The component-2 chart reverses the planar transverse axis.
            crossing = (crossing[0], -crossing[1])
        points.append(component._output_transform().apply(crossing))
    assert dist(*points) <= 1e-12 * binary.separation
