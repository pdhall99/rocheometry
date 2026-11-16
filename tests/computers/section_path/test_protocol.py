"""Sections use declared operations, including with opaque custom providers."""

from dataclasses import dataclass, field, replace
from math import cos, pi, sin

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.section_path import SectionPathComputer
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._computers.surface_radius import BrentSurfaceRadiusComputer
from rocheometry._computers.watershed import WatershedComputer
from rocheometry._geometry import Point3D
from rocheometry._numerics.roots import brent_root
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired


@dataclass
class AnalyticWatershed:
    """Equal-mass sheet x=1/2, with no gravity-line or callable interface."""

    depths: list[float] = field(default_factory=list)

    def rim_at_depth(self, f: RocheField, depth: float, psi: float, /) -> Point3D:
        self.depths.append(depth)
        if depth == 0:
            return (0.5, 0, 0)
        u = f.potential(0.5) - depth
        c, s = cos(psi), sin(psi)
        rho = brent_root(lambda r: f.potential(0.5, r * c, r * s) - u, 0, 0.5)
        return (0.5, rho * c, rho * s)

    def rim(self, f: RocheField, depth: float, psi: float, /) -> Point3D:
        return self.rim_at_depth(f, depth, psi)

    def cut_height(self, f: RocheField, rho: float, psi: float, /) -> float:
        return 0.5

    def sheet_point_on_ray(self, f: RocheField, d: Point3D, /) -> Point3D:
        return (0.5, 0.5 * d[1] / d[0], 0.5 * d[2] / d[0])


@pytest.mark.parametrize("fitted", (False, True))
@pytest.mark.parametrize(
    "plane",
    (
        rocheometry.ORBITAL_PLANE,
        rocheometry.CentrelinePlane(pi / 3),
        rocheometry.AxisNormalPlane("y", 0.01),
        rocheometry.AxisNormalPlane("z", 0.01),
    ),
)
def test_noncallable_custom_watershed_drives_sections(
    fitted: bool, plane: rocheometry.SectionPlane
) -> None:
    base = Reference()._computers() if not fitted else Fitted()._computers()
    custom = AnalyticWatershed()
    watershed: WatershedComputer = custom
    assert not callable(watershed)
    radius = BrentSurfaceRadiusComputer(
        l1_computer=base.l1, outer_computer=base.outer, watershed_computer=watershed
    )
    traced = TracedSectionPathComputer(
        radius_computer=radius,
        l1_computer=base.l1,
        outer_computer=base.outer,
        watershed_computer=watershed,
        tolerance=1e-10 if not fitted else 1e-5,
    )
    section: SectionPathComputer = traced
    if fitted:
        section = PlaneRoutedSectionPathComputer(
            centreline=GaugeFixedSectionPathComputer(
                radius_computer=radius,
                l1_computer=base.l1,
                outer_computer=base.outer,
                watershed_computer=watershed,
            ),
            section=traced,
            l1_computer=base.l1,
            outer_computer=base.outer,
        )
    graph = replace(base, watershed=watershed, radius=radius, section=section)
    surface = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(graph)).surface(
        fill_out_factor=0.5
    )
    outline = surface.section(plane)
    envelope = surface.section_envelope(plane)
    assert len(outline) == len(envelope) == 1
    assert all(s in outline.paths[0].segments for s in envelope.paths[0].segments)
    assert envelope.paths[0].segments[0].p0[0] == pytest.approx(0.5, abs=1e-14)
    assert custom.depths
    if isinstance(plane, rocheometry.CentrelinePlane):
        assert any(
            0 < d < base.l1(RocheField(1)).potential.value - surface.potential.value
            for d in custom.depths
        )
