"""Saddle clustering consumes depth/positions, never rounded critical reports."""

from dataclasses import replace

import pytest

import rocheometry
from rocheometry._computers.surface_sample.angular import GaussLegendreSurfaceSampler
from rocheometry._geometry import Point3D
from rocheometry._roche.field import RocheField
from tests._helpers import RadiusDelegate


@pytest.mark.parametrize("qi", (1e-8, 1e-3, 1.0, 1e3, 1e8))
def test_cluster_inputs_ignore_rounded_critical_potential_reports(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    graph, field = geometry._computers, geometry.field

    def radius(field: RocheField, depth: float, direction: Point3D, /) -> float:
        return 1.0

    sampler = GaussLegendreSurfaceSampler(
        RadiusDelegate(radius),
        32,
        8,
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        watershed_computer=graph.watershed,
    )

    def l1(field: RocheField, /) -> rocheometry.CriticalPoint:
        point = graph.l1(field)
        return replace(
            point, potential=rocheometry.Potential.jacobi(point.potential.value + 0.125)
        )

    def outer(field: RocheField, /) -> rocheometry.CriticalPoint:
        point = graph.outer(field)
        return replace(
            point, potential=rocheometry.Potential.jacobi(point.potential.value - 0.125)
        )

    changed = replace(sampler, l1_computer=l1, outer_computer=outer)
    for surface in (
        geometry.surface(volume_radius_fill=0.8),
        geometry.lobe(),
        geometry.surface(fill_out_factor=0.5),
    ):
        assert changed(field, surface._depth) == sampler(field, surface._depth)
