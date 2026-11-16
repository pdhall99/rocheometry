"""Sampled integration preserves the original directional accumulation exactly."""

from typing import cast

import pytest

from rocheometry._computers.surface_sample import SurfaceSampler
from rocheometry._computers.surface_sample.angular import (
    _azimuth_quadrature,
    _detached_quadrature,
    _neck_width,
)
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.potential import Potential
from tests._helpers import _depth, numerical_graph


@pytest.mark.parametrize("qi", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("offset", [0.0, 0.05])
def test_sampled_volume_is_bit_identical_to_direct_node_radius_integration(
    qi: float, offset: float
) -> None:
    graph = numerical_graph()
    field = RocheField(qi)
    level = Potential.jacobi(graph.l1(field).potential.value * (1 + offset))
    total = 0.0
    # Nodes deliberately stop at adjacent floats; public radius queries retain
    # their tolerance stop. Keep the exact accumulation assertion on node roots.
    node_radius = graph.radius.node_radius
    neck_width = _neck_width(field, _depth(field, level), graph.l1(field))
    azimuths = _azimuth_quadrature(16, 1 - graph.l1(field).position[0])
    for direction, weight in _detached_quadrature(
        32,
        azimuths,
        neck_width,
        _neck_width(field, _depth(field, level), graph.outer(field), graph.l1(field)),
        (1 - graph.l1(field).position[0]) / graph.l1(field).position[0],
    ):
        radius = node_radius(field, _depth(field, level), direction)
        total += weight * radius**3
    assert (
        SampledVolumeComputer(sample=graph.sample)(field, _depth(field, level)).hex()
        == (total / 3.0).hex()
    )


def test_sampled_volume_requires_a_sampler() -> None:
    with pytest.raises(TypeError, match="sample: expected a callable"):
        SampledVolumeComputer(sample=cast(SurfaceSampler, None))
