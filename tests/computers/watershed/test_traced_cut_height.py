"""Height inversion is consistent with level-selected watershed rims."""

from math import hypot, pi

import pytest

import rocheometry
from rocheometry import Reference
from tests._helpers import _Wired


@pytest.mark.parametrize("qi", (1e-6, 1e-3, 0.05, 1.0, 20.0, 1e3, 1e6))
@pytest.mark.parametrize("psi", (0.0, pi / 6, pi / 3, pi / 2))
def test_cut_height_inverts_rim(qi: float, psi: float) -> None:
    graph = Reference()._computers()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    assert (
        graph.watershed.cut_height(geometry.field, 0.0, psi)
        == geometry.l1().position[0]
    )
    surface = geometry.surface(fill_out_factor=0.5)
    rim = graph.watershed.rim(geometry.field, surface._depth, psi)
    height = graph.watershed.cut_height(geometry.field, hypot(*rim[1:]), psi)
    assert abs(height - rim[0]) <= 1e-11 * geometry.l1().position[0]
