"""Accuracy, inversion and structural endpoints of the wide potential map."""

from math import isfinite
from typing import cast

import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.packaged import load_packaged_surface_potential_map
from rocheometry._computers.surface_potential.fitted.computer import (
    FittedSurfacePotentialComputer,
    SurfacePotentialMap,
    fill_times_t,
)
from rocheometry._computers.surface_selection import FittedSelection
from rocheometry._roche.field import RocheField


@pytest.fixture(scope="module")
def forward() -> FittedSurfacePotentialComputer:
    graph = Fitted()._computers()
    selection = cast(FittedSelection, graph.selection)
    return FittedSurfacePotentialComputer(
        l1_computer=graph.l1,
        lobe_volume_radius_computer=selection.lobe_volume_radius_computer,
        model=load_packaged_surface_potential_map(),
    )


def test_dense_monotonicity_200_mass_ratios_by_400_fills(
    forward: FittedSurfacePotentialComputer,
) -> None:
    for i in range(200):
        field = RocheField(10 ** (-8 + 16 * i / 199))
        previous = -float("inf")
        for j in range(1, 401):
            potential = forward(field, j / 400)
            assert isfinite(potential)
            assert potential > previous, (field.qi, j / 400)
            previous = potential


@pytest.mark.parametrize("qi", [1e-8, 1e-6, 1e-4, 0.03, 1.0, 100.0, 1e4, 1e8])
@pytest.mark.parametrize("fill", [1e-8, 0.01, 0.1, 0.5, 0.8, 0.99, 0.999999, 1.0])
def test_forward_selection_reports_independently_measured_fill(
    forward: FittedSurfacePotentialComputer, qi: float, fill: float
) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    depth = forward(geometry.field, fill)
    measured = geometry._surface_at_depth(depth).volume_radius_fill()
    assert abs(measured - fill) <= 2e-6


@pytest.mark.parametrize("qi", [1e-8, 1e-4, 0.3, 1, 1e4, 1e8])
def test_exact_endpoints_and_small_fill_asymptote(
    forward: FittedSurfacePotentialComputer, qi: float
) -> None:
    field = RocheField(qi)
    assert forward(field, 1.0) == 0.0
    assert rocheometry.ComponentGeometry(qi).lobe().volume_radius_fill() == 1.0
    fill = 1e-8
    m = qi / (1 + qi)
    u0 = 1 / (1 + qi) + 0.5 / (1 + qi) ** 2
    expected = u0 + m / (fill * forward.lobe_volume_radius_computer(field))
    assert field._level_value(
        forward.l1_computer(field).position[0], forward(field, fill)
    ) == pytest.approx(expected, rel=1e-14)


def test_model_json_and_domain(forward: FittedSurfacePotentialComputer) -> None:
    model = forward.model
    assert model is not None
    assert SurfacePotentialMap.from_json(model.to_json()) == model
    assert model.coordinate(1e-8) == -1
    assert model.coordinate(1e8) == 1
    for qi in (1e-9, 1e9):
        with pytest.raises(ValueError, match="potential map requires"):
            model.coefficients_at(qi)
        with pytest.raises(ValueError, match="outside fitted"):
            forward(RocheField(qi), 0.8)


@pytest.mark.parametrize("rows", [(), ((),), ((1.0,), (1.0, 2.0)), ((float("nan"),),)])
def test_model_rejects_invalid_tensors(rows: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(ValueError):
        SurfacePotentialMap(rows)


def test_runtime_tensor_and_analytic_derivative_match_numpy() -> None:
    matrix = ((0.2, 0.1, -0.04), (-0.1, 0.02, 0.01), (0.03, -0.01, 0.005))
    model = SurfacePotentialMap(matrix)
    for qi in (1e-8, 0.3, 1.0, 1e8):
        coefficients = model.coefficients_at(qi)
        for fill in (0.01, 0.5, 0.99):
            z, s = model.coordinate(qi), np.sqrt(1 - fill)
            expected_r = C.chebval2d(z, 2 * s - 1, np.asarray(matrix))
            expected = fill * (
                1 + 0.1 * (1 - fill**3) + (1 - fill) * fill**3 * expected_r
            )
            actual, derivative = fill_times_t(fill, 1.1, coefficients)
            assert actual == pytest.approx(expected, rel=2e-15)
            h = 1e-6
            difference = (
                fill_times_t(fill + h, 1.1, coefficients)[0]
                - fill_times_t(fill - h, 1.1, coefficients)[0]
            ) / (2 * h)
            assert derivative == pytest.approx(difference, rel=1e-8)
