"""Accuracy and integration budgets of the default Brent surface strategies."""

import pytest

from rocheometry._computers.surface_potential import BrentSurfacePotentialComputer
from rocheometry._computers.surface_volume import SampledVolumeComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import volume_radius_from_volume
from tests._helpers import numerical_graph

QI_GRID = (1.0e-4, 0.01, 0.3, 1.0, 3.0, 100.0, 1.0e4)


@pytest.mark.parametrize("qi", QI_GRID)
@pytest.mark.parametrize(
    "fill", (0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999, 0.99999, 0.999999)
)
def test_brent_potential_meets_fill_tolerance_with_at_most_twelve_integrals(
    qi: float, fill: float
) -> None:
    graph = numerical_graph()
    field = RocheField(qi)
    lobe_volume = graph.selection.fast_lobe_volume(field)
    volume = SampledVolumeComputer(sample=graph.sample)
    calls: list[float] = []

    def count(actual: RocheField, level: float, /) -> float:
        assert actual is field
        calls.append(level)
        return volume(actual, level)

    computer = BrentSurfacePotentialComputer(
        volume_computer=count,
        l1_computer=graph.l1,
        lobe_volume_computer=graph.selection.fast_lobe_volume,
    )
    selected = computer(field, fill)
    assert len(calls) <= 12
    assert selected in calls
    assert 0.0 not in calls
    recovered = volume_radius_from_volume(volume(field, selected))
    target = volume_radius_from_volume(lobe_volume) * fill
    assert abs(recovered / target - 1.0) <= computer.volume_radius_fill_tolerance


def test_brent_critical_potential_requires_no_volume_integrals() -> None:
    graph = numerical_graph()
    field = RocheField(0.3)

    def unexpected(actual: RocheField, level: float, /) -> float:
        raise AssertionError("critical selection must not integrate")

    computer = BrentSurfacePotentialComputer(
        volume_computer=unexpected,
        l1_computer=graph.l1,
        lobe_volume_computer=graph.selection.fast_lobe_volume,
    )
    assert computer(field, 1.0) == 0.0


def test_brent_potential_reports_an_exhausted_iteration_budget() -> None:
    graph = numerical_graph()
    field = RocheField(0.3)
    computer = BrentSurfacePotentialComputer(
        volume_computer=SampledVolumeComputer(sample=graph.sample),
        l1_computer=graph.l1,
        lobe_volume_computer=graph.selection.fast_lobe_volume,
        max_iterations=1,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        computer(field, 0.8)
