"""Regime classification is independent of the selected binary component."""

from math import inf, nextafter

import pytest

import rocheometry
from rocheometry._roche import RocheField
from tests._helpers import _opening_level as outer_level
from tests._helpers import _Wired, computer_graph

QI = (1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6)


@pytest.mark.parametrize("qi", QI)
def test_geometry_regime_boundaries(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    u1 = geometry.l1().potential.value
    u2 = outer_level(geometry.field, geometry._computers.outer)
    delta = u1 - u2
    for value, expected in (
        (u1, rocheometry.LevelRegime.CRITICAL),
        (nextafter(u1, -inf), rocheometry.LevelRegime.CONTACT),
        (u2, rocheometry.LevelRegime.OPEN),
        (nextafter(u2, -inf), rocheometry.LevelRegime.OPEN),
        (u1 + 1e-6 * delta, rocheometry.LevelRegime.DETACHED),
    ):
        assert geometry.level_regime(rocheometry.Potential.jacobi(value)) is expected


@pytest.mark.parametrize("q", QI)
def test_binary_regime_symmetry_in_every_convention(q: float) -> None:
    binary = rocheometry.BinaryGeometry(q)
    first, second = binary.component(1), binary.component(2)
    u1 = max(first.l1().potential.value, second.l1().potential.value)
    u2 = max(first.outer().potential.value, second.outer().potential.value)
    delta = u1 - u2
    for value in (
        u1 + 1e-6 * delta,
        first.l1().potential.value,
        second.l1().potential.value,
        u1 - 0.5 * delta,
        u2,
        nextafter(u2, -inf),
        u2 - 0.001 * delta,
    ):
        canonical = rocheometry.Potential.jacobi(value)
        expected = first.level_regime(canonical)
        assert second.level_regime(canonical) is expected
        for component_id in (1, 2):
            kopal = binary.convert_potential(
                canonical,
                convention=rocheometry.PotentialConvention.KOPAL,
                component_id=component_id,
            )
            # Conversion of exactly the outer threshold can round by one ulp;
            # compare both views at the actual converted canonical level.
            assert first.level_regime(kopal) is second.level_regime(kopal)
            # Materializing Jacobi rounds a different public level. Classify
            # Kopal directly at the depth-conversion boundary instead.
            from rocheometry._levels import _levels
            from rocheometry._roche.regime import level_regime

            levels = _levels(binary.geometry1.field, binary._computers)
            assert first.level_regime(kopal) is level_regime(
                levels.depth(kopal), levels.admissibility_gap
            )


def test_classification_uses_both_components_of_the_supplied_graph() -> None:
    calls: list[tuple[str, float]] = []

    def l1(field: RocheField, /) -> rocheometry.CriticalPoint:
        calls.append(("l1", field.qi))
        return rocheometry.Fitted()._computers().l1(field)

    def outer(field: RocheField, /) -> rocheometry.CriticalPoint:
        calls.append(("outer", field.qi))
        return rocheometry.Fitted()._computers().outer(field)

    graph = computer_graph(l1_computer=l1, outer_computer=outer)
    geometry = rocheometry.ComponentGeometry(0.25, accuracy=_Wired(graph))
    assert (
        geometry.level_regime(rocheometry.Potential.jacobi(1.0))
        is rocheometry.LevelRegime.OPEN
    )
    assert set(calls) == {("l1", 0.25), ("outer", 0.25), ("outer", 4.0)}


def test_intrinsic_regime_requires_canonical_potential() -> None:
    kopal = rocheometry.Potential.kopal(4.0, component_id=1)
    with pytest.raises(ValueError, match="expected a Jacobi potential"):
        rocheometry.ComponentGeometry(0.3).level_regime(kopal)
    with pytest.raises(ValueError, match="expected a Jacobi potential"):
        rocheometry.ComponentGeometry(0.3).surface(potential=kopal)
