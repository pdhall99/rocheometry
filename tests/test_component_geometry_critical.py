import pytest

import rocheometry
from rocheometry._computers.critical_point import (
    CriticalPointComputer,
)
from rocheometry._frames import transform
from rocheometry._roche import RocheField
from tests._helpers import _Wired, computer_graph


class _CountedPointComputer:
    def __init__(
        self,
        name: str,
        computer: CriticalPointComputer,
        calls: list[str],
    ) -> None:
        self.name = name
        self.computer = computer
        self.calls = calls

    def __call__(self, field: RocheField) -> rocheometry.CriticalPoint:
        self.calls.append(self.name)
        return self.computer(field)


def test_l1_and_outer_points_are_solved_independently() -> None:
    calls: list[str] = []
    base = rocheometry.ComponentGeometry(0.4)
    geometry = rocheometry.ComponentGeometry(
        0.4,
        accuracy=_Wired(
            computer_graph(
                l1_computer=_CountedPointComputer("l1", base._computers.l1, calls),
                outer_computer=_CountedPointComputer(
                    "outer", base._computers.outer, calls
                ),
            )
        ),
    )

    l1 = geometry.l1()
    assert isinstance(l1, rocheometry.CriticalPoint)
    assert calls == ["l1"]

    assert geometry.l1() == l1
    assert calls == ["l1"]

    outer = geometry.outer()
    assert calls == ["l1", "outer"]

    assert geometry.outer() == outer
    assert calls == ["l1", "outer"]


def test_multiple_binary_views_share_memoized_critical_points() -> None:
    calls: list[str] = []
    base = rocheometry.ComponentGeometry(0.4)
    computers = computer_graph(
        l1_computer=_CountedPointComputer("l1", base._computers.l1, calls)
    )
    first = rocheometry.BinaryGeometry(
        0.4, separation=2.0, frame="binary", accuracy=_Wired(computers)
    )
    second = rocheometry.BinaryGeometry(
        0.4, separation=5.0, frame="centre_of_mass", accuracy=_Wired(computers)
    )

    _ = first.l1()
    assert calls == ["l1"]

    _ = second.l1()
    assert calls == ["l1"]


def test_lobe_geometry_reuses_configured_l1() -> None:
    calls: list[str] = []
    base = rocheometry.ComponentGeometry(1.0)
    geometry = rocheometry.ComponentGeometry(
        1.0,
        accuracy=_Wired(
            computer_graph(
                l1_computer=_CountedPointComputer("l1", base._computers.l1, calls)
            )
        ),
    )

    assert geometry.lobe().volume_radius() > 0.0
    assert calls == ["l1"]

    assert geometry.l1().position[0] == pytest.approx(0.5)
    assert calls == ["l1"]


def test_equal_mass_component_critical_points() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    l1 = geometry.l1()
    outer = geometry.outer()

    assert l1.position == pytest.approx((0.5, 0.0, 0.0))
    assert l1.potential == rocheometry.Potential.jacobi(2.0)
    assert outer.position[0] < 0.0


@pytest.mark.parametrize("qi", [1.0e-4, 0.03, 0.4, 1.0, 2.5, 30.0, 1.0e4])
def test_each_component_critical_point_satisfies_stationary_field(qi: float) -> None:
    geometry = rocheometry.ComponentGeometry(qi)
    field = RocheField(qi)

    for point in (geometry.l1(), geometry.outer()):
        assert field.gradient(point.position[0])[0] == pytest.approx(
            0.0,
            abs=2.0e-9,
        )
        assert point.potential.value == pytest.approx(
            field.potential(point.position[0]),
            rel=2.0e-15,
            abs=2.0e-15,
        )


@pytest.mark.parametrize("q", [0.03, 0.4, 2.5, 30.0])
def test_l1_obeys_component_exchange(q: float) -> None:
    original = rocheometry.ComponentGeometry(q)
    swapped = rocheometry.ComponentGeometry(1.0 / q)

    assert original.l1().position[0] == pytest.approx(
        1.0 - swapped.l1().position[0],
        abs=2.0e-13,
    )


@pytest.mark.parametrize("q", [0.03, 0.4, 1.0, 2.5, 30.0])
def test_reciprocal_l1_points_are_the_same_physical_l1(q: float) -> None:
    binary = rocheometry.BinaryGeometry(q=q)
    first = binary.geometry1.l1()
    second = binary.geometry2.l1()
    second_in_binary = transform(
        second.position,
        source="component2",
        target="binary",
        q=q,
    )

    assert first.position == pytest.approx(second_in_binary, abs=2.0e-13)
    assert first.potential.value == pytest.approx(
        second.potential.value,
        abs=2.0e-13,
    )


def test_equal_mass_binary_outer_points_are_symmetric() -> None:
    binary = rocheometry.BinaryGeometry(q=1.0)
    outer1 = binary.component(1).outer()
    outer2 = binary.component(2).outer()

    assert outer1.position[0] < 0.0
    assert outer2.position[0] > 1.0
    assert outer1.position[0] == pytest.approx(
        1.0 - outer2.position[0],
        abs=2.0e-13,
    )
    assert outer1.potential.value == pytest.approx(
        outer2.potential.value,
        abs=2.0e-14,
    )


def test_l2_l3_follow_mass_order_and_equal_mass_tie() -> None:
    below = rocheometry.BinaryGeometry(q=0.4)
    above = rocheometry.BinaryGeometry(q=2.5)
    equal = rocheometry.BinaryGeometry(q=1.0)

    assert below.l2() == below.component(1).outer()
    assert below.l3() == below.component(2).outer()
    assert above.l2() == above.component(2).outer()
    assert above.l3() == above.component(1).outer()
    assert equal.l2() == equal.component(2).outer()
    assert equal.l3() == equal.component(1).outer()


def test_binary_view_transforms_and_scales_l1_position() -> None:
    q = 0.4
    separation = 3.5
    binary = rocheometry.BinaryGeometry(
        q=q,
        separation=separation,
        frame="centre_of_mass",
    )
    point = binary.l1()
    expected_normalized = transform(
        binary.geometry1.l1().position,
        source="component1",
        target="centre_of_mass",
        q=q,
    )

    assert point.potential == binary.geometry1.l1().potential
    assert point.position == pytest.approx(
        tuple(separation * value for value in expected_normalized)
    )


def test_binary_view_transforms_component_two_outer_point() -> None:
    q = 0.4
    binary = rocheometry.BinaryGeometry(q=q, frame="binary")
    point = binary.component(2).outer()
    expected = transform(
        binary.geometry2.outer().position,
        source="component2",
        target="binary",
        q=q,
    )

    assert point.position == pytest.approx(expected)
