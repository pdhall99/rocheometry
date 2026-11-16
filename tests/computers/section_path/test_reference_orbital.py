import math
from collections.abc import Callable
from typing import cast

import pytest

from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.section_path import (
    SectionReference,
    reference_component_section,
)
from rocheometry._roche import RocheField
from rocheometry._roche.mass import component_mass_ratio
from rocheometry._types import ComponentId


def _critical_section(
    q: float,
    component_id: ComponentId,
    *,
    num_upper_points: int = 257,
) -> SectionReference:
    return reference_component_section(
        q,
        component_id,
        1.0,
        num_upper_points=num_upper_points,
        polar_order=4,
        azimuth_order=4,
    )


def test_closed_section_has_exact_reflection_symmetry() -> None:
    section = _critical_section(0.37, 1, num_upper_points=33)

    assert section.lower == tuple((x, -y) for x, y in reversed(section.upper))
    assert section.closed_points[0] == section.closed_points[-1]
    assert section.closed_points.count(section.landmarks.inner) == 2
    assert section.closed_points.count(section.landmarks.back) == 1


def test_critical_section_stays_on_inner_branch_near_l1() -> None:
    section = _critical_section(1.0e-3, 1, num_upper_points=1025)
    l1_distance = section.landmarks.inner[0]

    assert math.hypot(*section.upper[1]) < l1_distance
    assert section.upper[1][0] < l1_distance


def test_l1_is_a_cusp_after_reflection() -> None:
    section = _critical_section(1.0, 1, num_upper_points=257)
    l1 = section.landmarks.inner
    upper_neighbour = section.upper[1]
    lower_neighbour = section.closed_points[-2]

    outgoing = (
        upper_neighbour[0] - l1[0],
        upper_neighbour[1] - l1[1],
    )
    incoming = (
        l1[0] - lower_neighbour[0],
        l1[1] - lower_neighbour[1],
    )

    assert outgoing[0] < 0.0
    assert incoming[0] > 0.0
    assert outgoing[1] > 0.0
    assert incoming[1] > 0.0


@pytest.mark.parametrize("q", [1.0e-4, 0.01, 1.0, 100.0, 1.0e4])
@pytest.mark.parametrize("component_id", [1, 2])
def test_top_is_exact_maximum_y_landmark(
    q: float,
    component_id: ComponentId,
    reference_top_check: Callable[[SectionReference], None],
) -> None:
    section = _critical_section(q, component_id, num_upper_points=65)
    reference_top_check(section)
    top_y = section.landmarks.top[1]
    assert top_y >= max(y for _, y in section.upper) - 1.0e-13


@pytest.mark.parametrize("q", [1.0e-4, 0.01, 1.0, 100.0, 1.0e4])
@pytest.mark.parametrize("component_id", [1, 2])
def test_backside_has_vertical_tangent(q: float, component_id: ComponentId) -> None:
    section = _critical_section(q, component_id, num_upper_points=33)
    field = RocheField(component_mass_ratio(q, component_id))
    back_x, back_y = section.landmarks.back
    gradient_x, gradient_y, _ = field.gradient(back_x, back_y)

    assert back_y == 0.0
    assert gradient_y == 0.0
    assert gradient_x > 0.0


def test_component_exchange_preserves_component_centred_shape() -> None:
    component_1 = _critical_section(0.125, 1, num_upper_points=65)
    component_2_after_exchange = _critical_section(8.0, 2, num_upper_points=65)

    assert component_1.upper == component_2_after_exchange.upper
    assert component_1.landmarks == component_2_after_exchange.landmarks


def test_dense_log_q_grid_stays_on_critical_equipotential() -> None:
    for index in range(33):
        q = 10.0 ** (-4.0 + 8.0 * index / 32.0)
        for component_id in (1, 2):
            section = _critical_section(q, component_id, num_upper_points=17)
            field = RocheField(component_mass_ratio(q, component_id))
            l1 = default_l1_computer()(field)
            critical = l1.potential.value

            for x, y in section.upper:
                actual = field.potential(x, y)
                assert actual == pytest.approx(critical, rel=3.0e-13, abs=3.0e-13)


@pytest.mark.parametrize("component_id", [0, 3])
def test_critical_reference_rejects_invalid_component(component_id: int) -> None:
    with pytest.raises(ValueError, match="component_id: expected 1 or 2"):
        reference_component_section(
            1.0,
            cast(ComponentId, component_id),
            1.0,
            polar_order=4,
            azimuth_order=4,
        )


def test_critical_reference_rejects_bool_component() -> None:
    with pytest.raises(TypeError, match="component_id: expected int, got bool"):
        reference_component_section(
            1.0,
            cast(ComponentId, True),
            1.0,
            polar_order=4,
            azimuth_order=4,
        )


@pytest.mark.parametrize("num_upper_points", [1, 0, -1])
def test_critical_reference_rejects_invalid_sample_count(
    num_upper_points: int,
) -> None:
    with pytest.raises(ValueError, match="num_upper_points"):
        reference_component_section(
            1.0,
            1,
            1.0,
            num_upper_points=num_upper_points,
            polar_order=4,
            azimuth_order=4,
        )


@pytest.mark.parametrize("num_upper_points", [True, 2.5])
def test_critical_reference_rejects_wrong_sample_count_type(
    num_upper_points: object,
) -> None:
    with pytest.raises(TypeError, match="num_upper_points"):
        reference_component_section(
            1.0,
            1,
            1.0,
            num_upper_points=cast(int, num_upper_points),
            polar_order=4,
            azimuth_order=4,
        )
