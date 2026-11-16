"""Computer graphs share checked results while remaining independent of fields."""

from dataclasses import replace

import pytest

import rocheometry
from rocheometry import Fitted
from rocheometry._computers.composition import Memoized
from rocheometry._computers.defaults import default_l1_computer
from rocheometry._computers.lobe_volume.from_surface import (
    LobeVolumeFromSurfaceComputer,
)
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._roche.field import RocheField
from rocheometry._roche.volume import sphere_volume
from tests._helpers import (
    CountedL1,
    RadiusDelegate,
    _Wired,
    computer_graph,
    numerical_graph,
    single_path,
)


def test_memoizer_evicts_least_recently_used_results_and_can_clear() -> None:
    calls: list[float] = []

    def computer(field: RocheField, /) -> float:
        calls.append(field.qi)
        return field.qi

    wrapped = Memoized(computer, maxsize=2)
    for qi in (1.0, 2.0, 1.0, 3.0, 2.0):
        assert wrapped(RocheField(qi)) == qi
    assert calls == [1.0, 2.0, 3.0, 2.0]
    assert wrapped.cache_info() == (1, 4, 2, 2)
    wrapped.cache_clear()
    assert wrapped.cache_info() == (0, 0, 2, 0)


@pytest.mark.parametrize("maxsize", [0, None])
def test_memoizer_supports_disabled_and_unbounded_caches(maxsize: int | None) -> None:
    wrapped = Memoized(lambda field: field.qi, maxsize=maxsize)
    wrapped(RocheField(1.0))
    wrapped(RocheField(1.0))
    assert wrapped.cache_info().currsize == (0 if maxsize == 0 else 1)


@pytest.mark.parametrize(
    ("maxsize", "error"), [(-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_memoizer_rejects_invalid_capacity(
    maxsize: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="maxsize"):
        Memoized(default_l1_computer(), maxsize=maxsize)  # type: ignore[arg-type]


def test_equal_values_share_results_without_canonical_geometry_identity() -> None:
    l1 = CountedL1()
    computers = computer_graph(l1_computer=l1)
    first = rocheometry.ComponentGeometry(0.4, accuracy=_Wired(computers))
    second = rocheometry.ComponentGeometry(0.4, accuracy=_Wired(computers))
    assert first == second and first is not second
    assert hash(first) == hash(second)
    assert first.l1() is second.l1()
    assert l1.calls == 1
    one = first.surface(volume_radius_fill=0.8)
    two = second.surface(volume_radius_fill=0.8)
    assert one == two and one is not two
    assert one.potential == two.potential
    assert l1.calls == 1


def test_distinct_graphs_define_distinct_geometry_values() -> None:
    first = rocheometry.ComponentGeometry(
        0.4, accuracy=_Wired(numerical_graph(polar_order=8, azimuth_order=6))
    )
    second = rocheometry.ComponentGeometry(
        0.4, accuracy=_Wired(numerical_graph(polar_order=8, azimuth_order=6))
    )
    assert first != second
    assert first.l1() == second.l1()


def test_one_binary_configuration_serves_both_fields_and_other_binaries() -> None:
    l1 = CountedL1()
    computers = computer_graph(l1_computer=l1)
    first = rocheometry.BinaryGeometry(0.4, accuracy=_Wired(computers))
    second = rocheometry.BinaryGeometry(
        0.4, separation=3.0, frame="centre_of_mass", accuracy=_Wired(computers)
    )
    for binary in (first, second):
        assert binary._computers is computers
        assert binary.geometry1._computers is binary.geometry2._computers is computers
        binary.component(1).lobe().named_radius("pole")
        binary.component(2).lobe().named_radius("pole")
    assert l1.calls == 2
    # A strategy taken from A remains correct when passed B's field.
    b = first.geometry2
    actual = first.geometry1._computers.radius(b.field, 0.0, (0.0, 0.0, 1.0))
    assert actual == b.lobe().named_radius("pole")


def test_replacing_binary_computers_rebuilds_both_components() -> None:
    original = rocheometry.BinaryGeometry(0.4)
    computers = numerical_graph(polar_order=8, azimuth_order=6)
    changed = replace(original, accuracy=_Wired(computers))

    assert changed != original
    assert changed == rocheometry.BinaryGeometry(0.4, accuracy=_Wired(computers))
    assert changed.geometry1._computers is changed.geometry2._computers is computers
    assert changed.geometry1.qi == original.geometry1.qi
    assert changed.geometry2.qi == original.geometry2.qi
    assert original._computers is Fitted()._computers()


def test_normalization_is_shared_by_potential_inversion_and_selected_volumes() -> None:
    calls = 0

    def volume(field: RocheField, level: float, /) -> float:
        nonlocal calls
        calls += 1
        return (default_l1_computer()(field).potential.value - level) ** -3

    computers = computer_graph(volume_computer=volume)
    geometry = rocheometry.ComponentGeometry(1.0, accuracy=_Wired(computers))
    lobe_volume = computers.selection.fast_lobe_volume(geometry.field)
    assert calls == 1
    for fill in (0.2, 0.6, 0.8):
        surface = geometry.surface(volume_radius_fill=fill)
        assert surface.volume() == pytest.approx(fill**3 * lobe_volume, rel=6.0e-8)
        potential = surface.potential
        assert volume(geometry.field, surface._depth) == pytest.approx(
            surface.volume(), rel=6.0e-8
        )
        before = calls
        assert geometry.surface(volume_radius_fill=fill).potential == potential
        assert calls == before
    assert computers.selection.fast_lobe_volume(geometry.field) is lobe_volume


def test_integrated_lobe_computer_uses_its_injected_field_agnostic_dependencies() -> (
    None
):
    fields: list[RocheField] = []
    points: list[float] = []
    l1 = Memoized(default_l1_computer())

    def volume(field: RocheField, potential: float, /) -> float:
        fields.append(field)
        points.append(potential)
        return sphere_volume(field.qi)

    computer = LobeVolumeFromSurfaceComputer(volume_computer=volume, l1_computer=l1)
    for qi in (0.4, 2.5):
        field = RocheField(qi)
        assert computer(field) == pytest.approx(sphere_volume(qi))
        assert fields[-1] is field
        assert points[-1] == 0.0


def test_custom_detached_methods_do_not_force_lobe_integration() -> None:
    def radius(
        field: RocheField, level: float, ray: tuple[float, float, float], /
    ) -> float:
        return field.qi - level

    def potential(field: RocheField, fill: float, /) -> float:
        return -(3.0 + field.qi + fill)

    computers = computer_graph(
        radius_computer=RadiusDelegate(radius), depth_computer=potential
    )
    surface = rocheometry.ComponentGeometry(0.4, accuracy=_Wired(computers)).surface(
        volume_radius_fill=0.8
    )
    for name in ("point", "back", "side", "pole"):
        assert surface.named_radius(name) == pytest.approx(4.6)


@pytest.mark.parametrize("fill", [0.8, 1.0])
def test_tracer_uses_the_surface_level_and_preserves_the_supplied_field(
    fill: float,
) -> None:
    graph = numerical_graph(polar_order=8, azimuth_order=6)
    fields: list[RocheField] = []
    levels: list[float] = []
    supplied = RocheField(0.4)
    expected = 0.0 if fill == 1.0 else graph.selection.depth(supplied, fill)

    def radius(
        field: RocheField, level: float, direction: tuple[float, float, float], /
    ) -> float:
        fields.append(field)
        levels.append(level)
        return graph.radius(field, level, direction)

    computer = TracedSectionPathComputer(
        radius_computer=RadiusDelegate(radius), l1_computer=graph.l1
    )
    points = single_path(
        computer(supplied, expected, rocheometry.ORBITAL_PLANE)
    ).sample(9)
    assert len(points) == 9
    assert points[0] == points[-1]
    assert fields and all(field is supplied for field in fields)
    assert all(level is expected for level in levels)
