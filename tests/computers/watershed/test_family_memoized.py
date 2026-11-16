"""Watershed sharing uses graph cache semantics under concurrent access."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass, replace
from math import atan2, hypot, pi
from random import Random
from typing import cast

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.composition import ComputerSet, Memoized
from rocheometry._computers.watershed import TracedWatershedComputer
from rocheometry._computers.watershed.family import _BuildCutSection
from rocheometry._roche.field import RocheField
from tests._helpers import _Wired


@pytest.mark.parametrize("graph", (Reference()._computers(), Fitted()._computers()))
def test_watershed_operations_are_explicit_memoized_nodes(
    graph: ComputerSet,
) -> None:
    w = cast(TracedWatershedComputer, graph.watershed)
    for node in (
        w.line,
        w.section_computer,
        w.cut_section_computer,
        w.depth_computer,
        w.rim_computer,
    ):
        assert isinstance(node, Memoized)
        assert node.cache_info().maxsize == 128
    assert w.family.cache_info().maxsize == 4
    assert w.cut_height_computer.section is w.section_computer
    assert w.cut_height_computer.cut_section is w.cut_section_computer
    assert w.ray_computer.cut_section is w.cut_section_computer.computer
    assert isinstance(w.ray_computer.cut_section, _BuildCutSection)


@pytest.mark.parametrize("qi", (1e-3, 1e3))
def test_ray_iterations_preserve_cached_cut_sections(qi: float) -> None:
    graph = Reference()._computers()
    w = TracedWatershedComputer(
        l1_computer=graph.l1, outer_computer=graph.outer, maxsize=2
    )
    field = RocheField(qi)
    point = w.rim(
        field,
        rocheometry.ComponentGeometry(qi).surface(fill_out_factor=0.5)._depth,
        0.4,
    )
    rho = hypot(*point[1:])
    # Seed the public cut cache, then vary all Newton trial radii. Neither its
    # entries nor its bookkeeping should change, even when queried repeatedly.
    w.cut_height(field, rho, 0.4)
    cached = w.cut_section_computer(field, rho)
    before = w.cut_section_computer.cache_info()
    for factor in (0.3, 0.6, 0.9, 1.0):
        direction = (point[0], factor * point[1], factor * point[2])
        ray = w.sheet_point_on_ray(field, direction)
        assert w.sheet_point_on_ray(field, direction) == ray
    assert w.cut_section_computer.cache_info() == before
    assert w.cut_section_computer(field, rho) is cached


@pytest.mark.parametrize("qi", (1e-3, 1.0, 1e3))
@pytest.mark.parametrize("reference", (False, True))
def test_twenty_cut_rays_are_bit_identical_without_memoized_newton_trials(
    qi: float, reference: bool
) -> None:
    graph = Reference()._computers() if reference else Fitted()._computers()
    w = TracedWatershedComputer(
        l1_computer=graph.l1,
        outer_computer=graph.outer,
        tolerance=cast(TracedWatershedComputer, graph.watershed).tolerance,
        maxsize=2,
    )
    # Deliberately restore the memoized dependency only in the comparison path.
    cached_ray = replace(
        w.ray_computer,
        cut_section=cast(_BuildCutSection, w.cut_section_computer),
    )
    geometry = rocheometry.ComponentGeometry(qi, accuracy=_Wired(graph))
    rng = Random(20261021)
    for _ in range(20):
        rim = w.rim(
            geometry.field,
            geometry.surface(fill_out_factor=0.5)._depth,
            rng.uniform(0, 2 * pi),
        )
        fraction = rng.uniform(0.05, 0.95)
        direction = (rim[0], fraction * rim[1], fraction * rim[2])
        before = w.cut_section_computer.cache_info()
        actual = w.sheet_point_on_ray(geometry.field, direction)
        assert w.cut_section_computer.cache_info() == before
        expected = cached_ray(geometry.field, direction)
        assert tuple(v.hex() for v in actual) == tuple(v.hex() for v in expected)
    if qi != 1.0:
        assert w.cut_section_computer.cache_info().misses > 0


def test_concurrent_queries_and_eviction_use_lru_cache() -> None:
    graph = Reference()._computers()
    w = TracedWatershedComputer(
        l1_computer=graph.l1, outer_computer=graph.outer, maxsize=2
    )
    field = RocheField(1.0)

    def query(i: int) -> tuple[float, float, float]:
        depth = 1e-3 + (i % 3) * 1e-4
        psi = (i % 7) * 0.2
        p = w.rim_at_depth(field, depth, psi)
        assert p[0] == 0.5
        assert field.potential(*p) == pytest.approx(
            graph.l1(field).potential.value - depth, abs=2e-14
        )
        assert atan2(p[2], p[1]) == pytest.approx(psi, abs=2e-14)
        return p

    with ThreadPoolExecutor(max_workers=8) as pool:
        points = tuple(pool.map(query, range(64)))
    assert all(hypot(*p) > 0 for p in points)
    assert w.line.cache_info().currsize <= 2
    assert w.depth_computer.cache_info().currsize <= 2
    w.depth_computer.cache_clear()
    p = w.rim_at_depth(field, 0.001, 0.3)
    assert w.rim_at_depth(field, 0.001, 0.3) is p
    assert w.depth_computer.cache_info().hits == 1


@pytest.mark.parametrize("qi", (0.05, 1.0, 20.0, 1e6))
def test_zero_depth_and_critical_level_are_exact(qi: float) -> None:
    g = Reference()._computers()
    f = RocheField(qi)
    for psi in (0.0, 0.3, 1.0):
        assert g.watershed.rim_at_depth(f, 0.0, psi) == g.l1(f).position
        assert g.watershed.rim(f, 0.0, psi) == g.l1(f).position


def test_watershed_graph_has_no_mutable_mapping_caches() -> None:
    graph = Reference()._computers()
    seen: set[int] = set()

    def visit(node: object) -> None:
        if id(node) in seen or isinstance(node, type):
            return
        seen.add(id(node))
        assert not isinstance(node, dict)
        if is_dataclass(node):
            for item in fields(node):
                visit(getattr(node, item.name))

    visit(graph.watershed)
