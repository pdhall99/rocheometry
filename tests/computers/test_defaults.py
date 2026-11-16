"""Accuracy values resolve shared private graphs with bounded cache policies."""

from dataclasses import fields, is_dataclass
from typing import cast

import pytest

import rocheometry
from rocheometry._computers.composition import Memoized
from rocheometry._computers.defaults import _cached_reference
from rocheometry._computers.section_path.gauge_fixed import (
    GaugeFixedSectionPathComputer,
)
from rocheometry._computers.section_path.routed import PlaneRoutedSectionPathComputer
from rocheometry._computers.section_path.traced import TracedSectionPathComputer
from rocheometry._computers.surface_sample import GaussLegendreSurfaceSampler
from rocheometry._computers.surface_selection import (
    FittedSelection,
    VolumeInversionSelection,
)
from rocheometry._computers.watershed import TracedWatershedComputer


def test_accuracy_values_resolve_the_same_graph() -> None:
    for accuracy in (
        rocheometry.Fitted(),
        rocheometry.Reference(),
        rocheometry.Reference(polar_order=16, azimuth_order=8),
    ):
        geometry = rocheometry.ComponentGeometry(0.3, accuracy=accuracy)
        assert geometry._computers is accuracy._computers()
        assert geometry.accuracy == accuracy
        assert not hasattr(geometry, "computers")
        assert repr(accuracy) in repr(geometry)
        binary = rocheometry.BinaryGeometry(0.3, accuracy=accuracy)
        assert (
            binary._computers
            is binary.geometry1._computers
            is binary.geometry2._computers
            is geometry._computers
        )
        assert binary.geometry1.accuracy is accuracy is binary.geometry2.accuracy
        assert (
            rocheometry.BinaryGeometry.from_masses(3, 10, accuracy=accuracy) == binary
        )


def test_reference_graph_contains_no_fitted_or_packaged_nodes() -> None:
    graph = rocheometry.Reference()._computers()
    seen: set[int] = set()

    def visit(node: object) -> None:
        if id(node) in seen or isinstance(node, type) or not is_dataclass(node):
            return
        seen.add(id(node))
        assert not type(node).__name__.startswith(("Packaged", "Fitted"))
        for item in fields(node):
            visit(getattr(node, item.name))

    visit(graph)
    assert isinstance(graph.selection, VolumeInversionSelection)
    assert graph.selection.volume_radius_fill_tolerance == 1e-13
    assert isinstance(graph.section, TracedSectionPathComputer)
    assert graph.section.tolerance == 1e-10
    assert isinstance(graph.watershed, TracedWatershedComputer)
    assert graph.watershed.tolerance == 1e-13
    assert _cached_reference.cache_info().maxsize == 8


def test_fitted_strategy_choices_and_cache_capacities_are_preserved() -> None:
    graph = rocheometry.Fitted()._computers()
    assert isinstance(graph.selection, FittedSelection)
    assert isinstance(graph.selection.fallback, VolumeInversionSelection)
    assert graph.selection.fallback.volume_radius_fill_tolerance == 2e-8
    assert isinstance(graph.section, PlaneRoutedSectionPathComputer)
    assert isinstance(graph.section.centreline, GaugeFixedSectionPathComputer)
    assert isinstance(graph.section.section, TracedSectionPathComputer)
    assert graph.section.section.tolerance == 1e-5
    assert isinstance(graph.watershed, TracedWatershedComputer)
    assert graph.watershed.tolerance == 1e-11
    for accuracy, orders in (
        (rocheometry.Fitted(), (64, 24)),
        (rocheometry.Reference(), (128, 64)),
    ):
        graph = accuracy._computers()
        sample = cast(Memoized[..., object], graph.sample)
        assert sample.maxsize == 16
        sampler = cast(GaussLegendreSurfaceSampler, sample.computer)
        assert (sampler.polar_order, sampler.azimuth_order) == orders
        for name in (
            "l1",
            "outer",
            "lobe_slope",
            "area",
            "mean_curvature",
            "mean_gravity",
        ):
            assert cast(Memoized[..., object], getattr(graph, name)).maxsize == 4096


@pytest.mark.parametrize(
    "accuracy",
    (
        rocheometry.Fitted(),
        rocheometry.Reference(),
        rocheometry.Reference(polar_order=16, azimuth_order=8),
    ),
    ids=repr,
)
def test_quadrature_cache_capacity_does_not_exceed_sample_capacity(
    accuracy: rocheometry.Fitted | rocheometry.Reference,
) -> None:
    graph = accuracy._computers()
    assert isinstance(graph.quadrature, Memoized)
    assert isinstance(graph.sample, Memoized)
    assert graph.quadrature.maxsize == 4
    assert graph.sample.maxsize is not None
    assert graph.quadrature.maxsize <= graph.sample.maxsize
