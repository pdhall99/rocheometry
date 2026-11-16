"""Path references keep their metadata and derive samples without new solves."""

from dataclasses import replace
from math import atan2, pi

import pytest

import rocheometry
from rocheometry import Fitted, Reference
from rocheometry._computers.defaults import _Settings
from rocheometry._computers.section_path.reference import (
    reference_component_section,
    reference_section,
)
from rocheometry._roche.field import RocheField
from tests._helpers import _depth, _Wired, single_path


def test_default_reference_uses_the_shared_high_accuracy_graph() -> None:
    reference = reference_component_section(0.3, 1, 0.8, num_upper_points=17)
    surface = rocheometry.ComponentGeometry(
        0.3, accuracy=_Wired(Reference()._computers())
    ).surface(volume_radius_fill=0.8)
    assert reference.path == single_path(surface.section())
    assert reference.surface_potential == surface.potential
    assert reference.volume == surface.volume()
    assert reference.path_tolerance == 1e-10


def test_compatibility_resolution_does_not_change_the_path_or_validation() -> None:
    from tests._section_validation import (
        evaluate_component_section_fit_error,
    )

    reference = reference_component_section(
        0.3, 1, 0.8, num_upper_points=5, polar_order=8, azimuth_order=6
    )
    dense = replace(reference, num_upper_points=257)
    assert dense.path is reference.path
    assert len(reference.upper) == 5 and len(dense.upper) == 257
    for index, point in enumerate(reference.upper):
        assert atan2(point[1], point[0]) == pytest.approx(pi * index / 4, abs=3e-14)
    fitted = single_path(
        Fitted()
        ._computers()
        .section(
            RocheField(0.3),
            _depth(RocheField(0.3), reference.surface_potential),
            rocheometry.ORBITAL_PLANE,
        )
    )
    first = evaluate_component_section_fit_error(
        reference, fitted, distribution_samples=65
    )
    second = evaluate_component_section_fit_error(
        dense, fitted, distribution_samples=65
    )
    assert first == second


def test_samples_and_landmarks_do_not_recompute_field_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = reference_component_section(
        0.3, 1, 0.8, num_upper_points=17, polar_order=8, azimuth_order=6
    )

    def fail(*args: object, **kwargs: object) -> float:
        raise AssertionError("stored path operations must not resolve the field")

    monkeypatch.setattr(RocheField, "potential", fail)
    monkeypatch.setattr(RocheField, "gradient", fail)
    assert len(reference.upper) == len(reference.lower) == 17
    assert reference.closed_points[0] == reference.closed_points[-1]
    assert reference.upper_point(0.0) == reference.landmarks.inner
    assert reference.upper_point(pi) == reference.landmarks.back


def test_reference_sections_use_the_same_path_contract() -> None:
    plane = rocheometry.AxisNormalPlane("z", 0.1)
    reference = reference_section(
        0.3, 1, 0.8, plane, num_upper_points=9, polar_order=8, azimuth_order=6
    )
    graph = _Settings(
        polar_order=8,
        azimuth_order=6,
        fill_tolerance=1e-13,
        watershed_tolerance=1e-13,
        fitted_selection=False,
        exact_anchor_everywhere=True,
        fixed_paths=False,
        path_tolerance=1e-10,
    )._computers()
    assert reference.path == single_path(
        graph.section(
            RocheField(0.3), graph.selection.depth(RocheField(0.3), 0.8), plane
        )
    )
    assert reference.plane == plane
    assert len(reference.upper) == 9


@pytest.mark.parametrize("angle", [-0.1, pi + 0.1, float("nan"), float("inf")])
def test_upper_point_checks_its_angle(angle: float) -> None:
    reference = reference_component_section(
        0.3, 1, 1.0, num_upper_points=3, polar_order=8, azimuth_order=6
    )
    with pytest.raises(ValueError, match="angle"):
        reference.upper_point(angle)
