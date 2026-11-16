from typing import cast

import pytest
from crosschecks.distance.bezier import _PathDistance

from rocheometry import ORBITAL_PLANE, Fitted
from rocheometry._bezier import BezierPath, CubicBezier
from rocheometry._computers.section_path import (
    SectionReference,
    reference_component_section,
)
from rocheometry._roche.field import RocheField
from tests._helpers import _depth, single_path
from tests._section_validation import (
    _polyline_self_intersects,
    evaluate_component_section_fit_error,
)


def test_directed_distance_requires_both_directions_for_symmetry() -> None:
    horizontal = BezierPath(
        (CubicBezier((0.0, 0.0), (1 / 3, 0.0), (2 / 3, 0.0), (1.0, 0.0)),), closed=False
    )
    diagonal = BezierPath(
        (CubicBezier((0.0, 0.0), (1 / 3, 1 / 3), (2 / 3, 2 / 3), (1.0, 1.0)),),
        closed=False,
    )
    horizontal_to_diagonal = max(
        _PathDistance(diagonal)(p) for p in horizontal.sample(17)
    )
    diagonal_to_horizontal = max(
        _PathDistance(horizontal)(p) for p in diagonal.sample(17)
    )
    assert horizontal_to_diagonal == pytest.approx(2.0**-0.5)
    assert diagonal_to_horizontal == pytest.approx(1.0)


def test_self_intersection_check_detects_bow_tie() -> None:
    simple = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    )
    bow_tie = (
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    )

    assert not _polyline_self_intersects(simple)
    assert _polyline_self_intersects(bow_tie)


def _validation_pair() -> tuple[
    SectionReference,
    tuple[CubicBezier, ...],
]:
    reference = reference_component_section(
        1.0,
        1,
        0.7,
        num_upper_points=65,
        polar_order=8,
        azimuth_order=8,
    )
    segments = single_path(
        Fitted()
        ._computers()
        .section(
            RocheField(1.0),
            _depth(RocheField(1.0), reference.surface_potential),
            ORBITAL_PLANE,
        )
    ).segments[1:26]
    return reference, segments


def test_error_report_is_ordered_and_normalized() -> None:
    reference, segments = _validation_pair()
    report = evaluate_component_section_fit_error(
        reference,
        BezierPath(segments, closed=False),
        distribution_samples=129,
    )

    assert report.scale == reference.volume_radius
    assert report.hausdorff_distance > 0.0
    assert report.normalized_hausdorff == pytest.approx(
        report.hausdorff_distance / report.scale
    )
    assert 0.0 <= report.median_distance <= report.p95_distance
    assert report.p95_distance <= report.p99_distance
    assert report.p99_distance <= report.hausdorff_distance


def test_error_report_rejects_too_few_distribution_samples() -> None:
    reference, segments = _validation_pair()

    with pytest.raises(ValueError, match="distribution_samples"):
        evaluate_component_section_fit_error(
            reference,
            BezierPath(segments, closed=False),
            distribution_samples=1,
        )


@pytest.mark.parametrize("value", [True, "2"])
def test_error_report_rejects_wrong_control_types(value: object) -> None:
    reference, segments = _validation_pair()

    with pytest.raises(TypeError, match="distribution_samples"):
        evaluate_component_section_fit_error(
            reference,
            BezierPath(segments, closed=False),
            distribution_samples=cast(int, value),
        )


def test_a_reference_path_has_zero_self_error() -> None:
    reference, _ = _validation_pair()
    report = evaluate_component_section_fit_error(
        reference, reference.path, distribution_samples=65
    )
    assert report.hausdorff_distance / reference.volume_radius <= 1e-12
    assert (
        report.reference_tolerance == reference.path_tolerance * reference.volume_radius
    )


def test_validation_uses_no_field_solves_after_reference_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, segments = _validation_pair()

    def fail(*args: object, **kwargs: object) -> float:
        raise AssertionError("path validation must not re-solve geometry")

    monkeypatch.setattr(RocheField, "potential", fail)
    monkeypatch.setattr(RocheField, "gradient", fail)
    report = evaluate_component_section_fit_error(
        reference, BezierPath(segments, closed=False), distribution_samples=33
    )
    assert report.hausdorff_distance > 0.0


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_validation_rejects_invalid_normalization(scale: float) -> None:
    reference, segments = _validation_pair()
    with pytest.raises(ValueError, match="scale"):
        evaluate_component_section_fit_error(
            reference, BezierPath(segments, closed=False), scale=scale
        )
