"""Conveniences for users: query sequences, readable reprs, hints and version."""

import importlib.metadata
from typing import Any, cast

import pytest

import rocheometry


def test_point_queries_accept_any_sequence_of_three_numbers() -> None:
    geometry = rocheometry.ComponentGeometry(0.5)
    surface = geometry.surface(volume_radius_fill=0.8)
    binary = (
        rocheometry.BinaryGeometry(2.0).component(2).surface(volume_radius_fill=0.8)
    )

    assert surface.contains([0.0, 0.0, 0.0]) is surface.contains((0.0, 0.0, 0.0))
    assert surface.radius([1, 0, 0]) == surface.radius((1.0, 0.0, 0.0))
    assert geometry.potential_at([0.1, 0.0, 0.0]) == geometry.potential_at(
        (0.1, 0.0, 0.0)
    )
    assert binary.contains([1.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="point: expected three numbers, got str"):
        surface.contains(cast(Any, "abc"))
    with pytest.raises(ValueError, match="expected three Cartesian coordinates"):
        surface.contains([0.0, 0.0])


def test_point_queries_accept_numpy_arrays() -> None:
    np = pytest.importorskip("numpy")
    surface = rocheometry.ComponentGeometry(0.5).surface(volume_radius_fill=0.8)

    assert surface.contains(np.zeros(3))
    assert surface.radius(np.array([0.0, 0.0, 1.0])) == surface.named_radius("pole")


def test_reprs_summarize_paths_and_show_how_to_select_surfaces() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    path = geometry.lobe().section()

    assert (
        repr(geometry.lobe()) == "ComponentGeometry(qi=1.0, accuracy=Fitted()).lobe()"
    )
    surface = geometry.surface(fill_out_factor=-0.5)
    assert repr(surface).startswith(
        "ComponentGeometry(qi=1.0, accuracy=Fitted()).surface(fill_out_factor="
    )
    assert repr(path).startswith(
        "<CompoundBezierPath: 1 path, 52 cubic segments, bounds=("
    )
    assert repr(path.paths[0]).startswith(
        "<BezierPath: 52 cubic segments, closed, bounds=("
    )
    assert repr(rocheometry.CompoundBezierPath(())) == "<CompoundBezierPath: empty>"
    view = rocheometry.BinaryGeometry(1.0).component(2).lobe()
    assert repr(view).endswith("component_id=2).lobe()")


def test_potential_errors_say_what_to_do() -> None:
    geometry = rocheometry.ComponentGeometry(1.0)
    with pytest.raises(TypeError, match=r"Potential\.jacobi\(value\)"):
        geometry.surface(potential=cast(Any, 2.0))
    with pytest.raises(ValueError, match=r"BinaryGeometry\.component\(i\)"):
        geometry.surface(potential=rocheometry.Potential.kopal(3.75, component_id=1))


def test_level_regime_compares_and_prints_as_its_value() -> None:
    regime = rocheometry.ComponentGeometry(1.0).lobe().regime

    assert regime == "critical"
    assert str(regime) == "critical"


def test_version_is_the_distribution_version() -> None:
    assert rocheometry.__version__ == importlib.metadata.version("rocheometry")
    with pytest.raises(AttributeError):
        _ = rocheometry.not_a_name  # type: ignore[attr-defined]
