"""Accuracy settings reproduce their recorded numerical values exactly."""

import importlib
import json
import pickle
import platform
from dataclasses import FrozenInstanceError, replace
from math import pi
from pathlib import Path
from typing import Any, TypedDict

import pytest

import rocheometry
from rocheometry._accuracy import _Accuracy
from tests import _fingerprints as fingerprints


class _Case(TypedDict):
    reference: bool
    qi: float
    kind: str
    values: dict[str, str]


_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/accuracy-values.json").read_text()
)
_CASES: list[_Case] = _BASELINE["cases"]
# Byte fingerprints include libm rounding. Other environments run the exact-
# field geometry and independent quadrature checks below, without skipping.
_SAME_ENVIRONMENT = _BASELINE["environment"] == {
    "system": platform.system(),
    "machine": platform.machine(),
    "python": list(platform.python_version_tuple()[:2]),
}


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda c: (
        f"{'reference' if c['reference'] else 'fitted'}-{c['qi']}-{c['kind']}"
    ),
)
def test_accuracy_settings_are_bit_identical_to_previous_graphs(case: _Case) -> None:
    s = fingerprints.surface(case["reference"], case["qi"], case["kind"])
    values = fingerprints.values(s)
    if _SAME_ENVIRONMENT:
        assert {
            name: fingerprints.digest(value) for name, value in values.items()
        } == case["values"]
    else:
        _assert_portable_geometry(s)


def _assert_portable_geometry(
    surface: rocheometry.ComponentSurfaceGeometry,
) -> dict[str, Any]:
    """Run the contract's exact-field checks when libm digests cannot match."""
    from crosschecks.distance.outlines import normal_error
    from crosschecks.instrumented.quadrature import inspect

    reference = isinstance(surface.component.accuracy, rocheometry.Reference)
    geometry = inspect(surface)
    for name in ("level_backward", "gradient_backward", "gravity_backward"):
        assert geometry[name] <= 1.0, (name, geometry[name])
    for name in (
        "unit",
        "area_consistency",
        "curvature_consistency",
        "gravity_consistency",
    ):
        assert geometry[name] <= 1e-12, (name, geometry[name])
    assert geometry["outward"] == geometry["reflections"] == 0
    plane = rocheometry.CentrelinePlane(pi / 3)
    path_errors = {}
    for silhouette in (False, True):
        envelope = (
            surface.silhouette_envelope(plane)
            if silhouette
            else surface.section_envelope(plane)
        )
        path = envelope if silhouette else surface.section(plane)
        error, _, _ = normal_error(
            surface,
            path,
            plane,
            envelope,
            silhouette=silhouette,
            samples_per_segment=17,
        )
        assert error <= (1e-10 if reference else 2.25e-4), error
        path_errors["silhouette" if silhouette else "section"] = error
    truth = rocheometry.ComponentGeometry(
        surface.component.qi,
        accuracy=rocheometry.Reference(polar_order=512, azimuth_order=192),
    )
    measured = truth._surface_at_depth(surface._depth)
    scalar_errors = {}
    for name in (
        "volume",
        "volume_radius",
        "volume_radius_fill",
        "area",
        "integrated_mean_curvature",
        "mean_gravity",
    ):
        error = abs(getattr(surface, name)() / getattr(measured, name)() - 1)
        # The README's scalar contract: 1e-6 Fitted; 1e-10 Reference, except
        # 2e-8 for mean curvature at the default orders.
        bound = (
            (2e-8 if name == "integrated_mean_curvature" else 1e-10)
            if reference
            else 1e-6
        )
        assert error <= bound, (name, error, bound)
        scalar_errors[name] = dict(error=error, bound=bound)
    return dict(nodes=geometry, paths_R_vol=path_errors, scalar_relative=scalar_errors)


@pytest.mark.parametrize("case", _CASES)
def test_portable_fingerprint_checks_run_when_platform_differs(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    """Exercise all 18 non-recording cases on the digest recording machine too."""
    monkeypatch.setattr(
        __import__(__name__, fromlist=["_SAME_ENVIRONMENT"]),
        "_SAME_ENVIRONMENT",
        False,
    )
    test_accuracy_settings_are_bit_identical_to_previous_graphs(case)


@pytest.mark.parametrize(
    "accuracy",
    (
        rocheometry.Fitted(),
        rocheometry.Reference(),
        rocheometry.Reference(polar_order=16, azimuth_order=8),
    ),
)
def test_accuracy_is_a_frozen_hashable_value(accuracy: _Accuracy) -> None:
    assert pickle.loads(pickle.dumps(accuracy)) == accuracy
    assert hash(pickle.loads(pickle.dumps(accuracy))) == hash(accuracy)
    # Python 3.11's slotted frozen dataclasses raise TypeError, not
    # FrozenInstanceError, for an attribute that is not a field (fixed in 3.12).
    with pytest.raises((FrozenInstanceError, TypeError)):
        setattr(accuracy, "polar_order", 4)
    assert rocheometry.ComponentGeometry(
        1, accuracy=accuracy
    ) == rocheometry.ComponentGeometry(1, accuracy=pickle.loads(pickle.dumps(accuracy)))


@pytest.mark.parametrize("name", ("polar_order", "azimuth_order"))
@pytest.mark.parametrize(
    "value,error",
    (
        (3, ValueError),
        (0, ValueError),
        (True, TypeError),
        (4.0, TypeError),
        (None, TypeError),
    ),
)
def test_reference_validates_orders(
    name: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=name):
        rocheometry.Reference(**{name: value})  # type: ignore[arg-type]


def test_accuracy_is_part_of_geometry_equality_and_replacement() -> None:
    fitted = rocheometry.ComponentGeometry(1)
    precise = replace(fitted, accuracy=rocheometry.Reference())
    assert fitted != precise
    assert fitted.accuracy == rocheometry.Fitted()
    assert precise._computers is rocheometry.Reference()._computers()
    binary = replace(rocheometry.BinaryGeometry(1), accuracy=rocheometry.Reference())
    assert binary.geometry1.accuracy is binary.accuracy is binary.geometry2.accuracy


def test_computer_extension_api_has_been_removed() -> None:
    assert not any("computer" in name.lower() for name in rocheometry.__all__)
    for name in (
        "ComputerSet",
        "fitted_computers",
        "reference_computers",
        "packaged_lobe_volume_computer",
        "packaged_lobe_volume_radius_computer",
    ):
        assert not hasattr(rocheometry, name)
    with pytest.raises(ModuleNotFoundError, match="rocheometry.computers"):
        importlib.import_module("rocheometry.computers")
    with pytest.raises(TypeError, match="computers"):
        rocheometry.ComponentGeometry(1, computers=object())  # type: ignore[unexpected-keyword]


def test_fingerprints_cover_every_case() -> None:
    # Digests change exactly when an output does: re-record them with
    # `hatch run hatch-test.py3.14:fingerprints` on the release machine.
    assert [(c["reference"], c["qi"], c["kind"]) for c in _CASES] == list(
        fingerprints.CASES
    )
