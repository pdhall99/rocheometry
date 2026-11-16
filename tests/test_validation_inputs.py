import math
from typing import cast

import pytest

from rocheometry._roche.mass import component_mass_ratio
from rocheometry._types import ComponentId
from rocheometry._validation import (
    normalize_component_mass_ratio,
    normalize_mass_ratio,
    normalize_separation,
    normalize_volume_radius_fill,
    require_binary_frame,
    require_component_id,
)


def test_system_mass_ratio_uses_m1_over_m2() -> None:
    assert normalize_mass_ratio(2) == 2.0


def test_mass_ratio_wraps_float_overflow() -> None:
    with pytest.raises(ValueError, match="q: expected a finite number"):
        normalize_mass_ratio(10**400)


@pytest.mark.parametrize(
    ("q", "component_id", "expected"),
    [
        (4.0, 1, 4.0),
        (4.0, 2, 0.25),
        (0.25, 1, 0.25),
        (0.25, 2, 4.0),
    ],
)
def test_component_mass_ratio_is_component_over_companion(
    q: float,
    component_id: ComponentId,
    expected: float,
) -> None:
    assert component_mass_ratio(q, component_id) == pytest.approx(expected)


@pytest.mark.parametrize("q", [0.0, -1.0, math.inf, math.nan])
def test_mass_ratio_rejects_nonphysical_values(q: float) -> None:
    with pytest.raises(ValueError, match="q:"):
        normalize_mass_ratio(q)


@pytest.mark.parametrize("qi", [0.0, -1.0, math.inf, math.nan])
def test_component_mass_ratio_rejects_nonphysical_values(qi: float) -> None:
    with pytest.raises(ValueError, match="qi:"):
        normalize_component_mass_ratio(qi)


@pytest.mark.parametrize("value", [True, "1.0"])
def test_mass_ratio_rejects_wrong_runtime_types(value: object) -> None:
    with pytest.raises(TypeError, match="q: expected a real number"):
        normalize_mass_ratio(cast(float, value))


@pytest.mark.parametrize("volume_radius_fill", [0.0, -0.1, 1.1, math.inf, math.nan])
def test_volume_radius_fill_rejects_nonphysical_values(
    volume_radius_fill: float,
) -> None:
    with pytest.raises(ValueError, match="volume_radius_fill:"):
        normalize_volume_radius_fill(volume_radius_fill)


def test_volume_radius_fill_rejects_bool() -> None:
    with pytest.raises(TypeError, match="volume_radius_fill: expected a real number"):
        normalize_volume_radius_fill(cast(float, True))


@pytest.mark.parametrize("separation", [0.0, -1.0, math.inf, math.nan])
def test_separation_rejects_nonphysical_values(separation: float) -> None:
    with pytest.raises(ValueError, match="separation:"):
        normalize_separation(separation)


def test_require_component_id_returns_same_value() -> None:
    component_id = 2

    assert require_component_id(component_id) is component_id


@pytest.mark.parametrize("component_id", [0, 3])
def test_require_component_id_rejects_invalid_identity(component_id: int) -> None:
    with pytest.raises(ValueError, match="component_id: expected 1 or 2"):
        require_component_id(component_id)


def test_require_component_id_rejects_bool() -> None:
    with pytest.raises(TypeError, match="component_id: expected int, got bool"):
        require_component_id(True)


def test_require_binary_frame_preserves_valid_frame() -> None:
    frame = "centre_of_mass"
    assert require_binary_frame(frame) is frame


def test_require_binary_frame_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match="frame: expected"):
        require_binary_frame("component")
