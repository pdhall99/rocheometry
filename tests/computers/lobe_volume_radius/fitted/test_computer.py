import json
import math
from typing import ClassVar

import pytest

from rocheometry._computers.lobe_volume_radius import LobeVolumeRadiusComputer
from rocheometry._computers.lobe_volume_radius.fitted import (
    InterpolatedLobeVolumeRadiusComputer,
    LinearInterpolator,
    PiecewiseSixthRootHornerLobeVolumeRadiusComputer,
    lobe_volume_radius_computer_from_json,
)
from rocheometry._computers.packaged import load_packaged_lobe_volume_radius_computer
from rocheometry._roche.field import RocheField
from rocheometry._roche.mass import component_mass_ratio


def _filled_radius(
    computer: LobeVolumeRadiusComputer,
    qi: float,
    fill: float,
) -> float:
    return fill * computer(RocheField(qi))


def test_packaged_volume_radius_computer_is_selected_piecewise_model() -> None:
    computer = load_packaged_lobe_volume_radius_computer()

    assert isinstance(
        computer,
        PiecewiseSixthRootHornerLobeVolumeRadiusComputer,
    )
    assert computer.MODEL_FORM == "piecewise-sixth-root-horner"
    assert computer.COORDINATE == "sixth-root-compact"
    assert computer.CONTINUITY == "C2"
    assert computer.JOIN_QI == 1.0
    assert len(computer.low_coefficients) == 18
    assert len(computer.high_coefficients) == 20
    assert computer.coefficient_count == 38


def test_lobe_volume_radius_computers_share_one_critical_interface() -> None:
    interpolated = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, 0.0, 1.0),
        radii=(0.2, 0.4, 0.6),
        interpolator=LinearInterpolator(),
    )
    piecewise = PiecewiseSixthRootHornerLobeVolumeRadiusComputer(
        low_coefficients=(1.0,),
        high_coefficients=(1.0,),
    )
    for computer in (
        interpolated,
        piecewise,
        load_packaged_lobe_volume_radius_computer(),
    ):
        critical = computer(RocheField(1.0))
        assert critical > 0.0
        assert _filled_radius(computer, 1.0, 0.5) == pytest.approx(0.5 * critical)


def test_interpolated_computer_supports_nonuniform_log_qi_nodes() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, -0.25, 1.0),
        radii=(10.0, 20.0, 30.0),
        interpolator=LinearInterpolator(),
    )

    assert computer.qi_min == pytest.approx(0.1)
    assert computer.qi_max == pytest.approx(10.0)
    assert computer(RocheField(0.1)) == 10.0
    assert computer(RocheField(10.0**-0.625)) == pytest.approx(15.0)
    assert computer(RocheField(10.0**-0.25)) == pytest.approx(20.0)
    assert computer(RocheField(10.0**0.375)) == pytest.approx(25.0)
    assert computer(RocheField(10.0)) == 30.0


def test_fill_scaling_is_external_to_lobe_radius_computer() -> None:
    computer = load_packaged_lobe_volume_radius_computer()
    critical = computer(RocheField(1.0))

    assert _filled_radius(computer, 1.0, 0.25) == pytest.approx(0.25 * critical)


@pytest.mark.parametrize(
    ("qi", "expected"),
    (
        (1.0e-30, 4.933062311395613e-11),
        (1.0e-4, 0.022649176270789127),
        (1.0, 0.37986324149936024),
        (1.0e4, 0.8047450416239559),
        (1.0e30, 0.8148856767821752),
    ),
)
def test_packaged_piecewise_model_matches_selected_values(
    qi: float,
    expected: float,
) -> None:
    computer = load_packaged_lobe_volume_radius_computer()
    if qi == 1e30:
        # Outside the contract domain, this stored pair fails the exact
        # mass-error representability guard before a field can be supplied.
        with pytest.raises(ValueError, match="mass-pair error"):
            RocheField(qi)
        return
    assert computer(RocheField(qi)) == pytest.approx(
        expected,
        rel=2.0e-15,
    )


def test_component_mass_ratio_views_share_one_packaged_computer() -> None:
    samples = (
        1.0e-4,
        1.37e-4,
        0.0037,
        0.27,
        1.0,
        3.7,
        270.0,
        7310.0,
        1.0e4,
    )
    computer = load_packaged_lobe_volume_radius_computer()

    for qi in samples:
        expected = computer(RocheField(qi))
        assert computer(RocheField(component_mass_ratio(qi, 1))) == pytest.approx(
            expected,
            rel=0.0,
            abs=2.0e-15,
        )
        assert computer(RocheField(component_mass_ratio(1.0 / qi, 2))) == pytest.approx(
            expected,
            rel=0.0,
            abs=2.0e-15,
        )


def test_interpolated_computer_canonicalizes_table_values() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1, 0, 1),
        radii=(1, 2, 3),
    )

    assert computer.log_qis == (-1.0, 0.0, 1.0)
    assert computer.radii == (1.0, 2.0, 3.0)


def test_interpolated_computer_rejects_wrong_table_types() -> None:
    with pytest.raises(TypeError, match=r"log_qis\[1\]: expected a real number"):
        InterpolatedLobeVolumeRadiusComputer(
            log_qis=(-1.0, True, 1.0),
            radii=(1.0, 2.0, 3.0),
        )


def test_interpolated_computer_rejects_values_outside_model_range() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-4.0, 4.0),
        radii=(0.1, 0.8),
    )

    with pytest.raises(ValueError, match="qi"):
        computer(RocheField(9.0e-5))
    with pytest.raises(ValueError, match="qi"):
        computer(RocheField(1.1e4))


def test_piecewise_computer_serialization_round_trip() -> None:
    computer = PiecewiseSixthRootHornerLobeVolumeRadiusComputer(
        low_coefficients=(1.0, 0.25),
        high_coefficients=(1.25, -0.125),
    )

    loaded = lobe_volume_radius_computer_from_json(computer.to_json())

    assert loaded == computer
    assert loaded(RocheField(1.0)) == pytest.approx(0.3125)


def test_interpolated_computer_serialization_round_trip() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, -0.25, 1.0),
        radii=(10.0, 20.0, 30.0),
        interpolator=LinearInterpolator(),
    )

    mapping = computer.to_mapping()
    text = computer.to_json()
    loaded = InterpolatedLobeVolumeRadiusComputer.from_json(text)

    assert InterpolatedLobeVolumeRadiusComputer.from_mapping(mapping) == computer
    assert loaded == computer
    assert loaded(RocheField(10.0**-0.25)) == pytest.approx(20.0)

    document = json.loads(text)
    assert document == {
        "interpolator": "linear",
        "log_qis": [-1.0, -0.25, 1.0],
        "model_form": "interpolator",
        "radii": [10.0, 20.0, 30.0],
    }


def test_interpolated_computer_rejects_interpolator_name_masquerade() -> None:
    class MasqueradingInterpolator:
        NAME: ClassVar[str] = "linear"

        def __call__(
            self,
            xs: tuple[float, ...],
            ys: tuple[float, ...],
            x: float,
        ) -> float:
            return ys[0]

    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, 1.0),
        radii=(10.0, 30.0),
        interpolator=MasqueradingInterpolator(),
    )

    with pytest.raises(ValueError, match="registered strategy"):
        computer.to_json()


def test_interpolated_computer_rejects_wrong_model_form() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, 1.0),
        radii=(10.0, 30.0),
    )
    document = json.loads(computer.to_json())
    document["model_form"] = "analytic"

    with pytest.raises(ValueError, match="model form"):
        InterpolatedLobeVolumeRadiusComputer.from_json(json.dumps(document))


def test_interpolated_computer_rejects_unknown_interpolator() -> None:
    computer = InterpolatedLobeVolumeRadiusComputer(
        log_qis=(-1.0, 1.0),
        radii=(10.0, 30.0),
    )
    document = json.loads(computer.to_json())
    document["interpolator"] = "cubic"

    with pytest.raises(ValueError, match="interpolator"):
        InterpolatedLobeVolumeRadiusComputer.from_json(json.dumps(document))


def _derivatives(coefficients: tuple[float, ...], x: float) -> tuple[float, ...]:
    """Value and first two derivatives of a power-basis polynomial at x."""
    terms = list(enumerate(coefficients))
    return (
        math.fsum(a * x**i for i, a in terms),
        math.fsum(i * a * x ** (i - 1) for i, a in terms if i),
        math.fsum(i * (i - 1) * a * x ** (i - 2) for i, a in terms if i > 1),
    )


def test_packaged_pieces_join_twice_differentiably_at_equal_mass() -> None:
    # R = s**2 P(4s - c) with s = 1/2 at qi = 1, so the pieces meet C2 exactly
    # when the low piece at x = 1 and the high piece at x = -1 agree to second order.
    computer = load_packaged_lobe_volume_radius_computer()
    assert isinstance(computer, PiecewiseSixthRootHornerLobeVolumeRadiusComputer)
    low = _derivatives(computer.low_coefficients, 1.0)
    high = _derivatives(computer.high_coefficients, -1.0)
    for a, b in zip(low, high, strict=True):
        assert a == pytest.approx(b, rel=1e-12)


def test_packaged_radius_has_the_hill_power_law_and_a_finite_heavy_limit() -> None:
    # The form makes R ~ qi**(1/3) as qi -> 0 and R -> P_high(1) as qi -> inf;
    # in between the radius must rise and its log slope fall strictly.
    computer = load_packaged_lobe_volume_radius_computer()
    assert isinstance(computer, PiecewiseSixthRootHornerLobeVolumeRadiusComputer)
    grid = [10 ** (k / 1000) for k in range(-12000, 12001)]
    radii = [computer(RocheField(q)) for q in grid]
    slopes = [computer.logarithmic_slope(RocheField(q)) for q in grid]
    assert all(a < b for a, b in zip(radii, radii[1:]))
    assert all(a > b > 0 for a, b in zip(slopes, slopes[1:]))
    assert computer.logarithmic_slope(RocheField(1e-300)) == pytest.approx(1 / 3)
    hill = computer(RocheField(1e-300)) / 1e-100
    assert computer(RocheField(1e-30)) / 1e-10 == pytest.approx(hill, rel=1e-10)
    assert slopes[-1] < 1e-6
    heavy = math.fsum(computer.high_coefficients)
    assert computer(RocheField(1e15)) == pytest.approx(heavy, rel=1e-7)
