import math
from typing import cast

import pytest

from rocheometry._frames import (
    Frame,
    component_frame,
    frame_transform,
    transform,
    transform_vector,
)
from rocheometry._types import ComponentId


@pytest.mark.parametrize(
    ("component_id", "error"),
    [
        (0, ValueError),
        (3, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        (None, TypeError),
    ],
)
def test_component_frame_rejects_invalid_component_ids(
    component_id: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="component_id: expected"):
        component_frame(cast(ComponentId, component_id))


@pytest.mark.parametrize(
    ("component_id", "frame"), [(1, "component1"), (2, "component2")]
)
def test_component_frame_selects_valid_component_ids(
    component_id: ComponentId, frame: Frame
) -> None:
    assert component_frame(component_id) == frame


@pytest.mark.parametrize("argument", ["source", "target", "both"])
@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("typo", ValueError),
        ("", ValueError),
        (None, TypeError),
        (1, TypeError),
        ([], TypeError),
    ],
)
def test_frame_transform_validates_names_before_identity_or_com_fallback(
    argument: str, invalid: object, error: type[Exception]
) -> None:
    source = cast(Frame, invalid) if argument != "target" else "binary"
    target = cast(Frame, invalid) if argument != "source" else "binary"
    name = "target" if argument == "target" else "source"
    with pytest.raises(error, match=f"{name}: expected"):
        frame_transform(source=source, target=target, q=0.3)


@pytest.mark.parametrize("q", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("target", ["binary", "centre_of_mass"])
def test_frame_transform_rejects_invalid_mass_ratios(q: float, target: Frame) -> None:
    with pytest.raises(ValueError, match="q: expected"):
        frame_transform(source="binary", target=target, q=q)


@pytest.mark.parametrize("q", [True, "0.3"])
def test_frame_transform_rejects_wrong_mass_ratio_types(q: object) -> None:
    with pytest.raises(TypeError, match="q: expected a real number"):
        frame_transform(source="binary", target="centre_of_mass", q=cast(float, q))


@pytest.mark.parametrize(
    "frame", ["binary", "component1", "component2", "centre_of_mass"]
)
def test_identity_frame_transform_needs_no_mass_ratio(frame: Frame) -> None:
    assert frame_transform(source=frame, target=frame).apply((0.2, 0.3)) == (0.2, 0.3)


@pytest.mark.parametrize("name", ["position", "vector"])
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (None, TypeError, "expected tuple"),
        ([0.0, 0.0, 0.0], TypeError, "expected tuple"),
        ((0.0, 0.0), ValueError, "expected three Cartesian coordinates"),
        ((0.0, 0.0, 0.0, 0.0), ValueError, "expected three Cartesian coordinates"),
        ((0.0, 0.0, None), TypeError, r"\[2\]: expected a real number"),
        ((0.0, 0.0, math.nan), ValueError, r"\[2\]: expected a finite number"),
        ((0.0, 0.0, math.inf), ValueError, r"\[2\]: expected a finite number"),
    ],
)
def test_frame_transforms_validate_all_three_coordinates(
    name: str, value: object, error: type[Exception], message: str
) -> None:
    operation = transform if name == "position" else transform_vector
    with pytest.raises(error, match=f"{name}.*{message}"):
        operation(
            cast(tuple[float, float, float], value), source="binary", target="binary"
        )


def test_all_frame_transforms_round_trip() -> None:
    position = (0.23, -0.17, 0.09)
    frames: tuple[Frame, ...] = ("component1", "component2", "binary", "centre_of_mass")

    for source in frames:
        for target in frames:
            converted = transform(position, source=source, target=target, q=0.37)
            restored = transform(converted, source=target, target=source, q=0.37)
            assert restored == pytest.approx(position)


def test_all_frame_vector_transforms_round_trip() -> None:
    vector = (0.23, -0.17, 0.09)
    frames: tuple[Frame, ...] = (
        "component1",
        "component2",
        "binary",
        "centre_of_mass",
    )

    for source in frames:
        for target in frames:
            converted = transform_vector(
                vector,
                source=source,
                target=target,
                q=0.37,
            )
            restored = transform_vector(
                converted,
                source=target,
                target=source,
                q=0.37,
            )
            assert restored == pytest.approx(vector)


def test_vector_transform_ignores_frame_translation() -> None:
    vector = (0.23, -0.17, 0.09)

    assert transform_vector(
        vector,
        source="binary",
        target="centre_of_mass",
        q=0.37,
    ) == pytest.approx(vector)
    assert transform_vector(
        vector,
        source="component2",
        target="binary",
    ) == pytest.approx((-0.23, 0.17, 0.09))


def test_frame_transform_matches_point_transform() -> None:
    point = (0.23, -0.17, 0.09)
    frames: tuple[Frame, ...] = (
        "component1",
        "component2",
        "binary",
        "centre_of_mass",
    )

    for source in frames:
        for target in frames:
            transformed = frame_transform(
                source=source,
                target=target,
                q=0.37,
            ).apply((point[0], point[1]))
            expected = transform(point, source=source, target=target, q=0.37)

            assert transformed == pytest.approx(expected[:2])
            assert expected[2] == point[2]


def test_frame_transform_inverse_matches_reverse_frame_change() -> None:
    forward = frame_transform(
        source="component2",
        target="centre_of_mass",
        q=0.37,
    )
    reverse = frame_transform(
        source="centre_of_mass",
        target="component2",
        q=0.37,
    )

    point = (0.31, -0.22)
    assert forward.inverse().apply(point) == pytest.approx(reverse.apply(point))


def test_component_two_affine_frame_transform_is_rotation_and_translation() -> None:
    transform_2d = frame_transform(
        source="component2",
        target="binary",
    )

    assert transform_2d.apply((0.25, 0.1)) == (0.75, -0.1)


def test_binary_to_centre_of_mass_affine_transform_uses_barycentre() -> None:
    transform_2d = frame_transform(
        source="binary",
        target="centre_of_mass",
        q=0.25,
    )

    assert transform_2d.apply((0.0, 0.0)) == pytest.approx((-0.8, 0.0))


def test_component_two_frame_is_right_handed_pi_rotation_about_z() -> None:
    origin = transform(
        (0.0, 0.0, 0.0),
        source="component2",
        target="binary",
    )
    x_tip = transform(
        (1.0, 0.0, 0.0),
        source="component2",
        target="binary",
    )
    y_tip = transform(
        (0.0, 1.0, 0.0),
        source="component2",
        target="binary",
    )
    z_tip = transform(
        (0.0, 0.0, 1.0),
        source="component2",
        target="binary",
    )

    ex = tuple(x_tip[index] - origin[index] for index in range(3))
    ey = tuple(y_tip[index] - origin[index] for index in range(3))
    ez = tuple(z_tip[index] - origin[index] for index in range(3))
    cross_xy = (
        ex[1] * ey[2] - ex[2] * ey[1],
        ex[2] * ey[0] - ex[0] * ey[2],
        ex[0] * ey[1] - ex[1] * ey[0],
    )

    assert origin == (1.0, 0.0, 0.0)
    assert ex == (-1.0, 0.0, 0.0)
    assert ey == (0.0, -1.0, 0.0)
    assert ez == (0.0, 0.0, 1.0)
    assert cross_xy == ez


def test_component_two_frame_points_towards_component_one() -> None:
    assert transform(
        (0.25, 0.0, 0.0),
        source="component2",
        target="binary",
    ) == (0.75, -0.0, 0.0)


def test_transform_between_component_frames() -> None:
    assert transform(
        (0.25, 0.1, -0.2),
        source="component1",
        target="component2",
    ) == (0.75, -0.1, -0.2)


def test_com_transform_requires_mass_ratio() -> None:
    with pytest.raises(ValueError, match="q is required"):
        transform((0.0, 0.0, 0.0), source="binary", target="centre_of_mass")
