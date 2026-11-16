from math import sqrt

import pytest

from rocheometry._numerics import interval_nodes


def test_interval_nodes_support_uniform_and_chebyshev_lobatto() -> None:
    assert interval_nodes("uniform", -2.0, 2.0, 5) == (-2.0, -1.0, 0.0, 1.0, 2.0)
    assert interval_nodes("chebyshev-lobatto", -2.0, 2.0, 5) == pytest.approx(
        (-2.0, -sqrt(2.0), 0.0, sqrt(2.0), 2.0)
    )


def test_interval_nodes_reproduce_the_component_lobatto_sequence() -> None:
    nodes = interval_nodes("chebyshev-lobatto", -4.0, 4.0, 17)

    assert nodes[3] == -3.3258784492101814
    assert nodes[8] == 2.4492935982947064e-16
    assert nodes[13] == 3.3258784492101814


def test_interval_nodes_preserve_exact_endpoints() -> None:
    nodes = interval_nodes("chebyshev-lobatto", 0.1, 0.975, 17)

    assert nodes[0] == 0.1
    assert nodes[-1] == 0.975


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum", "count"),
    (
        ("unknown", 0.0, 1.0, 3),
        ("uniform", 1.0, 0.0, 3),
        ("uniform", 0.0, 1.0, 1),
    ),
)
def test_interval_nodes_reject_invalid_meshes(
    kind: str,
    minimum: float,
    maximum: float,
    count: int,
) -> None:
    with pytest.raises(ValueError):
        interval_nodes(kind, minimum, maximum, count)  # type: ignore[bad-argument-type]


@pytest.mark.parametrize(
    ("kind", "minimum", "maximum", "count", "match"),
    [
        (1, 0.0, 1.0, 3, "kind: expected str"),
        ("uniform", True, 1.0, 3, "minimum: expected a real number"),
        ("uniform", 0.0, 1.0, True, "count: expected int"),
        ("uniform", 0.0, 1.0, 2.5, "count: expected int"),
    ],
)
def test_interval_nodes_rejects_wrong_input_types(
    kind: object,
    minimum: object,
    maximum: float,
    count: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        interval_nodes(
            kind,  # type: ignore[bad-argument-type]
            minimum,  # type: ignore[bad-argument-type]
            maximum,
            count,  # type: ignore[bad-argument-type]
        )
