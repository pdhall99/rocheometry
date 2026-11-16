import json
from collections.abc import Callable, Mapping
from types import MappingProxyType

import pytest

from rocheometry._json import (
    boolean,
    child_mapping,
    float_tuple,
    formatted_json_text,
    integer,
    number,
    parse_mapping,
    sequence,
    string,
)


def test_json_serializers_are_deterministic() -> None:
    document = {"b": 2, "a": 1}

    assert formatted_json_text(document) == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_json_serializers_reject_nonfinite_numbers() -> None:
    document = {"value": float("nan")}

    with pytest.raises(ValueError):
        formatted_json_text(document)


def test_parse_mapping_requires_mapping_at_top_level() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_mapping("[1, 2, 3]", "artifact")


def test_parse_mapping_reports_source_and_keeps_the_decode_error() -> None:
    with pytest.raises(ValueError, match="invalid artifact") as caught:
        parse_mapping("{", "artifact")

    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "decoder",
    [
        child_mapping,
        string,
        boolean,
        number,
        integer,
        sequence,
        float_tuple,
    ],
)
def test_json_accessors_report_missing_fields_consistently(
    decoder: Callable[[Mapping[str, object], str, str], object],
) -> None:
    with pytest.raises(ValueError, match="artifact is missing 'value'") as caught:
        decoder({}, "value", "artifact")

    assert isinstance(caught.value.__cause__, KeyError)


def test_decoders_accept_read_only_mappings() -> None:
    document: Mapping[str, object] = MappingProxyType(
        {
            "child": {"name": "example"},
            "value": 2,
        }
    )

    child = child_mapping(document, "child", "artifact")

    assert string(child, "name", "artifact child") == "example"
    assert number(document, "value", "artifact") == 2.0


@pytest.mark.parametrize(
    ("decoder", "value"),
    [
        (string, 1),
        (boolean, 1),
        (number, True),
        (integer, True),
        (sequence, (1, 2)),
        (float_tuple, [True]),
    ],
)
def test_json_decoders_use_type_error_for_wrong_types(
    decoder: Callable[[Mapping[str, object], str, str], object], value: object
) -> None:
    with pytest.raises(TypeError, match="artifact"):
        decoder({"value": value}, "value", "artifact")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400])
def test_json_numbers_use_value_error_for_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        number({"value": value}, "value", "artifact")
    with pytest.raises(ValueError, match="finite"):
        float_tuple({"value": [value]}, "value", "artifact")
