"""Small typed helpers for dependency-free JSON artefacts."""

import json
from collections.abc import Mapping
from math import isfinite
from typing import cast


def mapping(value: object, label: str) -> dict[str, object]:
    """Return the original string-keyed dictionary after validation."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return cast(dict[str, object], value)


def parse_mapping(text: str, source: str) -> dict[str, object]:
    """Parse JSON text whose top level must be a string-keyed mapping.

    Args:
        text: JSON document to parse.
        source: Caller-facing source label used in validation messages.

    Raises:
        ValueError: The document is not syntactically valid JSON.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {source}") from exc
    return mapping(value, source)


def child_mapping(
    value: Mapping[str, object],
    key: str,
    source: str,
) -> dict[str, object]:
    """Return a string-keyed child dictionary without copying it.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    return mapping(_item(value, key, source), f"{source} {key}")


def string(value: Mapping[str, object], key: str, source: str) -> str:
    """Return a string field without coercion.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    item = _item(value, key, source)
    if not isinstance(item, str):
        raise TypeError(f"{source} {key!r} must be a string")
    return item


def boolean(value: Mapping[str, object], key: str, source: str) -> bool:
    """Return a boolean field without coercion.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    item = _item(value, key, source)
    if not isinstance(item, bool):
        raise TypeError(f"{source} {key!r} must be a boolean")
    return item


def number(value: Mapping[str, object], key: str, source: str) -> float:
    """Return a finite float from a JSON number, excluding booleans.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    item = _item(value, key, source)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{source} {key!r} must be a number")
    try:
        result = float(item)
    except OverflowError as error:
        raise ValueError(f"{source} {key!r} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{source} {key!r} must be finite")
    return result


def integer(value: Mapping[str, object], key: str, source: str) -> int:
    """Return an integer field, excluding booleans and float coercion.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    item = _item(value, key, source)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{source} {key!r} must be an integer")
    return item


def sequence(value: Mapping[str, object], key: str, source: str) -> list[object]:
    """Return the original list stored in an array field.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    item = _item(value, key, source)
    if not isinstance(item, list):
        raise TypeError(f"{source} {key!r} must be an array")
    return cast(list[object], item)


def float_tuple(
    value: Mapping[str, object],
    key: str,
    source: str,
) -> tuple[float, ...]:
    """Return finite floats from a numeric array, excluding booleans.

    Args:
        value: Source mapping; it must contain the requested key.
        key: Field name to read.
        source: Caller-facing source label used in validation messages.
    """
    values = sequence(value, key, source)
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{source} {key!r} must contain only numbers")
        try:
            item_value = float(item)
        except OverflowError as error:
            raise ValueError(
                f"{source} {key!r} must contain only finite numbers"
            ) from error
        if not isfinite(item_value):
            raise ValueError(f"{source} {key!r} must contain only finite numbers")
        result.append(item_value)
    return tuple(result)


def formatted_json_text(value: Mapping[str, object]) -> str:
    """Serialize a mapping to deterministic human-readable JSON text.

    The encoding uses sorted keys, two-space indentation, and a trailing
    newline. Non-finite numbers are rejected.
    """
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _item(value: Mapping[str, object], key: str, source: str) -> object:
    try:
        return value[key]
    except KeyError as error:
        raise ValueError(f"{source} is missing {key!r}") from error
