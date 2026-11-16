"""Frozen constant names and values are checked independently of computation."""

import ast
import dataclasses
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def bits(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, dict):
        if not value:
            raise TypeError("An empty mutable cache is not a numeric definition")
        return ["dict", [[str(k), bits(v)] for k, v in value.items()]]
    if isinstance(value, tuple):
        return ["tuple", [bits(v) for v in value]]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            "dataclass",
            type(value).__name__,
            [[f.name, bits(getattr(value, f.name))] for f in dataclasses.fields(value)],
        ]
    raise TypeError("Not a numeric constant")


def records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((_ROOT / "src/rocheometry").rglob("*.py")):
        if path.name == "_version.py":
            continue
        module = ".".join(path.relative_to(_ROOT / "src").with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[:-9]
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = (
                [v.id for v in node.targets if isinstance(v, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node.target, ast.Name)
                else []
            )
            for name in names:
                if not name.isupper():
                    continue
                value = getattr(importlib.import_module(module), name)
                try:
                    encoded = bits(value)
                except TypeError:
                    continue
                rows.append(
                    dict(
                        module=module,
                        name=name,
                        bits=encoded,
                        source=str(path.relative_to(_ROOT)),
                    )
                )
    return rows


ROWS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures/numeric-constant-bits.json").read_text()
)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_constant_value_bits(row: dict[str, Any]) -> None:
    value = getattr(importlib.import_module(row["module"]), row["name"])
    assert bits(value) == row["bits"]


def test_numeric_constant_names_are_unique_and_table_is_complete() -> None:
    expected = {(r["module"], r["name"]) for r in ROWS}
    actual = {(r["module"], r["name"]) for r in records()}
    assert actual == expected
    assert len({r["name"] for r in ROWS}) == len(ROWS)
