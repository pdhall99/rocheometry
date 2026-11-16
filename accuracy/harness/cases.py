"""Frozen contract inputs, stored as hex floats under ``accuracy/contract/grids/``.

Every file records how and when its cases were drawn. A grid or holdout is
never redrawn: a holdout that finds a defect stays as permanent coverage and a
fresh holdout is drawn to certify the fix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

GRIDS = Path(__file__).resolve().parents[1] / "contract/grids"

#: Holdout files whose silhouette views are replayed on the installed outline.
OUTLINE_HOLDOUTS = ("holdout-silhouettes-a.json", "holdout-silhouettes-b.json")


def decode(value: Any) -> Any:
    """Turn hex-float strings back into floats, leaving other values unchanged."""
    if isinstance(value, str) and value.lstrip("-").startswith("0x") and "p" in value:
        return float.fromhex(value)
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    return value


@cache
def load(name: str) -> dict[str, Any]:
    """Return one decoded grid file."""
    return decode(json.loads((GRIDS / name).read_text()))


def sha256(name: str) -> str:
    return hashlib.sha256((GRIDS / name).read_bytes()).hexdigest()


def views(name: str) -> list[dict[str, Any]]:
    """Return the silhouette views of a holdout file.

    Each view has a ``qi``, ``kind``, ``level`` and ``inclination``.
    """
    data = load(name)
    return [dict(zip(data["fields"], case, strict=True)) for case in data["cases"]]


def manifest() -> dict[str, dict[str, Any]]:
    """Checksums and draw records of every frozen input, for run manifests."""
    result = {}
    for path in sorted(GRIDS.glob("*.json")):
        data = json.loads(path.read_text())
        result[path.name] = dict(
            sha256=sha256(path.name),
            cases=len(data.get("cases", ())) or None,
            seed=data.get("seed"),
            drawn=data.get("drawn"),
        )
    return result


def grouped(cases: Iterable[dict[str, Any]]) -> list[tuple[float, list[Any]]]:
    """Jobs of one mass ratio each, with all of its levels."""
    result: dict[float, list[Any]] = {}
    for c in cases:
        result.setdefault(c["qi"], []).append((c["kind"], c["level"]))
    return list(result.items())
