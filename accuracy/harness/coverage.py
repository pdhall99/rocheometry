"""Check ``accuracy/catalogue.toml`` against this checkout and the release results.

Every reference in the catalogue is resolved: tests must exist, literature
modules and fit folders must exist, timing operations must be in the
benchmark grid. Contract and diagnostic cells come from the accuracy map of
the committed results: a row belongs to the entry whose key is its name or
a dotted prefix of it (``integrals.volume.radius`` to ``integrals.volume``).
Every quantity name in the accuracy map must belong to an entry.

Problems (a dangling reference, an unmapped quantity, a promise without
contract rows) are errors: they mean the catalogue is wrong. Gaps are not
errors: they are counted, not failed.
"""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "accuracy/catalogue.toml"
RESULTS = ROOT / "accuracy/results"

#: Declared columns, in display order, with their short headings.
DECLARED = (
    ("structure", "Structure"),
    ("crosscheck", "Crosscheck"),
    ("guard", "Guard"),
    ("fingerprint", "Fingerprint"),
    ("literature", "Literature"),
    ("fit", "Fit"),
    ("timing", "Timing"),
)
CALIBRATIONS = ("independent", "self-convergence", "partial", "n/a")


@dataclass
class Cell:
    state: str  # "filled", "n/a", "gap" or "to do"
    refs: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Entry:
    key: str
    data: dict[str, Any]
    cells: dict[str, Cell]
    contract: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


@cache
def _test_functions(path: str) -> dict[str, bool]:
    """Test function names in a file, mapped to whether they are marked slow."""
    file = ROOT / path
    if not file.exists():
        return {}
    tree = ast.parse(file.read_text())
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            result[node.name] = any(
                "slow" in ast.unparse(d) for d in node.decorator_list
            )
    return result


def _literature_implemented(path: str, key: str) -> bool | None:
    """Check whether the comparison study at ``path`` measures ``key``.

    Returns:
        Whether ``key`` is among the study's quantities, or `None` if there is
        no such file.
    """
    from accuracy.harness.run import REGISTRY

    if not (ROOT / path).exists():
        return None
    study = REGISTRY.get(path.rpartition("/")[2].removesuffix(".py"))
    return study is not None and key in study.quantities


@cache
def _operations() -> tuple[str, ...]:
    tree = ast.parse((ROOT / "speed/grid.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "OPERATIONS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError("speed/grid.py has no OPERATIONS")


#: Where test references may point: the library's tests and the crosschecks' own.
TEST_ROOTS = ("tests/", "crosschecks/tests/")


def _resolve(column: str, ref: str) -> str | None:
    """Return a problem if ``ref`` does not exist, else None."""
    if ref.startswith(TEST_ROOTS):
        path, _, name = ref.partition("::")
        if name not in _test_functions(path):
            return f"no test {ref}"
        return None
    if column == "literature" and ref.startswith("accuracy/literature/"):
        return None if (ROOT / ref).exists() else f"no literature module {ref}"
    if column == "fit":
        folder = ROOT / ref
        return None if (folder / "provenance.json").exists() else f"no fit {ref}"
    if column == "timing":
        return None if ref in _operations() else f"no timing operation {ref!r}"
    if column == "calibration" and ref.startswith("accuracy/contract/"):
        from accuracy.harness.run import STUDIES

        name = ref.removeprefix("accuracy/contract/").removesuffix(".py")
        exists = (ROOT / ref).exists() and name in STUDIES
        return None if exists else f"no contract study {ref}"
    return f"cannot resolve {ref!r} in column {column}"


def _cell(column: str, value: Any, problems: list[str], key: str) -> Cell:
    if isinstance(value, str):
        kind, _, note = value.partition(":")
        if kind not in ("n/a", "gap"):
            problems.append(f"{key}.{column}: write a list, 'n/a: …' or 'gap: …'")
            return Cell("gap", note=value)
        return Cell(kind, note=note.strip())
    if not isinstance(value, list) or not value:
        problems.append(f"{key}.{column}: an empty cell must say n/a or gap")
        return Cell("gap")
    for ref in value:
        problem = _resolve(column, ref)
        if problem:
            problems.append(f"{key}.{column}: {problem}")
    state = "filled"
    if column == "literature":
        planned = [
            r
            for r in value
            if r.startswith("accuracy/literature/")
            and not _literature_implemented(r, key)
        ]
        if len(planned) == len(value):
            state = "to do"
    return Cell(state, list(value))


#: Row names that report a failure to run, not a quantity.
EXECUTION = ("harness.",)
EXECUTION_SUFFIXES = (".evaluation", ".construction")


def recorded() -> bool:
    """Whether a release run has been recorded."""
    return (RESULTS / "accuracy.json").exists()


def owner(name: str, keys: list[str]) -> str | None:
    """Return the catalogue key a row name belongs to: its longest dotted prefix."""
    matches = [k for k in keys if name == k or name.startswith(k + ".")]
    return max(matches, key=len, default=None)


def load() -> tuple[list[Entry], list[str]]:
    """Parse the catalogue against the recorded release results.

    Rows belong to the entry whose key is their name or a dotted prefix of
    it. Before a release run is recorded there are no rows, and only the
    references are checked.
    """
    problems: list[str] = []
    catalogue = tomllib.loads(CATALOGUE.read_text())["quantity"]
    rows = (
        json.loads((RESULTS / "accuracy.json").read_text())["quantities"]
        if recorded()
        else []
    )
    keys = [data["key"] for data in catalogue]
    for key in sorted({k for k in keys if keys.count(k) > 1}):
        problems.append(f"{key} has more than one entry")
    from accuracy.harness.run import LITERATURE

    # Comparisons with publications fill the literature column, not the contract.
    measured = [
        r
        for r in rows
        if not r["quantity"].startswith(EXECUTION)
        and not r["quantity"].endswith(EXECUTION_SUFFIXES)
        and r.get("study") not in LITERATURE
    ]
    for name in sorted({r["quantity"] for r in measured}):
        if owner(name, keys) is None:
            problems.append(f"{name} in the accuracy map belongs to no entry")
    entries = []
    for data in catalogue:
        key = data["key"]
        if data.get("calibration") not in CALIBRATIONS:
            problems.append(f"{key}: calibration must be one of {CALIBRATIONS}")
        for ref in data.get("calibration_refs", []):
            problem = _resolve("calibration", ref)
            if problem:
                problems.append(f"{key}.calibration_refs: {problem}")
        shares = data.get("shares", [])
        if shares and data.get("calibration") not in ("independent", "partial"):
            problems.append(f"{key}: only an independent or partial truth shares code")
        for share in shares:
            problems += [f"{key}.shares: {p}" for p in _share_problems(share)]
        cells = {}
        for column, _ in DECLARED:
            if column not in data:
                problems.append(f"{key}: missing column {column}")
                cells[column] = Cell("gap")
            else:
                cells[column] = _cell(column, data[column], problems, key)
        mine = [r for r in measured if owner(r["quantity"], keys) == key]
        contract = [r for r in mine if r["contracted_bound"] is not None]
        diagnostics = [r for r in mine if r["contracted_bound"] is None]
        if recorded():
            if data["promised"] and not contract:
                problems.append(f"{key}: promised, but no contract rows recorded")
            if contract and not data["promised"]:
                problems.append(f"{key}: has contract rows but is not promised")
        entries.append(Entry(key, data, cells, contract, diagnostics))
    return entries, problems


def _share_problems(share: Any) -> list[str]:
    """Return what is wrong with one shared-component entry."""
    if not isinstance(share, dict) or not isinstance(share.get("component"), str):
        return ["each entry needs a component"]
    if set(share) - {"component", "checked_by", "note"}:
        return [
            f"{share['component']}: unknown keys {set(share) - {'component', 'checked_by', 'note'}}"
        ]
    checked = share.get("checked_by")
    if isinstance(checked, str):
        return (
            []
            if checked.startswith("gap: ")
            else [f"{share['component']}: write a list of tests or 'gap: …'"]
        )
    if not isinstance(checked, list) or not checked:
        return [f"{share['component']}: write a list of tests or 'gap: …'"]
    return [p for ref in checked if (p := _resolve("crosscheck", ref))]
