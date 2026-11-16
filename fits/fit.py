"""The base class of a packaged fit, and how a fit is checked.

A fit lives in ``fits/<name>/``: ``config.yaml`` (its settings),
``build.py`` (the builder), ``fit.py`` (its ``Fit`` subclass), ``holdout.json``
(frozen cases it was never trained on), ``provenance.json`` (the committed
record of the packaged file) and a ``README.md``. Every fit is registered by
name in ``FITS`` in ``fits/__main__.py``.

``build.py`` stays apart from the class because a build's inputs are hashed:
``provenance.json`` records the hashes of ``config.yaml``, ``build.py`` and
``fits/common.py``, and a reproducible fit must be rebuilt when they
change. This module is not an input, so a change to how fits are checked never
changes how they are built.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from fits.common import DATA, ROOT, build_inputs, environment, sha256


@dataclass(frozen=True)
class Holdout:
    """A packaged fit's worst error on its frozen holdout, against its bound.

    Attributes:
        quantity: The catalogue key the error measures.
        worst: The worst relative error over the holdout.
        bound: The bound from the fit's ``config.yaml``.
        cases: How many holdout cases were measured.
        at: Where the worst error occurred, such as its mass ratio.
    """

    quantity: str
    worst: float
    bound: float
    cases: int
    at: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.worst <= self.bound


@dataclass
class Report:
    """The outcome of one fit's check."""

    name: str
    problems: list[str]
    notes: list[str]
    holdouts: list[Holdout]

    @property
    def passed(self) -> bool:
        return not self.problems

    def tally(self) -> str:
        """One line, in the style of the other tools: name, count, statuses."""
        passed = sum(h.passed for h in self.holdouts)
        failed = len(self.holdouts) - passed
        statuses = ", ".join(
            f"{n} {s}" for n, s in ((passed, "pass"), (failed, "fail")) if n
        )
        return f"{self.name:24s} {len(self.holdouts):6d} rows  {statuses or 'none'}"


class Fit(ABC):
    """A packaged fit: how to build it, and how to judge it on its holdout."""

    #: The fit's folder under fits/, and its name in commands.
    name: ClassVar[str]

    @abstractmethod
    def build(
        self, config: dict[str, Any], *, workers: int = 1
    ) -> tuple[str, dict[str, Any]]:
        """Build the artefact's text and a build report from ``config.yaml``."""

    @abstractmethod
    def holdout(self) -> list[Holdout]:
        """Measure the packaged file's errors on its frozen holdout."""

    @property
    def folder(self) -> Path:
        return ROOT / "fits" / self.name

    def config(self) -> dict[str, Any]:
        import yaml

        return yaml.safe_load((self.folder / "config.yaml").read_text())

    def provenance(self) -> dict[str, Any]:
        return json.loads((self.folder / "provenance.json").read_text())

    def packaged(self) -> Path:
        return DATA / self.provenance()["artefact"]

    def check_hash(self) -> list[str]:
        """Check that the packaged file is the one its provenance record describes."""
        record = self.provenance()
        path = DATA / record["artefact"]
        if not path.exists():
            return [f"{self.name}: {path.relative_to(ROOT)} is missing"]
        actual = sha256(path)
        if actual != record["sha256"]:
            return [
                f"{self.name}: {path.relative_to(ROOT)} has sha256 {actual[:16]}…, "
                f"its provenance record says {record['sha256'][:16]}…"
            ]
        return []

    def compare_rebuild(self, rebuilt: bytes) -> tuple[bool, bool, str]:
        """Compare a rebuilt artefact with the packaged file.

        Returns:
            Whether the bytes are identical, whether every number is, and a
            message saying which.
        """
        original = self.packaged().read_bytes()
        if rebuilt == original:
            return True, True, "rebuild is bit-identical to the packaged file"
        try:
            old, new = json.loads(original), json.loads(rebuilt)
        except json.JSONDecodeError:
            return False, False, "rebuild differs from the packaged file"
        old_numbers = _numbers({k: v for k, v in old.items() if k != "provenance"})
        new_numbers = _numbers({k: v for k, v in new.items() if k != "provenance"})
        if len(old_numbers) != len(new_numbers):
            return (
                False,
                False,
                (
                    f"rebuild has {len(new_numbers)} numbers, the packaged file "
                    f"{len(old_numbers)}"
                ),
            )
        pairs = list(zip(old_numbers, new_numbers, strict=True))
        same = sum(a == b for a, b in pairs)
        if same == len(pairs):
            return (
                False,
                True,
                (
                    "rebuild differs only in its embedded build record; every number "
                    "is bit-identical"
                ),
            )
        largest = max((abs(a - b) for a, b in pairs), default=0.0)
        return (
            False,
            False,
            (
                f"rebuild differs: {same}/{len(old_numbers)} numbers bit-identical, "
                f"largest absolute coefficient change {largest:.3g}"
            ),
        )

    def check(self, *, rebuild: bool | None = None, workers: int = 1) -> Report:
        """Hash, holdout and (when reproducible or asked) a bit-for-bit rebuild."""
        record = self.provenance()
        problems = self.check_hash()
        notes: list[str] = []
        holdouts = self.holdout()
        for h in holdouts:
            notes.append(
                f"holdout: {h.quantity} fitted {h.worst:.3g} over {h.cases} cases "
                f"(bound {h.bound:g}): {'pass' if h.passed else 'fail'}"
            )
        if not all(h.passed for h in holdouts):
            problems.append(f"{self.name}: a holdout error exceeds its bound")
        reproducible = record.get("reproducible", False)
        if reproducible and record.get("inputs") != build_inputs(self.name):
            problems.append(
                f"{self.name}: config.yaml, build.py or common.py changed since the "
                "packaged build; rebuild it"
            )
        # Bytes depend on the platform's libm as well as the pinned packages, so
        # a rebuild must match only in the environment that made the packaged file.
        same_environment = record.get("environment") == environment()
        if reproducible and not same_environment:
            notes.append(
                f"built in {record.get('environment')}; this is {environment()}, "
                "so a rebuild is compared for information only"
            )
        if rebuild is None:
            rebuild = reproducible
        if rebuild:
            text, _ = self.build(self.config(), workers=workers)
            identical, _, message = self.compare_rebuild(text.encode())
            notes.append(message)
            if reproducible and same_environment and not identical:
                problems.append(f"{self.name}: {message}")
        if not reproducible:
            notes.append(
                "not yet reproducible: " + record.get("note", "no builder record")
            )
        return Report(self.name, problems, notes, holdouts)


def _numbers(value: Any) -> list[float]:
    """Every number in a JSON value, in a fixed order."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [x for v in value for x in _numbers(v)]
    if isinstance(value, dict):
        return [x for k in sorted(value) for x in _numbers(value[k])]
    return []
