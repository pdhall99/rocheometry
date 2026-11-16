"""What a study is: the two base classes every study subclasses.

A study measures Rocheometry over frozen inputs. It names the catalogue keys it
measures, describes its inputs in words, lists them as jobs, picks one small
job for smoke runs, and measures one job at a time into rows made by
``accuracy.harness.record.record``. The runner sends jobs to worker processes
in any order, so a study holds configuration only, never results: anything
it caches belongs at module level (``functools.cache``), where it is
recomputed per process.

Every study is registered by name in ``accuracy.harness.run.REGISTRY``, and
lives in ``accuracy/contract/<name>.py`` or ``accuracy/literature/<name>.py``,
one study per file: the module docstring explains it, the class comes
straight after the imports and constants, and helpers follow as plain
functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal

#: One unit of work: a mass ratio, a batch of cases or a table, say. Jobs are
#: plain data, so they pickle to worker processes.
Job = Any
#: One result row, as ``accuracy.harness.record.record`` makes it.
Row = dict[str, Any]


class Study(ABC):
    """A measurement over frozen inputs, one job at a time."""

    #: As used in commands, results and the study's file name.
    name: ClassVar[str]
    #: Whether it measures the accuracy contract or compares with publications.
    kind: ClassVar[Literal["contract", "literature"]]
    #: The catalogue keys it measures (accuracy/catalogue.toml).
    quantities: ClassVar[tuple[str, ...]]
    #: Its inputs, in words, as the reports show them.
    domain: ClassVar[str]

    @abstractmethod
    def jobs(self) -> list[Job]:
        """Every job of the study: its frozen domain."""

    @abstractmethod
    def smoke(self) -> Job | None:
        """One small job that exercises the study, or None while it is to do."""

    @abstractmethod
    def measure(self, job: Job, /) -> list[Row]:
        """Measure one job and return its rows."""


class DerivedStudy(ABC):
    """A study computed from other studies' rows instead of from jobs."""

    name: ClassVar[str]
    kind: ClassVar[Literal["contract"]] = "contract"
    quantities: ClassVar[tuple[str, ...]]
    domain: ClassVar[str]

    @abstractmethod
    def derive(self, read: Callable[[str], list[Row]], /) -> list[Row]:
        """Rows computed from the rows ``read(name)`` returns for another study."""
