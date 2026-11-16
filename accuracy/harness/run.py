"""Run the accuracy contract: every study, fresh, over its frozen domain.

A run writes its observation store (every row, compressed) to an output
directory. A release run also replaces the committed summaries in
``accuracy/results/``. No earlier observation or
checkpoint is read: each invocation measures everything it reports.

After a failed batch no further jobs are submitted, and a partial map is
labelled partial.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from accuracy.contract.continuity import Continuity
from accuracy.contract.critical_points import CriticalPoints
from accuracy.contract.guarantees import Guarantees
from accuracy.contract.independent_truth import IndependentTruth
from accuracy.contract.joins import Joins
from accuracy.contract.joint_holdout import JointHoldout
from accuracy.contract.outline_corners import OutlineCorners
from accuracy.contract.outline_holdouts import OutlineHoldouts
from accuracy.contract.outlines import Outlines
from accuracy.contract.quadrature import Quadrature
from accuracy.contract.radii import Radii
from accuracy.contract.saddle_quadrature import SaddleQuadrature
from accuracy.contract.scalars import Scalars
from accuracy.contract.selection import Selection
from accuracy.harness import code
from accuracy.harness.study import DerivedStudy, Study
from accuracy.literature.asymptotic_limits import AsymptoticLimits
from accuracy.literature.eggleton import Eggleton
from accuracy.literature.literature_tables import LiteratureTables
from accuracy.literature.seidov import Seidov

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "accuracy/results"
#: Every study, by name, in execution order: the contract, then the comparisons
#: with publications, then the studies derived from their rows.
REGISTRY: dict[str, Study | DerivedStudy] = {
    study.name: study
    for study in (
        CriticalPoints(),
        Radii(),
        Selection(),
        Scalars(),
        IndependentTruth(),
        SaddleQuadrature(),
        Quadrature(),
        Outlines(),
        Guarantees(),
        Joins(),
        OutlineCorners(),
        OutlineHoldouts(),
        JointHoldout(),
        LiteratureTables(),
        AsymptoticLimits(),
        Seidov(),
        Eggleton(),
        Continuity(),
    )
}
#: The names of each kind, in execution order.
CONTRACT = tuple(
    n for n, s in REGISTRY.items() if isinstance(s, Study) and s.kind == "contract"
)
LITERATURE = tuple(
    n for n, s in REGISTRY.items() if isinstance(s, Study) and s.kind == "literature"
)
STUDIES = CONTRACT + LITERATURE
DERIVED = tuple(n for n, s in REGISTRY.items() if isinstance(s, DerivedStudy))
#: Studies added or changed since the recorded release run: the committed
#: results have no rows from them, or rows from their earlier job list, until
#: the next release run.
UNRECORDED: tuple[str, ...] = (
    "literature_tables",
    "asymptotic_limits",
    "seidov",
    "eggleton",
)


def study(name: str) -> Study:
    """Return a measured study, contract or literature."""
    found = REGISTRY.get(name)
    if not isinstance(found, Study):
        raise ValueError(f"unknown measured study {name!r}")
    return found


def derived(name: str) -> DerivedStudy:
    """Return a study derived from other studies' rows."""
    found = REGISTRY.get(name)
    if not isinstance(found, DerivedStudy):
        raise ValueError(f"unknown derived study {name!r}")
    return found


def jobs(name: str) -> list[Any]:
    """Return the full job list of one measured study: its frozen domain."""
    return list(study(name).jobs())


def smoke_jobs(name: str) -> list[Any]:
    """One small job, enough to exercise the study; none while it is to do."""
    job = study(name).smoke()
    return [] if job is None else [job]


def source_tree() -> str:
    """Git's hash of the committed ``src/`` tree, or why there is none.

    It depends only on the files, not on history, so it identifies the measured
    library source after a squash merge or in a copy with different history:
    ``git rev-parse HEAD:src`` in any checkout gives the same hash when its
    ``src/`` holds the same files.
    """
    if _git("status", "--porcelain", "src"):
        return "modified"
    return _git("rev-parse", "HEAD:src") or "unknown"


def source_identity() -> dict[str, str]:
    """Hash every library and measurement source file with SHA-256.

    Comparing the hashes before and after a run detects edits made while it
    was in progress.
    """
    paths = [
        path
        for folder in ("src/rocheometry", "crosschecks", "accuracy")
        for path in sorted((ROOT / folder).rglob("*.py"))
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.name != "_version.py"
    }


def _initialize(output: str) -> None:
    os.environ["ROCHEOMETRY_STUDY_OUTPUT"] = output
    os.environ["ROCHEOMETRY_FRESH_STUDY"] = "1"


def _normalize(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from accuracy.contract.bounds import OUTLINE_MARGIN
    from accuracy.harness.common import OUTLINES
    from accuracy.harness.contracted import contracted_rows

    result = []
    for row in contracted_rows(name, rows):
        if (
            row["quantity"] in OUTLINES
            and row["set"] == "fitted"
            and row["bound"] is not None
            and row["status"] != "note"
        ):
            row = {**row, "public_contract": row["bound"], "bound": OUTLINE_MARGIN}
            if row["value"] is not None and not row.get("error"):
                row["status"] = "pass" if row["value"] <= OUTLINE_MARGIN else "fail"
        result.append(row)
    return result


def _case_of(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (float, int)):
        return dict(qi=value)
    first = value[0]
    return dict(qi=first if isinstance(first, (float, int)) else first[0], job=value)


def _measure(job: tuple[str, Any]) -> dict[str, Any]:
    name, value = job
    try:
        rows = _normalize(name, study(name).measure(value))
        return dict(
            rows=rows,
            failed=any(r["status"] in ("fail", "error") for r in rows),
            job=value,
        )
    except Exception as exc:
        import traceback

        error = f"{type(exc).__name__}: {exc}"
        return dict(
            rows=[
                dict(
                    quantity="harness." + name,
                    set="reference",
                    metric="execution",
                    value=None,
                    bound=0.0,
                    status="error",
                    case=_case_of(value),
                    error=error,
                    note=None,
                )
            ],
            failed=True,
            harness_error=True,
            job=value,
            error=error,
            traceback=traceback.format_exc(),
        )


def run_study(
    name: str,
    values: list[Any],
    output: Path,
    workers: int,
    *,
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    """Measure one study over its jobs and write its observations to ``output``."""
    from accuracy.harness.common import write_study

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(str(output),),
    ) as pool:
        todo = iter(enumerate(values))
        active = {}
        for _ in range(workers):
            item = next(todo, None)
            if item is not None:
                i, value = item
                active[pool.submit(_measure, (name, value))] = i
        failed = False
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                result = future.result()
                results.append(result)
                failed |= result["failed"]
                with (output / "progress.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(study=name, index=index, **result), allow_nan=False
                        )
                        + "\n"
                    )
                logger.info(
                    "%s job %d: %d rows%s",
                    name,
                    index,
                    len(result["rows"]),
                    ", failed" if result["failed"] else "",
                )
            if not (failed and stop_on_failure):
                for _ in done:
                    item = next(todo, None)
                    if item is not None:
                        i, value = item
                        active[pool.submit(_measure, (name, value))] = i
    rows = [row for result in results for row in result["rows"]]
    write_study(
        name,
        rows,
        time.perf_counter() - started,
        dict(
            fresh_run=True,
            resumed_jobs=0,
            requested_jobs=len(values),
            observed_jobs=len(results),
            complete=len(results) == len(values),
            workers=workers,
            grid=study(name).domain,
        ),
    )
    return dict(
        name=name,
        passed=not failed,
        complete=len(results) == len(values),
        observed_jobs=len(results),
        requested_jobs=len(values),
        harness_errors=[r for r in results if r.get("harness_error")],
        errors=[r.get("error") for r in results if r.get("error")],
    )


def run_derived(name: str) -> dict[str, Any]:
    """Run a derived study in this process, from the rows already recorded."""
    from accuracy.harness.common import read_rows, write_study

    started = time.perf_counter()
    try:
        rows = derived(name).derive(read_rows)
    except Exception as exc:
        import traceback

        return dict(
            name=name,
            passed=False,
            complete=False,
            harness_errors=[traceback.format_exc()],
            errors=[f"{type(exc).__name__}: {exc}"],
        )
    write_study(
        name, rows, time.perf_counter() - started, dict(grid=derived(name).domain)
    )
    observed = _normalize(name, read_rows(name))
    passed = all(row["status"] not in ("fail", "error") for row in observed)
    return dict(name=name, passed=passed, complete=True, harness_errors=[], errors=[])


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def store_files(output: Path) -> dict[str, str]:
    """Checksums of the observation store files that a release attaches."""
    names = sorted(
        p.name
        for p in output.iterdir()
        if p.is_file() and p.name not in ("progress.jsonl",)
    )
    return {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in names}


def publish(output: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> Path:
    """Replace the committed summaries in ``accuracy/results`` with a release run's.

    Git keeps the summaries of earlier release runs.
    """
    from accuracy.harness import accuracy_map

    accuracy_map.generate(output)
    target = RESULTS
    target.mkdir(parents=True, exist_ok=True)
    for name in ("accuracy.json", "failures.json"):
        shutil.copyfile(output / name, target / name)
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = dict(manifest, observation_store=store_files(output))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def run(
    *,
    output: Path,
    workers: int,
    studies: tuple[str, ...] = STUDIES,
    derived: tuple[str, ...] = DERIVED,
    select_jobs: Callable[[str], list[Any]] = jobs,
    release: bool = False,
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    """Run studies into ``output``; with ``release``, also publish results."""
    from accuracy.harness.cases import manifest as grid_manifest

    if release and _git("status", "--porcelain"):
        raise SystemExit("A release run requires a clean checkout")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; observations are never reused")
    output.mkdir(parents=True, exist_ok=True)
    _initialize(str(output))
    from accuracy.harness import common

    common.OUTPUT = output
    started = time.perf_counter()
    identity = source_identity()
    manifest = dict(
        commit=_git("rev-parse", "HEAD") or "unknown",
        source_tree=source_tree(),
        code_hash=code.code_hash(),
        grids=grid_manifest(),
        studies={name: len(select_jobs(name)) for name in studies},
        derived=list(derived),
        workers=workers,
        fresh_run=True,
        no_observations_reused=True,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    stopped = False
    for name in studies:
        result = run_study(
            name,
            select_jobs(name),
            output,
            workers,
            stop_on_failure=stop_on_failure,
        )
        results.append(result)
        if stop_on_failure and not (result["passed"] and result["complete"]):
            stopped = True
            break
    if not stopped:
        for name in derived:
            result = run_derived(name)
            results.append(result)
            if stop_on_failure and not result["passed"]:
                break
    if source_identity() != identity:
        raise RuntimeError("Source changed during the run")
    passed = all(r["passed"] and r["complete"] for r in results) and len(
        results
    ) == len(studies) + len(derived)
    summary = dict(
        passed=passed,
        complete=passed,
        commit=manifest["commit"],
        platform=platform.platform(),
        python=sys.version,
        logical_cpus=os.cpu_count(),
        workers=workers,
        seconds=time.perf_counter() - started,
        studies=[
            {k: v for k, v in r.items() if k != "harness_errors"} for r in results
        ],
        no_observations_reused=True,
        clean_checkout=release,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if release:
        target = publish(output, manifest, summary)
        logger.info("wrote %s", target.relative_to(ROOT))
    return dict(summary=summary, results=results)
