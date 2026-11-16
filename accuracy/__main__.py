"""How accurate Rocheometry is: the accuracy contract and the literature.

    python -m accuracy run [--workers N] [--output DIR]
    python -m accuracy smoke [--workers N]
    python -m accuracy sample [--every K] [--workers N] [--output DIR]
    python -m accuracy study NAME [NAME ...] [--workers N] [--output DIR]
    python -m accuracy check [--release] [--stores DIR]
    python -m accuracy report

``run`` is the release run: every study (contract, then literature, then
derived), fresh, from a clean checkout; it writes the observation store to
a new ``build/accuracy/release-<time>/``, replaces the committed summaries in
``accuracy/results/`` and regenerates the docs. ``smoke``
runs one small job of every study to check the machinery, not the numbers.
``sample`` runs every K-th job of every study, the reduced run made after
each change to the source or the studies. ``study`` runs chosen studies over
their full domain. None of these three publishes: each writes its store to a
new ``build/accuracy/<verb>-<time>/`` unless given ``--output``.
``check`` compares what is committed (results, generated docs, the
catalogue) with this checkout; with ``--release`` it also requires that the
release run measured this checkout's code, ignoring comments, formatting
and docstrings (see ``accuracy/harness/code.py``). ``report`` regenerates the
generated docs.
Every command prints one tally line per study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from accuracy import report
from accuracy.harness import accuracy_map, code, coverage, run
from accuracy.harness.common import read_rows
from accuracy.harness.record import tally

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _scratch(verb: str) -> Path:
    """Name a new folder for an unpublished run: observations are never reused."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return ROOT / "build/accuracy" / f"{verb}-{stamp}"


def _tallies(result: dict[str, Any], output: Path) -> None:
    """One line per study: rows and statuses, or why there are none."""
    for r in result["results"]:
        if r.get("harness_errors"):
            print(f"{r['name']:24s} harness error")
            for error in r["harness_errors"]:
                print(
                    error.get("traceback", error) if isinstance(error, dict) else error
                )
        elif r.get("requested_jobs") == 0:
            print(f"{r['name']:24s} to do")
        else:
            print(tally(r["name"], read_rows(r["name"], output)))


def _regenerate() -> None:
    for path in report.write():
        print(f"{path.relative_to(ROOT)} regenerated from accuracy/results/")
    # The results are already written; report catalogue problems rather than
    # lose the run.
    for problem in coverage.load()[1]:
        print("FAIL", problem, "(fix accuracy/catalogue.toml)")


def command_run(args: argparse.Namespace) -> int:
    output = args.output or _scratch("release")
    result = run.run(output=output, workers=args.workers, release=True)
    _tallies(result, output)
    if not result["summary"]["passed"]:
        print("STOP: the release run failed; see", output / "summary.json")
        return 1
    _regenerate()
    print(f"release run passed; results in {run.RESULTS.relative_to(ROOT)}")
    print(f"attach the observation store in {output} to the release")
    print("next: hatch run speed:run, on the same machine")
    return 0


def command_study(args: argparse.Namespace) -> int:
    output = args.output or _scratch("study")
    result = run.run(
        output=output,
        workers=args.workers,
        studies=tuple(n for n in run.STUDIES if n in args.names),
        derived=tuple(n for n in run.DERIVED if n in args.names),
        stop_on_failure=False,
    )
    _tallies(result, output)
    print(f"observations in {output}")
    return 0 if result["summary"]["passed"] else 1


def command_smoke(args: argparse.Namespace) -> int:
    """One small job of every study: harness errors fail, accuracy misses do not."""
    output = _scratch("smoke")
    result = run.run(
        output=output,
        workers=args.workers,
        select_jobs=run.smoke_jobs,
        stop_on_failure=False,
    )
    _tallies(result, output)
    if len(result["results"]) != len(run.STUDIES) + len(run.DERIVED):
        print("not every study ran")
        return 1
    return 1 if any(r["harness_errors"] for r in result["results"]) else 0


def command_sample(args: argparse.Namespace) -> int:
    """Every K-th job of every study: reports accuracy, fails only on harness errors."""
    output = args.output or _scratch("sample")
    result = run.run(
        output=output,
        workers=args.workers,
        select_jobs=lambda name: run.jobs(name)[:: args.every],
        stop_on_failure=False,
    )
    _tallies(result, output)
    accuracy_map.generate(output)
    failures = json.loads((output / "failures.json").read_text())
    print(f"{failures['count']} contract misses in the sample; store in {output}")
    for row in failures["failures"][:20]:
        print(
            f"  {row['quantity']} {row['set']} {row['value']} > {row['bound']} {row['case']}"
        )
    return 1 if any(r["harness_errors"] for r in result["results"]) else 0


def command_report(args: argparse.Namespace) -> int:
    """Regenerate the docs generated from the results and the catalogue."""
    entries, problems = coverage.load()
    for problem in problems:
        print("FAIL", problem)
    if problems:
        return 1
    if coverage.recorded():
        _regenerate()
    gaps = sum(c.state == "gap" for e in entries for c in e.cells.values())
    print(f"{len(entries)} quantities, {gaps} gaps")
    return 0


def _release_problems(manifest: dict[str, Any]) -> list[str]:
    """Whether the release run measured this checkout's code.

    Runs before code hashes were recorded are compared by git's tree hash,
    which changes with any edit to ``src/``.
    """
    if "code_hash" in manifest:
        recorded, current, kind = manifest["code_hash"], code.code_hash(), "code"
    else:
        recorded, current, kind = manifest["source_tree"], run.source_tree(), "tree"
    if recorded == current:
        return []
    return [
        f"the release run measured src/ {kind} {recorded[:12]}, but this "
        f"checkout's is {current[:12]}: make a release run"
    ]


def command_check(args: argparse.Namespace) -> int:
    """Check that what is committed describes this checkout."""
    problems = coverage.load()[1]
    if not coverage.recorded():
        problems.append("no release run recorded: make the release run")
    else:
        data = report.load()
        manifest, summary = data["manifest"], data["summary"]
        if args.release:
            problems += _release_problems(manifest)
        if not summary["passed"]:
            problems.append("the recorded release run did not pass")
        if data["failures"]["count"]:
            problems.append(f"{data['failures']['count']} contract failures recorded")
        problems += [
            f"{p.relative_to(ROOT)} is not generated from the results"
            for p in report.write(check=True)
        ]
        if args.stores:
            for name, digest in manifest["observation_store"].items():
                path = args.stores / name
                if not path.exists():
                    problems.append(f"store file {name} missing")
                elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    problems.append(f"store file {name} checksum differs")
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("accuracy/results/ and the docs match this checkout")
    return 1 if problems else 0


# --------------------------------------------------------------------------------------
# CLI
EPILOG = """\
exit codes:
  0    success
  1    failure (a study, check or report failed, or an unexpected error)
  2    invalid input (bad arguments)
  130  interrupted by user (Ctrl-C)
"""


def set_up_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level: progress is shown unless quiet, debug with -v."""
    if quiet:
        level = logging.WARNING
    else:
        level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)s: %(message)s", stream=sys.stderr, force=True
    )
    logging.captureWarnings(capture=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    # Every command takes -v/-q, after its name as the Hatch scripts pass them.
    logging_options = argparse.ArgumentParser(add_help=False)
    verbosity = logging_options.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also show debug output",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="hide progress; show warnings and errors",
    )
    parser = argparse.ArgumentParser(
        prog="python -m accuracy",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help, parents=[logging_options])

    release = command("run", "the release run")
    release.add_argument("--workers", type=int, default=8)
    release.add_argument("--output", type=Path)
    release.set_defaults(handler=command_run)
    smoke = command("smoke", "one small job of every study")
    smoke.add_argument("--workers", type=int, default=2)
    smoke.set_defaults(handler=command_smoke)
    sample = command("sample", "every K-th job (reduced run)")
    sample.add_argument("--every", type=int, default=25)
    sample.add_argument("--workers", type=int, default=4)
    sample.add_argument("--output", type=Path)
    sample.set_defaults(handler=command_sample)
    study = command("study", "chosen studies over their full domain")
    study.add_argument(
        "names", nargs="+", metavar="NAME", choices=run.STUDIES + run.DERIVED
    )
    study.add_argument("--workers", type=int, default=8)
    study.add_argument("--output", type=Path)
    study.set_defaults(handler=command_study)
    check = command(
        "check", "compare the committed results and docs with this checkout"
    )
    check.add_argument(
        "--release",
        action="store_true",
        help="also require that the release run measured this checkout's src/",
    )
    check.add_argument("--stores", type=Path, help="directory of downloaded stores")
    check.set_defaults(handler=command_check)
    regenerate = command("report", "regenerate the generated docs")
    regenerate.set_defaults(handler=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    try:
        args = build_parser().parse_args(argv)
        set_up_logging(verbose=args.verbose, quiet=args.quiet)
        return args.handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
