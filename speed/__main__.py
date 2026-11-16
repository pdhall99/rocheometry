"""How fast Rocheometry is: the timing grid.

    python -m speed run [--draws N]
    python -m speed sample [--quick] [--draws N] [--output FILE]
    python -m speed compare A B
    python -m speed profile OPERATION [--qi Q] [--setting S]
    python -m speed report
    python -m speed check

``run`` is the release timing: the whole grid, written to
``speed/results/speed.json``, then the speed sections of
the docs regenerated. The release procedure runs it on the recorded machine,
after ``accuracy run``. ``sample`` times the grid, or with ``--quick`` a small
slice of it, without publishing: it writes ``build/speed/<commit>.json``.
``compare`` accepts two timing files, or ``published`` for the published
timings, and prints the change in each cold and new-level median. ``profile``
runs cProfile on one call from the grid. ``report`` regenerates the speed
sections of the docs, and ``check`` compares them with the published timings.
Timing never gates CI: shared runners are too noisy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from speed import grid, report
from speed.profile import profile

logger = logging.getLogger(__name__)

RESULTS = report.RESULTS


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["set"],
        row["qi"],
        row.get("regime", ""),
        row["operation"],
        row["state"],
    )


def command_run(args: argparse.Namespace) -> int:
    """Time the whole grid, publish the timings and regenerate the docs."""
    result = grid.run(draws=args.draws)
    output = RESULTS / "speed.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output.relative_to(grid.ROOT)}")
    for path in report.write():
        print(f"{path.relative_to(grid.ROOT)} regenerated from the timings")
    return 0


def command_sample(args: argparse.Namespace) -> int:
    """Time the grid, or a quick slice of it, without publishing."""
    if args.quick:
        result = grid.run(qis=(1.0,), settings=("fitted",), draws=1)
    else:
        result = grid.run(draws=args.draws)
    output = args.output or grid.ROOT / "build/speed" / f"{result['commit'][:12]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    old = {_key(r): r["ms"] for r in json.loads(args.a.read_text())["medians"]}
    new = {_key(r): r["ms"] for r in json.loads(args.b.read_text())["medians"]}
    print(
        f"{'set':9s} {'qi':>8s} {'regime':8s} {'operation':14s} {'state':9s} "
        f"{'A ms':>10s} {'B ms':>10s} {'change':>8s}"
    )
    for key in sorted(old.keys() & new.keys()):
        if key[-1] == "cached":
            continue
        setting, qi, regime, name, state = key
        change = new[key] / old[key] - 1 if old[key] else float("nan")
        print(
            f"{setting:9s} {qi:8g} {regime:8s} {name:14s} {state:9s} "
            f"{old[key]:10.3f} {new[key]:10.3f} {change:+8.1%}"
        )
    missing = old.keys() ^ new.keys()
    if missing:
        logger.warning("%d medians are in only one file", len(missing))
    return 0


def command_report(args: argparse.Namespace) -> int:
    """Regenerate the speed sections of the docs from the published timings."""
    for path in report.write():
        print(
            f"{path.relative_to(grid.ROOT)} regenerated from speed/results/speed.json"
        )
    return 0


def command_check(args: argparse.Namespace) -> int:
    """Check that the published timings exist and the docs show them."""
    if not (RESULTS / "speed.json").exists():
        print("FAIL no published timings: run speed:run")
        return 1
    stale = report.write(check=True)
    for path in stale:
        print(f"FAIL {path.relative_to(grid.ROOT)} is not generated from the timings")
    if not stale:
        print("speed/results/speed.json and the docs match this checkout")
    return 1 if stale else 0


def command_profile(args: argparse.Namespace) -> int:
    profile(
        args.operation,
        qi=args.qi,
        setting=args.setting,
        regime=args.regime,
        state=args.state,
        sort=args.sort,
        limit=args.limit,
        output=args.output,
    )
    return 0


# --------------------------------------------------------------------------------------
# CLI
EPILOG = """\
exit codes:
  0    success
  1    failure (an unexpected error, or the docs do not match the timings)
  2    invalid input (bad arguments, or no such timing file)
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


def timing_file(value: str) -> Path:
    """Resolve a timing file, or ``published`` for the published timings."""
    path = RESULTS / "speed.json" if value == "published" else Path(value)
    if path.exists():
        return path
    raise argparse.ArgumentTypeError(f"no timing file {value!r}")


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
        prog="python -m speed",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help, parents=[logging_options])

    run = command("run", "the release timing: time, publish and regenerate the docs")
    run.add_argument("--draws", type=int, default=grid.DRAWS)
    run.set_defaults(handler=command_run)
    sample = command("sample", "time the grid without publishing")
    sample.add_argument("--output", type=Path)
    sample.add_argument("--draws", type=int, default=grid.DRAWS)
    sample.add_argument(
        "--quick", action="store_true", help="fitted only, qi=1, 1 draw"
    )
    sample.set_defaults(handler=command_sample)
    compare = command("compare", "compare two timing files, or one with `published`")
    compare.add_argument("a", type=timing_file, metavar="A")
    compare.add_argument("b", type=timing_file, metavar="B")
    compare.set_defaults(handler=command_compare)
    prof = command("profile", "cProfile one call")
    prof.add_argument("operation", choices=grid.OPERATIONS)
    prof.add_argument("--qi", type=float, default=1.0)
    prof.add_argument("--setting", choices=grid.SETTINGS, default="fitted")
    prof.add_argument("--regime", choices=grid.REGIMES, default="detached")
    prof.add_argument("--state", choices=("cold", "new_level"), default="cold")
    prof.add_argument("--sort", default="cumulative")
    prof.add_argument("--limit", type=int, default=40)
    prof.add_argument(
        "--output", type=Path, help="also write raw stats for snakeviz etc."
    )
    prof.set_defaults(handler=command_profile)
    regenerate = command("report", "regenerate the speed sections of the docs")
    regenerate.set_defaults(handler=command_report)
    check = command("check", "compare the docs with the published timings")
    check.set_defaults(handler=command_check)
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
