"""How the packaged fits are built: build them, and check them.

    python -m fits build ARTEFACT [--workers N] [--output FILE]
    python -m fits check [ARTEFACT ...] [--quick] [--rebuild | --no-rebuild]

``build`` writes a fresh artefact (by default into ``build/fits/``; use
``--output src/rocheometry/_data/...`` to replace the packaged file) and records
the build in the artefact's ``provenance.json`` when it replaces it. ``check``
verifies each packaged file against its provenance record, evaluates it on its
frozen holdout and, when the record says the build is reproducible (or with
``--rebuild``), rebuilds it and requires the bytes to match. ``check --quick``
compares the hashes only: the cheap check CI runs on every push.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from fits import common
from fits.fill_map.fit import FillMap
from fits.fit import Fit
from fits.lobe_volume_radius.fit import LobeVolumeRadius

logger = logging.getLogger(__name__)

#: Every packaged fit, by name.
FITS: dict[str, Fit] = {fit.name: fit for fit in (LobeVolumeRadius(), FillMap())}


def command_build(args: argparse.Namespace) -> int:
    name = args.artefact
    fit = FITS[name]
    text, report = fit.build(fit.config(), workers=args.workers)
    print(json.dumps(report, indent=2))
    _, same_numbers, message = fit.compare_rebuild(text.encode())
    record = fit.provenance()
    output = args.output or common.ROOT / "build/fits" / record["artefact"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    if output.resolve() != fit.packaged().resolve():
        print(f"wrote {output}; {message}")
        return 0
    record.update(
        sha256=common.sha256(output),
        reproducible=True,
        inputs=common.build_inputs(name),
        environment=common.environment(),
    )
    record.pop("note", None)
    path = fit.folder / "provenance.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"replaced {output.resolve().relative_to(common.ROOT)}; "
        f"updated {path.relative_to(common.ROOT)}; {message}"
    )
    if not same_numbers:
        print(
            "A new packaged fit changes Fitted() results: re-record the fingerprints "
            "and make a release run before the next release."
        )
    return 0


def command_check(args: argparse.Namespace) -> int:
    failed = False
    for name in args.artefacts or FITS:
        fit = FITS[name]
        if args.quick:
            problems = fit.check_hash()
            print(f"{name:24s} hash {'FAIL' if problems else 'ok'}")
            for line in problems:
                print("  FAIL " + line)
            failed |= bool(problems)
            continue
        report = fit.check(rebuild=args.rebuild, workers=args.workers)
        print(report.tally())
        for line in report.notes:
            print("  " + line)
        for line in report.problems:
            print("  FAIL " + line)
        failed |= not report.passed
    return 1 if failed else 0


# --------------------------------------------------------------------------------------
# CLI
EPILOG = """\
exit codes:
  0    success
  1    failure (a check or build failed, or an unexpected error)
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
        prog="python -m fits",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help, parents=[logging_options])

    build = command("build", "build one artefact")
    build.add_argument("artefact", choices=FITS)
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--output", type=Path)
    build.set_defaults(handler=command_build)
    check = command("check", "hash, holdout and rebuild checks")
    check.add_argument("artefacts", nargs="*", metavar="ARTEFACT", choices=FITS)
    rebuild = check.add_mutually_exclusive_group()
    rebuild.add_argument("--rebuild", action="store_true", default=None)
    rebuild.add_argument("--no-rebuild", dest="rebuild", action="store_false")
    check.add_argument("--workers", type=int, default=2)
    check.add_argument(
        "--quick", action="store_true", help="compare the packaged hashes only"
    )
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
