"""The recipe figures: write them, and check them.

    python docs/figures.py write
    python docs/figures.py check

Each recipe listed in ``FIGURES`` draws a Matplotlib figure named ``fig``.
``write`` runs the recipe page's pycon examples, then saves that figure to
``docs/site/recipes/figures/``, so every figure comes from the code its page
shows. ``check`` renders every figure again and fails when one is missing or
stale, when its page does not embed it, or when a file there is not listed.
The files are byte-identical between runs with the same Matplotlib version;
after upgrading Matplotlib, run ``write`` and commit the changes.
"""

from __future__ import annotations

import argparse
import contextlib
import doctest
import io
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "docs/site/recipes"
OUTPUT = RECIPES / "figures"

#: Recipe page name: figure format. Line art is SVG; filled maps and meshes
#: are PNG, which stays small.
FIGURES = {
    "contour-map": "svg",
    "lobe-radii": "svg",
    "mesh": "png",
    "potential-diagram": "svg",
    "potential-well": "svg",
    "roche-lobe-diagram": "svg",
    "temperature-map": "png",
}

#: Settings that make the files byte-identical between runs.
DETERMINISTIC = {"svg.hashsalt": "rocheometry", "svg.fonttype": "path"}
METADATA = {"svg": {"Date": None, "Creator": None}, "png": {"Software": None}}


def render(name: str, file_format: str) -> bytes:
    """Run one recipe page's examples and return its figure's bytes."""
    text = (RECIPES / f"{name}.md").read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": f"recipe_{name}"}
    with matplotlib.rc_context(DETERMINISTIC), contextlib.redirect_stdout(None):
        for example in doctest.DocTestParser().get_examples(text, name):
            exec(compile(example.source, f"{name}.md", "exec"), namespace)  # noqa: S102
        fig = namespace["fig"]
        buffer = io.BytesIO()
        fig.savefig(  # type: ignore[attr-defined]
            buffer,
            format=file_format,
            dpi=150,
            facecolor="white",
            bbox_inches="tight",
            metadata=METADATA[file_format],
        )
    plt.close("all")
    return buffer.getvalue()


def _path(name: str) -> Path:
    return OUTPUT / f"{name}.{FIGURES[name]}"


def _page_problems() -> list[str]:
    """Pages that do not embed their figure, and files not listed in FIGURES."""
    problems = [
        f"{name}.md does not embed figures/{_path(name).name}"
        for name in sorted(FIGURES)
        if f"](figures/{_path(name).name})"
        not in (RECIPES / f"{name}.md").read_text(encoding="utf-8")
    ]
    listed = {_path(name).name for name in FIGURES}
    problems += [
        f"figures/{path.name} is not listed in FIGURES"
        for path in sorted(OUTPUT.glob("*"))
        if path.name not in listed
    ]
    return problems


def command_write(args: argparse.Namespace) -> int:
    """Render every figure and write it next to the recipes."""
    OUTPUT.mkdir(exist_ok=True)
    for name in sorted(FIGURES):
        logger.info("rendering %s", name)
        _path(name).write_bytes(render(name, FIGURES[name]))
        print(f"{_path(name).relative_to(ROOT)} written from {name}.md")
    problems = _page_problems()
    for problem in problems:
        print("FAIL", problem)
    return 1 if problems else 0


def command_check(args: argparse.Namespace) -> int:
    """Check that every figure matches its recipe and its page embeds it."""
    problems = _page_problems()
    for name in sorted(FIGURES):
        logger.info("rendering %s", name)
        path = _path(name)
        if not path.exists() or path.read_bytes() != render(name, FIGURES[name]):
            problems.append(
                f"figures/{path.name} is missing or stale: run docs:figures"
            )
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("the recipe figures match the recipes")
    return 1 if problems else 0


# --------------------------------------------------------------------------------------
# CLI
EPILOG = """\
exit codes:
  0    success
  1    failure (a figure is missing, stale or not embedded, or an unexpected error)
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
        prog="python docs/figures.py",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=help, parents=[logging_options])

    write = command("write", "render every figure and write it")
    write.set_defaults(handler=command_write)
    check = command("check", "compare the figures with their recipes")
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
