"""Generate the speed sections of the docs from ``speed/results/speed.json``.

Generated text sits between ``<!-- generated:NAME -->`` and
``<!-- /generated:NAME -->`` markers in ``docs/site/guide/accuracy.md`` and
``README.md``. This module writes the ``timings`` and ``readme-speed``
sections; ``accuracy/report.py`` writes the others. ``check`` regenerates the
sections in memory and reports any difference.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "speed/results"
DOCUMENTS = (ROOT / "docs/site/guide/accuracy.md", ROOT / "README.md")
#: The published accuracy page: README links must also work on PyPI.
ACCURACY_PAGE = "https://pdhall99.github.io/rocheometry/guide/accuracy/"


def load() -> dict[str, Any]:
    """Return the published timings."""
    path = RESULTS / "speed.json"
    if not path.exists():
        raise FileNotFoundError(f"no published timings: {path}")
    return json.loads(path.read_text())


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", r"\|") for c in cells) + " |"


def section_timings(timings: dict[str, Any]) -> str:
    """Return the timing table of the guide's Speed section."""
    groups: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for r in timings["medians"]:
        key = (r["set"], r["qi"], r.get("regime", ""), r["operation"])
        groups[key][r["state"]] = r["ms"]
    environment = timings.get("environment", {})
    lines = [
        f"Medians of {timings['grid'].get('draws', 5)} draws in milliseconds on "
        f"{environment.get('platform', 'the recorded machine')}. Cold is a fresh "
        "geometry with empty caches; new level is a second level at the same mass ratio; "
        "cached repeats the new-level call.",
        "",
        "| Accuracy setting | qi | Regime | Operation | Cold ms | New level ms | Cached ms |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for (setting, qi, regime, name), values in sorted(groups.items()):
        lines.append(
            _row(
                [
                    setting,
                    f"{qi:g}",
                    regime,
                    name,
                    *(
                        "—" if values.get(s) is None else f"{values[s]:.6g}"
                        for s in ("cold", "new_level", "cached")
                    ),
                ]
            )
        )
    return "\n".join(lines)


def _range(
    timings: dict[str, Any], operation: str, state: str, regime: str = "detached"
) -> str:
    values = [
        r["ms"]
        for r in timings["medians"]
        if r["set"] == "fitted"
        and r["operation"] == operation
        and r["state"] == state
        and r.get("regime") == regime
        and r["ms"] is not None
    ]
    return f"{min(values):.2g}–{max(values):.2g}"


def section_readme_speed(timings: dict[str, Any]) -> str:
    """Return the README's paragraph on speed."""
    return "\n".join(
        (
            "Typical times with `Fitted()` for a detached surface, across the "
            "measured mass ratios:",
            "",
            "- selecting a surface: "
            f"**{_range(timings, 'selection', 'new_level')} ms**;",
            f"- measuring its volume: **{_range(timings, 'volume', 'new_level')} ms**;",
            f"- a section: **{_range(timings, 'section', 'new_level')} ms**;",
            "- a silhouette at 60° inclination: "
            f"**{_range(timings, 'silhouette_60', 'new_level')} ms**;",
            "- reading a volume already measured: "
            f"**{_range(timings, 'volume', 'cached')} ms**.",
            "",
            "These are medians on one machine.",
            f"[The timing table]({ACCURACY_PAGE}#speed) also gives the costs with "
            "`Reference()`, for contact surfaces, and of building a geometry for "
            "the first time.",
        )
    )


def sections() -> dict[str, str]:
    """Return every speed section, by marker name."""
    timings = load()
    return {
        "timings": section_timings(timings),
        "readme-speed": section_readme_speed(timings),
    }


_MARKER = re.compile(
    r"(<!-- generated:(?P<name>[\w-]+) -->)(?P<body>.*?)(<!-- /generated:(?P=name) -->)",
    re.S,
)


def write(*, check: bool = False) -> list[Path]:
    """Regenerate the speed sections; with ``check``, only list the files that differ."""
    generated = sections()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in generated:
            return match.group(0)
        return match.group(1) + "\n" + generated[name] + "\n" + match.group(4)

    stale = []
    for path in DOCUMENTS:
        text = path.read_text()
        updated = _MARKER.sub(replace, text)
        if updated != text:
            stale.append(path)
            if not check:
                path.write_text(updated)
    return stale
