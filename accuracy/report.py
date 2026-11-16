"""Generate the accuracy sections of the docs from ``accuracy/results/``.

Generated text sits between ``<!-- generated:NAME -->`` and
``<!-- /generated:NAME -->`` markers in ``docs/site/guide/accuracy.md`` and ``README.md``.
Everything outside the markers is written by hand, and ``speed/report.py``
writes the timing sections. ``check`` regenerates the
sections in memory and reports any difference, so CI keeps the docs and the
committed results consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "accuracy/results"
DOCUMENTS = (ROOT / "docs/site/guide/accuracy.md", ROOT / "README.md")
#: The published accuracy page: README links must also work on PyPI.
ACCURACY_PAGE = "https://pdhall99.github.io/rocheometry/guide/accuracy/"


def load() -> dict[str, Any]:
    if not (RESULTS / "manifest.json").exists():
        raise FileNotFoundError(f"no release run recorded in {RESULTS}")
    return {
        name: json.loads((RESULTS / f"{name}.json").read_text())
        for name in ("summary", "accuracy", "failures", "manifest")
    }


def _short(value: float) -> str:
    """Compact scientific form for bounds: 10<sup>−6</sup>, 2.5 × 10<sup>−4</sup>."""
    if value == 0 or 1e-3 <= abs(value) < 1e4:
        return f"{value:g}"
    mantissa, exponent = f"{value:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exponent)
    exp_str = f"−{abs(exp)}" if exp < 0 else str(exp)
    power = f"10<sup>{exp_str}</sup>"
    return power if mantissa == "1" else f"{mantissa} × {power}"


def _number(value: Any, digits: int = 12) -> str:
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) else str(value)


def _cell(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def diagnostic_explanation(row: dict[str, Any]) -> str:
    """Why a measured quantity is a diagnostic rather than a contract."""
    name = row["quantity"]
    if "eigenvalues_relative" in name:
        return (
            "Individual near-zero eigenvalues are ill-conditioned; the contract "
            "uses the Hessian spectral norm."
        )
    if name.endswith(("normal_angle", "near_normal_angle")):
        return (
            "Forward node-normal error is ill-conditioned near saddles; running "
            "backward bounds determine acceptance."
        )
    if name.endswith(("displacement", "near_displacement")):
        return (
            "Forward node displacement is diagnostic; acceptance uses the "
            "exact-ray residual running bound."
        )
    if "gauss_bonnet" in name:
        return (
            "Integrated Gaussian curvature diagnoses topology and apex "
            "conditioning; the critical guard is separate."
        )
    if "second_difference" in name:
        return (
            "Sampled second differences describe continuity on the recorded "
            "grid, without a universal bound."
        )
    if name.endswith("knots_normal"):
        return (
            "Normal displacement of a fitted knot near a neck is unbounded; its "
            "radial guarantee is checked separately."
        )
    if name.endswith("corner_rounding"):
        return (
            "Corner displacement is included in the two-way outline bound; it "
            "has no separate contract."
        )
    notes = " ".join(row.get("notes", []))
    return notes or f"{row['metric']}; recorded for diagnosis, without a public bound."


def section_summary(data: dict[str, Any]) -> str:
    summary, manifest = data["summary"], data["manifest"]
    rows = summary.get("verification", {}).get("recorded_rows")
    lines = [
        "The release run measured commit "
        f"`{manifest['commit'][:12]}`"
        + (
            f" (source tree `{manifest['source_tree'][:12]}`)"
            if "source_tree" in manifest
            else ""
        )
        + f" on {summary['platform']} in {summary['seconds'] / 3600:.1f} h with "
        f"{manifest['workers']} workers.",
        (
            f"It recorded {rows:,} observations; the failure list has "
            if rows
            else "The failure list has "
        )
        + f"**{data['failures']['count']}** entries.",
        "Summaries: [`accuracy/results/`](https://github.com/pdhall99/rocheometry/tree/main/accuracy/results).",
    ]
    return "\n".join(lines)


def section_bounds() -> str:
    from accuracy.contract.bounds import (
        FILL_BOUND,
        OUTLINE_MARGIN,
        PATH_BOUND,
        REFERENCE_CURVATURE_BOUND,
        SCALAR_BOUND,
    )

    lines = [
        "| Quantity | Fitted | Reference |",
        "| --- | ---: | ---: |",
        "| Integrated mean curvature and mean-curvature radius | "
        f"{_short(SCALAR_BOUND['fitted'])} relative | {_short(REFERENCE_CURVATURE_BOUND)} "
        "relative at default orders |",
        f"| Other measured scalars | {_short(SCALAR_BOUND['fitted'])} relative | "
        f"{_short(SCALAR_BOUND['reference'])} relative |",
        f"| Selection by volume-radius fill | {_short(FILL_BOUND)} absolute fill | "
        f"{_short(1e-13)} absolute fill |",
        f"| Sections and silhouettes | {_short(PATH_BOUND['fitted'])} R_vol, two-way "
        f"normal distance | {_short(PATH_BOUND['reference'])} R_vol, two-way normal "
        "distance |",
        f"| Lobe-radius logarithmic slope | {_short(SCALAR_BOUND['fitted'])} absolute | "
        f"{_short(SCALAR_BOUND['reference'])} absolute |",
        "",
        f"Fitted outlines are accepted at **{_short(OUTLINE_MARGIN)} R_vol** in the "
        "release run.",
    ]
    return "\n".join(lines)


def _published(row: dict[str, Any]) -> bool:
    from accuracy.harness.run import LITERATURE

    return row["study"] in LITERATURE


def section_contract(data: dict[str, Any]) -> str:
    lines = [
        "| Quantity | Accuracy setting | Study | Metric / units | Bound | Worst measured | Cases |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in data["accuracy"]["quantities"]:
        if row["contracted_bound"] is None or _published(row):
            continue
        lines.append(
            _row(
                [
                    row["quantity"],
                    row["set"],
                    row["study"],
                    row["metric"],
                    _number(row["contracted_bound"]),
                    _number(row["measured_worst"]),
                    str(row["domain"]["contracted_cases"]),
                ]
            )
        )
    return "\n".join(lines)


def section_literature(data: dict[str, Any]) -> str:
    """One line per quantity: its comparison closest to its own bound.

    Bounds differ between tables (a source's rounding and stated accuracy),
    so the largest difference alone can mislead; the ratio is what passes.
    """
    lines = [
        "| Quantity | Accuracy setting | Metric / units | Cases | Closest to its bound | Value / bound | Source |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in data["accuracy"]["quantities"]:
        if not _published(row):
            continue
        closest = row.get("closest") or dict(
            value=row["measured_worst"],
            bound=row["contracted_bound"],
            case=row["worst_case"],
        )
        lines.append(
            _row(
                [
                    row["quantity"],
                    row["set"],
                    row["metric"],
                    str(row["domain"]["cases"]),
                    f"{_number(closest['value'], 3)} of {_number(closest['bound'], 3)}",
                    f"{closest['value'] / closest['bound']:.2f}",
                    str(closest["case"].get("source", row["study"])),
                ]
            )
        )
    return "\n".join(lines)


def section_diagnostics(data: dict[str, Any]) -> str:
    lines = [
        "| Quantity | Accuracy setting | Study | Metric / units | Worst measured | Explanation |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in data["accuracy"]["quantities"]:
        if row["contracted_bound"] is not None or _published(row):
            continue
        lines.append(
            _row(
                [
                    row["quantity"],
                    row["set"],
                    row["study"],
                    row["metric"],
                    _number(row["measured_worst"]),
                    diagnostic_explanation(row),
                ]
            )
        )
    return "\n".join(lines)


def section_readme_contract() -> str:
    from accuracy.contract.bounds import (
        FILL_BOUND,
        PATH_BOUND,
        REFERENCE_CURVATURE_BOUND,
        SCALAR_BOUND,
    )

    reference = _short(SCALAR_BOUND["reference"])
    return "\n".join(
        (
            "Rocheometry's accuracy is measured for component mass ratios from "
            f"{_short(1e-8)} to {_short(1e8)}, at every level above L2.",
            "",
            "With the default `Fitted()`:",
            "",
            "- measured quantities, such as volumes, areas and radii, have relative "
            f"errors of at most **{_short(SCALAR_BOUND['fitted'])}**;",
            "- a surface selected by its fill has the requested fill to within "
            f"**{_short(FILL_BOUND)}**;",
            "- sections and silhouettes lie within "
            f"**{_short(PATH_BOUND['fitted'])}** of the true outline, in units of the "
            "surface's volume radius.",
            "",
            f"With `Reference()`, quantities and outlines are within **{reference}**.",
            "The exceptions are the integrated mean curvature and mean-curvature "
            f"radius, within **{_short(REFERENCE_CURVATURE_BOUND)}** at the default "
            f"orders and **{reference}** at `Reference(polar_order=512, "
            "azimuth_order=192)`.",
            "",
            f"The [accuracy page]({ACCURACY_PAGE}) gives every bound and the "
            "measurements behind it.",
        )
    )


def sections() -> dict[str, str]:
    data = load()
    return {
        "summary": section_summary(data),
        "bounds": section_bounds(),
        "contract": section_contract(data),
        "literature": section_literature(data),
        "diagnostics": section_diagnostics(data),
        "readme-contract": section_readme_contract(),
    }


_MARKER = re.compile(
    r"(<!-- generated:(?P<name>[\w-]+) -->)(?P<body>.*?)(<!-- /generated:(?P=name) -->)",
    re.S,
)


def render(text: str, generated: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in generated:
            # Another tool's section: speed/report.py writes the timings.
            return match.group(0)
        return match.group(1) + "\n" + generated[name] + "\n" + match.group(4)

    return _MARKER.sub(replace, text)


def write(*, check: bool = False) -> list[Path]:
    """Regenerate the docs; with ``check``, only list the files that differ."""
    generated = sections()
    stale = []
    for path in DOCUMENTS:
        text = path.read_text()
        updated = render(text, generated)
        if updated != text:
            stale.append(path)
            if not check:
                path.write_text(updated)
    return stale
