"""The one result record every measured check produces, and its tally.

Contract studies and comparisons with published results both report rows of
this shape, so every summary, failure list and coverage cell reads one format.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

#: Statuses, in the order a tally reports them.
STATUSES = ("pass", "fail", "note", "error")


def record(
    quantity: str,
    setting: str,
    value: float | None,
    bound: float | None,
    case: dict[str, Any],
    *,
    metric: str = "relative",
    error: str | None = None,
    note: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One observation of a catalogue quantity.

    Args:
        quantity: The catalogue key, with an aspect after a dot if needed.
        setting: ``"fitted"`` or ``"reference"``.
        value: The measured error, or None when it could not be measured.
        bound: The bound it is judged against, or None for a diagnostic.
        case: The inputs that identify the observation, including the truth's
            ``oracle`` or ``source`` where there is more than one.
        metric: How ``value`` is measured.
        error: Why the value could not be measured; makes the status "error".
        note: Why the row is informational; makes the status "note".
        extra: Further details kept with the row.
    """
    status = (
        "note"
        if note
        else "error"
        if error
        else "pass"
        if bound is None or (value is not None and value <= bound)
        else "fail"
    )
    return dict(
        quantity=quantity,
        set=setting,
        metric=metric,
        value=value,
        bound=bound,
        case=case,
        status=status,
        error=error,
        note=note,
        **extra,
    )


def failed(rows: Iterable[dict[str, Any]]) -> bool:
    return any(r["status"] in ("fail", "error") for r in rows)


def tally(name: str, rows: list[dict[str, Any]]) -> str:
    """One line per check: its name, row count and statuses."""
    counts = Counter(r["status"] for r in rows)
    statuses = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
    return f"{name:24s} {len(rows):6d} rows  {statuses or 'none'}"
