"""Gravity-line watershed operations and their traced implementation."""

from typing import TYPE_CHECKING

from rocheometry._computers.watershed.protocol import WatershedComputer

if TYPE_CHECKING:
    from rocheometry._computers.watershed.traced import (
        TracedWatershedComputer,
        WatershedLine,
    )

__all__ = ["TracedWatershedComputer", "WatershedComputer", "WatershedLine"]


def __getattr__(name: str) -> object:
    if name in {"TracedWatershedComputer", "WatershedLine"}:
        from rocheometry._computers.watershed import traced

        return getattr(traced, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
