"""Classification of canonical Jacobi levels without solving a Roche field."""

from enum import StrEnum


class LevelRegime(StrEnum):
    """Topology of a Jacobi level relative to L1 and the first outer opening.

    Classification describes levels only. Surface selectors accept detached,
    critical and contact levels strictly above the outer opening.
    """

    DETACHED = "detached"
    CRITICAL = "critical"
    CONTACT = "contact"
    OPEN = "open"


def level_regime(depth: float, gap: float) -> LevelRegime:
    """Classify the exact sign of depth, bounded by the opening gap."""
    if depth < 0:
        return LevelRegime.DETACHED
    if depth == 0:
        return LevelRegime.CRITICAL
    if depth < gap:
        return LevelRegime.CONTACT
    return LevelRegime.OPEN
