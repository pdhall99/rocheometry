"""Profile one call from the timing grid with cProfile (development only).

hatch run speed:profile silhouette_85 --qi 1e6 --setting reference
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from speed.grid import _levels, fresh, operation


def profile(
    name: str,
    *,
    qi: float,
    setting: str,
    regime: str,
    state: str,
    sort: str,
    limit: int,
    output: Path | None,
) -> None:
    """Profile one call and print its statistics; ``output`` also keeps the raw ones."""
    fills, potentials = _levels(qi, setting, regime, 2)
    levels = fills if name == "selection" else potentials
    g = fresh(qi, setting)
    if state == "new_level":
        operation(g, name, levels[0])()
    call = operation(g, name, levels[-1])
    profiler = cProfile.Profile()
    profiler.runcall(call)
    if output:
        profiler.dump_stats(output)
    pstats.Stats(profiler).strip_dirs().sort_stats(sort).print_stats(limit)
