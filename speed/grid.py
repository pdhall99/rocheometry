"""The standard timing grid.

Each operation is timed on a freshly wired computer graph with the packaged
settings and cache capacities, so Python imports and graph construction are
outside the timed region. Every draw times a pair on the same graph: the cold
call, then the same operation at a new level for the same mass ratio. A third
call repeats the second and shows what a cached read costs. Results are
medians over draws.

Surface levels are prepared on an independent graph, so numerical fill
inversion never warms the graph being timed.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import pi
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import rocheometry
from rocheometry._accuracy import _Accuracy
from rocheometry._computers.composition import ComputerSet
from rocheometry._computers.defaults import _wire_computers

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
QIS = (1e-3, 1.0, 1e3, 1e6, 1e8)
SETTINGS = ("fitted", "reference")
REGIMES = ("detached", "contact")
OPERATIONS = (
    "selection",
    "lobe_radius",
    "lobe_slope",
    "sample",
    "volume",
    "fill",
    "section",
    "silhouette_60",
    "silhouette_85",
)
#: Operations without a surface level: their second call is a cached read.
_LEVEL_FREE = ("lobe_radius", "lobe_slope")
DRAWS = 5


@dataclass(frozen=True)
class _Wired(_Accuracy):
    graph: ComputerSet

    def _computers(self) -> ComputerSet:  # type: ignore[missing-override-decorator]
        return self.graph


def fresh(qi: float, setting: str) -> rocheometry.ComponentGeometry:
    """Return a component geometry on a newly wired, empty computer graph."""
    return rocheometry.ComponentGeometry(
        qi, accuracy=_Wired(_wire_computers(_accuracy(setting)._settings()))
    )


def _accuracy(setting: str) -> _Accuracy:
    return rocheometry.Fitted() if setting == "fitted" else rocheometry.Reference()


def operation(
    g: rocheometry.ComponentGeometry, name: str, level: Any
) -> Callable[[], Any]:
    """Return the timed call; ``level`` is a fill for selection, else a potential."""
    if name == "lobe_radius":
        return g.lobe_volume_radius
    if name == "lobe_slope":
        return g.lobe_volume_radius_slope
    if name == "selection":
        return lambda: g.surface(volume_radius_fill=level)
    surface = g.surface(potential=level)
    if name == "sample":
        return lambda: g._computers.sample(g.field, surface._depth)
    if name == "volume":
        return surface.volume
    if name == "fill":
        return surface.volume_radius_fill
    if name == "section":
        return surface.section
    if name == "silhouette_60":
        return lambda: surface.silhouette_envelope(rocheometry.CentrelinePlane(pi / 3))
    if name == "silhouette_85":
        return lambda: surface.silhouette_envelope(
            rocheometry.CentrelinePlane(17 * pi / 36)
        )
    raise ValueError(f"unknown operation {name!r}")


def _time(call: Callable[[], Any]) -> float:
    started = perf_counter()
    call()
    return 1000 * (perf_counter() - started)


def _levels(
    qi: float, setting: str, regime: str, count: int
) -> tuple[list[float], list[rocheometry.Potential]]:
    """Fill targets and potentials of ``count`` nearby levels in one regime."""
    prepare = fresh(qi, setting)
    if regime == "detached":
        surfaces = [
            prepare.surface(volume_radius_fill=0.8 + j * 1e-4) for j in range(count)
        ]
    else:
        surfaces = [
            prepare.surface(fill_out_factor=0.5 + j * 1e-4) for j in range(count)
        ]
    fills = [s.volume_radius_fill() for s in surfaces]
    if regime == "detached":
        fills = [0.8 + j * 1e-4 for j in range(count)]
    return fills, [s.potential for s in surfaces]


def measure(
    qi: float, setting: str, name: str, regime: str = "detached", *, draws: int = DRAWS
) -> list[dict[str, Any]]:
    """Paired cold/new-level timings and a cached repeat, as median rows."""
    fills, potentials = _levels(qi, setting, regime, draws + 1)
    levels = fills if name == "selection" else potentials
    cold: list[float] = []
    warm: list[float] = []
    cached: list[float] = []
    for j in range(draws):
        g = fresh(qi, setting)
        cold.append(_time(operation(g, name, levels[j])))
        second = operation(g, name, levels[j + 1])
        warm.append(_time(second))
        cached.append(_time(second))
    states = (("cold", cold), ("cached", cached))
    if name not in _LEVEL_FREE:
        states = (("cold", cold), ("new_level", warm), ("cached", cached))
    return [
        dict(
            set=setting,
            qi=qi,
            regime=regime,
            operation=name,
            state=state,
            ms=median(values),
            samples_ms=values,
        )
        for state, values in states
    ]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(
    *,
    qis: Iterable[float] = QIS,
    settings: Iterable[str] = SETTINGS,
    regimes: Iterable[str] = REGIMES,
    operations: Iterable[str] = OPERATIONS,
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Time the whole grid in this process; run it with nothing else competing."""
    started = perf_counter()
    rows = []
    for setting in settings:
        for qi in qis:
            for regime in regimes:
                for name in operations:
                    if name in _LEVEL_FREE and regime != "detached":
                        continue
                    result = measure(qi, setting, name, regime, draws=draws)
                    rows.extend(result)
                    logger.info(
                        "%-9s qi=%-8g %-8s %-14s %10.3f ms cold",
                        setting,
                        qi,
                        regime,
                        name,
                        result[0]["ms"],
                    )
    return dict(
        commit=_git("rev-parse", "HEAD"),
        environment=dict(
            platform=platform.platform(),
            python=sys.version,
            logical_cpus=os.cpu_count(),
        ),
        grid=dict(
            qis=list(qis),
            settings=list(settings),
            regimes=list(regimes),
            operations=list(operations),
            draws=draws,
            levels=(
                "detached: fill 0.8, 0.8001, ...; contact: F 0.5, 0.5001, ...; "
                "one new level per draw"
            ),
        ),
        seconds=perf_counter() - started,
        medians=rows,
    )
