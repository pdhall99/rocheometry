"""Computer graphs, result checking, and explicit sharing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Generic, NamedTuple, ParamSpec, TypeVar

from rocheometry._computers.critical_point import CriticalPointComputer
from rocheometry._computers.lobe_slope import LobeVolumeRadiusSlopeComputer
from rocheometry._computers.mean_curvature import IntegratedMeanCurvatureComputer
from rocheometry._computers.mean_gravity import MeanGravityComputer
from rocheometry._computers.quadrature import SurfaceQuadratureComputer
from rocheometry._computers.section_path import SectionPathComputer
from rocheometry._computers.silhouette.protocol import SilhouettePathComputer
from rocheometry._computers.surface_area import SurfaceAreaComputer
from rocheometry._computers.surface_radius import SurfaceRadiusComputer
from rocheometry._computers.surface_sample import SurfaceSampler
from rocheometry._computers.surface_volume import SurfaceVolumeComputer
from rocheometry._computers.watershed.protocol import WatershedComputer
from rocheometry._validation import require_int

if TYPE_CHECKING:
    from functools import _lru_cache_wrapper

    from rocheometry._computers.surface_selection import SurfaceSelection

P = ParamSpec("P")
R = TypeVar("R")


class CacheInfo(NamedTuple):
    """LRU statistics for a memoized computer.

    Attributes:
        hits: Number of calls served from the cache.
        misses: Number of calls requiring computation.
        maxsize: Entry capacity, or `None` for an unbounded cache.
        currsize: Number of stored entries.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int


@dataclass(frozen=True, slots=True, eq=False)
class Memoized(Generic[P, R]):
    """An LRU result cache for a callable with hashable arguments.

    Sharing this instance shares results across fields and geometries. Failed
    calls are not cached. Concurrent first calls may compute the same result.
    Pickling preserves configuration, with an empty cache on reconstruction.

    Attributes:
        computer: Callable whose successful results are cached.
        maxsize: Non-negative entry capacity, or `None` for unbounded retention. Zero
            disables retention.
    """

    computer: Callable[P, R]
    maxsize: int | None = 4096
    _lru: _lru_cache_wrapper[R] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.computer):
            raise TypeError("computer: expected a callable")
        if self.maxsize is not None:
            require_int(self.maxsize, name="maxsize")
            if self.maxsize < 0:
                raise ValueError("maxsize: expected a non-negative integer or None")
        lru = lru_cache(maxsize=self.maxsize)(self.computer)
        object.__setattr__(self, "_lru", lru)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Return the cached result, or compute it on a cache miss."""
        return self._lru(*args, **kwargs)

    def cache_info(self) -> CacheInfo:
        """Return cache hits, misses, capacity and current size."""
        return CacheInfo(*self._lru.cache_info())

    def cache_clear(self) -> None:
        """Release cached results and reset the statistics."""
        self._lru.cache_clear()

    def __reduce__(self) -> tuple[object, ...]:
        """Reconstruct the configured computer with an empty cache."""
        return (Memoized, (self.computer, self.maxsize))


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ComputerSet:
    """Private immutable graph of injected strategies and shared result caches."""

    l1: CriticalPointComputer
    outer: CriticalPointComputer
    watershed: WatershedComputer
    radius: SurfaceRadiusComputer
    sample: SurfaceSampler
    volume: SurfaceVolumeComputer
    lobe_slope: LobeVolumeRadiusSlopeComputer
    area: SurfaceAreaComputer
    mean_curvature: IntegratedMeanCurvatureComputer
    mean_gravity: MeanGravityComputer
    quadrature: SurfaceQuadratureComputer
    selection: SurfaceSelection
    section: SectionPathComputer
    silhouette: SilhouettePathComputer
