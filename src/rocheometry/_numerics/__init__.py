"""Small dependency-free numerical helpers."""

from rocheometry._numerics.mesh import (
    INTERVAL_MESH_KINDS,
    IntervalMeshKind,
    interval_nodes,
)
from rocheometry._numerics.polynomial import (
    chebyshev_value,
    chebyshev_value_2d,
    power_value,
    power_value_2d,
)
from rocheometry._numerics.roots import bisect_root, brent_root, brent_root_adjacent

__all__ = [
    "INTERVAL_MESH_KINDS",
    "IntervalMeshKind",
    "bisect_root",
    "brent_root",
    "brent_root_adjacent",
    "chebyshev_value",
    "chebyshev_value_2d",
    "interval_nodes",
    "power_value",
    "power_value_2d",
]
