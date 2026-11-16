"""Build the fill map, ``src/rocheometry/_data/surface-potential-map.json``.

The map stores a tensor of Chebyshev coefficients for the residual t of the
endpoint-constrained form

    fill * t(qi, fill) = fill * (1 + (t_inf - 1)(1 - fill^3) + (1 - fill) fill^3 R)

where R is the tensor in the compact fourth-root coordinate of qi and in
2 sqrt(1 - fill) - 1. The form is exact at fill = 0 and fill = 1, so the fit
only has to carry the interior. Coefficients come from one linear least-squares
solve; NumPy is used here and nowhere in the library.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import cos, pi
from typing import Any

from fits.common import DATA, ROOT, build_inputs, environment, sha256

Sample = tuple[float, float, float, float]  # qi, measured fill, potential, L1 potential


def training_jobs(config: dict[str, Any]) -> list[tuple[float, float]]:
    """Lobatto nodes in compact cube-root qi and in s = sqrt(1 - fill)."""
    qi_min, qi_max = config["qi_min"], config["qi_max"]
    lo = qi_min ** (1 / 3) / (1 + qi_min ** (1 / 3))
    hi = qi_max ** (1 / 3) / (1 + qi_max ** (1 / 3))
    jobs = []
    for i in range(config["qi_nodes"]):
        z = lo + (hi - lo) * (1 - cos(pi * i / (config["qi_nodes"] - 1))) / 2
        qi = (z / (1 - z)) ** 3
        if i == 0:
            qi = qi_min
        elif i == config["qi_nodes"] - 1:
            qi = qi_max
        for j in range(config["s_intervals"]):
            s = 0.5 * (1 - cos(pi * j / config["s_intervals"]))
            if s < 0.995:
                jobs.append((qi, 1 - s * s))
    return jobs


@lru_cache(maxsize=1)
def _graph(polar: int, azimuth: int) -> Any:
    from rocheometry import Reference

    return Reference(polar_order=polar, azimuth_order=azimuth)._computers()


def sample(task: tuple[float, float, int, int]) -> Sample:
    """Invert the Reference volume at one node, and measure the fill it has.

    Nothing here reads a packaged fit, so the build does not depend on its own
    previous output and a rebuild reproduces it.
    """
    from rocheometry._computers.surface_selection import VolumeInversionSelection
    from rocheometry._roche.field import RocheField

    qi, fill, polar, azimuth = task
    reference = _graph(polar, azimuth)
    field = RocheField(qi)
    l1 = reference.l1(field).potential
    if fill == 1.0:
        return (qi, 1.0, l1.value, l1.value)
    selection = reference.selection
    assert isinstance(selection, VolumeInversionSelection)
    potential = selection.depth(field, fill)
    actual = selection.fill(field, potential)
    return (
        qi,
        actual,
        field._level_value(reference.l1(field).position[0], potential),
        l1.value,
    )


def reference_samples(
    jobs: list[tuple[float, float]], config: dict[str, Any], *, workers: int = 1
) -> list[Sample]:
    polar = config["reference"]["polar_order"]
    azimuth = config["reference"]["azimuth_order"]
    tasks = [(qi, fill, polar, azimuth) for qi, fill in jobs]
    if workers == 1:
        return [sample(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(sample, tasks, chunksize=config["s_intervals"]))


def fit(samples: list[Sample], config: dict[str, Any]) -> list[list[float]]:
    """Least-squares Chebyshev tensor of the interior residual."""
    import numpy as np
    from numpy.polynomial import chebyshev as C

    from rocheometry._computers.packaged import packaged_lobe_volume_radius_computer
    from rocheometry._roche.field import RocheField

    rows = [s for s in samples if s[1] < 1.0]
    qd, sd = config["qi_degree"], config["s_degree"]
    if len(rows) < (qd + 1) * (sd + 1):
        raise ValueError("not enough noncritical samples for the requested tensor")
    qi, fill, u, u1 = np.asarray(rows).T
    c = 1 / (1 + qi)
    m = qi * c
    u0 = c + c * c / 2
    t = (u1 - u0) / ((u - u0) * fill)
    radius = packaged_lobe_volume_radius_computer()
    radii = np.asarray([radius(RocheField(float(q))) for q in qi])
    tinf = (u1 - u0) * radii / m
    s = np.sqrt(1 - fill)
    qi_min, qi_max = config["qi_min"], config["qi_max"]
    low = qi_min**0.25 / (1 + qi_min**0.25)
    high = qi_max**0.25 / (1 + qi_max**0.25)
    z = 2 * (qi**0.25 / (1 + qi**0.25) - low) / (high - low) - 1
    tensor = (
        np.asarray(C.chebvander(z, qd), dtype=float)[:, :, None]
        * np.asarray(C.chebvander(2 * s - 1, sd), dtype=float)[:, None, :]
    ).reshape(len(rows), -1)
    matrix = tensor * ((1 - fill) * fill**3)[:, None]
    target = t - (1 + (tinf - 1) * (1 - fill**3))
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError("training matrix is rank deficient")
    return [[float(v) for v in row] for row in coefficients.reshape(qd + 1, sd + 1)]


def build(config: dict[str, Any], *, workers: int = 1) -> tuple[str, dict[str, Any]]:
    """Return the artefact text and a build report."""
    from rocheometry._computers.surface_potential.fitted.computer import (
        SurfacePotentialMap,
    )

    samples = reference_samples(training_jobs(config), config, workers=workers)
    coefficients = fit(samples, config)
    lobe = DATA / "volume-radius-model.json"
    provenance = dict(
        builder="fits/fill_map/build.py",
        inputs=build_inputs("fill_map"),
        lobe_volume_radius=dict(path=str(lobe.relative_to(ROOT)), sha256=sha256(lobe)),
        training_samples=len(samples),
        environment=environment(),
    )
    model = SurfacePotentialMap(
        tuple(tuple(row) for row in coefficients),
        qi_min=config["qi_min"],
        qi_max=config["qi_max"],
        provenance=provenance,
    )
    return model.to_json(), dict(training_samples=len(samples))
