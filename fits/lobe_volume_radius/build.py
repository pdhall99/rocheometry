"""Build the lobe volume radius, ``src/rocheometry/_data/volume-radius-model.json``.

The runtime evaluates

    s = qi^(1/6) / (1 + qi^(1/6))
    F = Horner(4s - 1, low)  if qi <= 1  else  Horner(4s - 3, high)
    R = s^2 F

Training values of F come from ``Reference()`` at Chebyshev-Lobatto nodes in
s, and from the Hill and infinite-mass-ratio limits at the two ends. The fit is
a least-squares solve in a Chebyshev basis with exact equality constraints
(endpoint values and slopes, C2 at the join) and linear inequality constraints
that keep dR/ds >= 0 on a dense grid; SciPy's SLSQP solves it from the
equality-constrained least-squares start. The Chebyshev coefficients are then
converted to the power basis the runtime uses.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from math import cos, pi
from typing import Any

from fits.common import build_inputs, environment


def training_nodes(config: dict[str, Any]) -> list[float]:
    count = config["training_nodes"]
    return sorted(0.5 * (cos(pi - pi * i / (count - 1)) + 1) for i in range(count))


def _qi(v: float) -> float:
    return (v / (1 - v)) ** 6


def _radius(task: tuple[float, int, int]) -> float:
    from rocheometry import ComponentGeometry, Reference

    qi, polar, azimuth = task
    accuracy = Reference(polar_order=polar, azimuth_order=azimuth)
    return ComponentGeometry(qi, accuracy=accuracy).lobe_volume_radius()


def training_values(
    config: dict[str, Any], *, workers: int = 1
) -> tuple[list[float], list[float]]:
    """Nodes v and values F(v), from Reference() and the two limits."""
    h0, h1 = config["hill_limit"], config["infinite_limit"]
    lo, hi = config["reference_qi_min"], config["reference_qi_max"]
    order = config["reference"]["polar_order"], config["reference"]["azimuth_order"]
    nodes = [v for v in training_nodes(config) if v in (0.0, 1.0) or lo <= _qi(v) <= hi]
    interior = [v for v in nodes if 0.0 < v < 1.0]
    tasks = [(_qi(v), *order) for v in interior]
    if workers == 1:
        radii = [_radius(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            radii = list(pool.map(_radius, tasks))
    values = dict(
        zip(
            interior,
            (r / (v * v) for v, r in zip(interior, radii, strict=True)),
            strict=True,
        )
    )
    values[0.0], values[1.0] = h0, h1
    return nodes, [values[v] for v in nodes]


def fit(
    v: list[float], f: list[float], config: dict[str, Any]
) -> tuple[list[float], list[float]]:
    """Constrained least squares; returns power-basis low and high coefficients."""
    import numpy as np
    from numpy.polynomial import Chebyshev
    from numpy.polynomial.chebyshev import cheb2poly, chebvander
    from scipy.linalg import null_space
    from scipy.optimize import LinearConstraint, minimize

    nl, nh = config["low_degree"] + 1, config["high_degree"] + 1
    h0, h1 = config["hill_limit"], config["infinite_limit"]
    v_arr, f_arr = np.asarray(v), np.asarray(f)
    left, right = v_arr <= 0.5, v_arr >= 0.5
    dl = chebvander(4.0 * v_arr[left] - 1.0, nl - 1)
    dr = chebvander(4.0 * v_arr[right] - 3.0, nh - 1)
    design = np.zeros((dl.shape[0] + dr.shape[0], nl + nh))
    design[: dl.shape[0], :nl] = dl
    design[dl.shape[0] :, nl:] = dr
    target = np.concatenate((f_arr[left], f_arr[right]))

    def basis(count: int, order: int, x: float) -> np.ndarray:
        return np.asarray([Chebyshev.basis(i).deriv(order)(x) for i in range(count)])

    rows, values = [], []
    for part, order, x, value in (
        ("low", 0, -1.0, h0),
        ("low", 1, -1.0, 0.5 * h0),  # dF/dx = (dF/dv)/4 = 2 H0 / 4
        ("high", 0, 1.0, h1),
        ("high", 1, 1.0, -0.5 * h1),
    ):
        row = np.zeros(nl + nh)
        if part == "low":
            row[:nl] = basis(nl, order, x)
        else:
            row[nl:] = basis(nh, order, x)
        rows.append(row)
        values.append(value)
    for order in range(3):  # C2 at v = 1/2
        row = np.zeros(nl + nh)
        row[:nl] = basis(nl, order, 1.0)
        row[nl:] = -basis(nh, order, -1.0)
        rows.append(row)
        values.append(0.0)
    equalities, equality_values = np.vstack(rows), np.asarray(values)

    grid = np.linspace(1.0e-8, 1.0 - 1.0e-8, config["monotone_grid"])
    monotone = np.zeros((grid.size, nl + nh))
    low = grid <= 0.5
    for mask, offset, count, x in (
        (low, 0, nl, 4.0 * grid[low] - 1.0),
        (~low, nl, nh, 4.0 * grid[~low] - 3.0),
    ):
        b = chebvander(x, count - 1)
        d = np.column_stack([Chebyshev.basis(i).deriv()(x) for i in range(count)])
        s = grid[mask][:, None]
        monotone[mask, offset : offset + count] = 2.0 * s * b + 4.0 * s**2 * d

    particular = np.linalg.lstsq(equalities, equality_values, rcond=None)[0]
    null = null_space(equalities)
    reduced = np.linalg.lstsq(design @ null, target - design @ particular, rcond=None)[
        0
    ]
    start = particular + null @ reduced

    def objective(c: np.ndarray) -> float:
        r = design @ c - target
        return 0.5 * float(r @ r)

    def gradient(c: np.ndarray) -> np.ndarray:
        return design.T @ (design @ c - target)

    result = minimize(
        objective,
        start,
        jac=gradient,
        constraints=(
            LinearConstraint(equalities, equality_values, equality_values),
            LinearConstraint(monotone, 0.0, np.inf),
        ),
        method="SLSQP",
        options={"ftol": 1.0e-18, "maxiter": 2000},
    )
    c = np.asarray(result.x)
    if np.max(np.abs(equalities @ c - equality_values)) > 1.0e-10:
        raise RuntimeError("endpoint and join constraints are not satisfied")
    if np.min(monotone @ c) < -1.0e-12:
        raise RuntimeError("the fitted radius is not monotone")
    return (
        [float(x) for x in cheb2poly(c[:nl])],
        [float(x) for x in cheb2poly(c[nl:])],
    )


def build(config: dict[str, Any], *, workers: int = 1) -> tuple[str, dict[str, Any]]:
    """Return the artefact text and a build report."""
    from rocheometry._computers.lobe_volume_radius.fitted import (
        PiecewiseSixthRootHornerLobeVolumeRadiusComputer,
    )

    v, f = training_values(config, workers=workers)
    low, high = fit(v, f, config)
    model = PiecewiseSixthRootHornerLobeVolumeRadiusComputer(
        low_coefficients=tuple(low), high_coefficients=tuple(high)
    )
    # The artefact schema has no provenance field; the build record lives in
    # fits/lobe_volume_radius/provenance.json.
    return model.to_json(), dict(
        training_nodes=len(v),
        inputs=build_inputs("lobe_volume_radius"),
        environment=environment(),
    )
