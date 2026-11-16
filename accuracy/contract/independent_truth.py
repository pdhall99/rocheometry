"""Contract quantities against truth that shares no integrator with the library.

The scalars study measures Fitted() against Reference(), and Reference()
against itself at higher orders: for the lobe, the volume integrals and the
smooth quadrature sums, that is self-convergence, which cannot see an error
the method makes at every order. This study measures both settings against
independent methods instead, at every decade of the mass ratio:

* the lobe radius, by Cartesian x-slices of the critical lobe
  (``crosschecks.cartesian``);
* the lobe slope, by Richardson-extrapolated central differences of those
  radii in ln qi;
* volume, area, integrated mean curvature and mean gravity of detached
  surfaces up to fill 0.9, the measured fill, and the quadrature's smooth
  sums, by a plain tensor rule written from the implicit-surface formulas
  (``crosschecks.integrals``);
* both components' volumes at the L2 level, together, by Cartesian slices
  of the whole body, which needs no dividing surface;
* the maximum fill, from that envelope less the companion's Reference volume
  and over the Cartesian lobe volume. The split between the components is
  the one thing here taken from the library; the sum is checked on its own,
  and the companion's volume is treated as exact in the truth's error, so a
  light component's truth error is a lower bound. At equal masses the split
  is exact by symmetry, and a test checks that case with nothing shared.

A truth is used only where its own error estimate is at most a tenth of the
bound. Elsewhere the row is a note, so an unresolved truth never passes or
fails a case. The tensor rule is not used on contact or near-critical
surfaces, whose necks it does not resolve.
"""

from __future__ import annotations

from math import fsum, pi
from typing import Any

from crosschecks import cartesian
from crosschecks.integrals import surface_integrals

import rocheometry
from accuracy.contract.bounds import SCALAR_BOUND, scalar_bound
from accuracy.harness.common import (
    FILLS,
    INTEGRALS,
    clear_samples,
    decade_jobs,
    relative,
    select,
)
from accuracy.harness.record import record
from accuracy.harness.study import Study
from rocheometry._levels import _levels

#: Detached fills on which the tensor rule converges spectrally.
SMOOTH_FILLS = tuple(f for f in FILLS if f <= 0.9)
#: The largest fraction of the bound a truth's own error may be.
TRUTH_MARGIN = 0.1
#: Surface readers measured against the tensor rule.
READERS = ("volume", "area", "integrated_mean_curvature", "mean_gravity")


class IndependentTruth(Study):
    name = "independent_truth"
    kind = "contract"
    quantities = (
        "lobe.volume_radius",
        "lobe.slope",
        "lobe.max_fill",
        "integrals.volume",
        "integrals.fill",
        "integrals.area",
        "integrals.mean_curvature",
        "integrals.mean_gravity",
        "quadrature.integrals",
    )
    domain = "The 17 decades of qi; detached fills up to 0.9 for the surface integrals, and the L2 level."

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, job: float) -> list[dict[str, Any]]:
        """Measure one mass ratio against every independent truth."""
        qi = job
        settings = dict(
            fitted=rocheometry.ComponentGeometry(qi),
            reference=rocheometry.ComponentGeometry(
                qi, accuracy=rocheometry.Reference()
            ),
        )
        try:
            rows, lobe = _lobe(qi, settings)
        except Exception as exc:
            return [_failed("lobe.volume_radius", dict(qi=qi, oracle="cartesian"), exc)]
        try:
            rows += _l2(qi, lobe)
        except Exception as exc:
            rows.append(
                _failed(
                    "lobe.max_fill",
                    dict(qi=qi, oracle="cartesian_envelope"),
                    exc,
                )
            )
        rows += _smooth(qi, settings, lobe)
        return rows


def _row(
    quantity: str,
    setting: str,
    actual: float,
    truth: float,
    truth_error: float,
    case: dict[str, Any],
    *,
    metric: str = "relative",
    bound: float | None = None,
) -> dict[str, Any]:
    """One observation, or a note when the truth cannot resolve the bound."""
    bound = SCALAR_BOUND[setting] if bound is None else bound
    value = abs(actual - truth) if metric == "absolute" else relative(actual, truth)
    extra = dict(actual=actual, truth=truth, truth_error=truth_error)
    if truth_error > TRUTH_MARGIN * bound:
        note = (
            f"truth unresolved: its own error {truth_error:.1e} exceeds a tenth "
            f"of the bound {bound:g}"
        )
        return record(
            quantity, setting, value, None, case, metric=metric, note=note, **extra
        )
    return record(quantity, setting, value, bound, case, metric=metric, **extra)


def _failed(quantity: str, case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return record(
        quantity,
        "reference",
        None,
        SCALAR_BOUND["reference"],
        case,
        error=f"{type(exc).__name__}: {exc}",
    )


def _lobe(
    qi: float, settings: dict[str, rocheometry.ComponentGeometry]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    case = dict(qi=qi, oracle="cartesian")
    radius = cartesian.estimate(lambda order: cartesian.lobe_volume_radius(qi, order))
    slope = cartesian.lobe_slope(qi, radius["error"])
    rows = []
    for setting, g in settings.items():
        for quantity, actual in (
            ("lobe.volume_radius", g.lobe().volume_radius()),
            ("lobe.volume_radius.fast", g.lobe_volume_radius()),
        ):
            rows.append(
                _row(quantity, setting, actual, radius["value"], radius["error"], case)
            )
        rows.append(
            _row(
                "lobe.slope",
                setting,
                g.lobe_volume_radius_slope(),
                slope["value"],
                slope["error"],
                dict(case, oracle="cartesian_difference"),
                metric="absolute",
            )
        )
    return rows, radius


def _gap(g: rocheometry.ComponentGeometry) -> float:
    """Depth of the L2 level below L1, as this component's frame rounds it.

    The two frames may differ by an ulp; each component uses its own.
    """
    return _levels(g.field, g._computers).admissibility_gap


def _l2(qi: float, lobe: dict[str, float]) -> list[dict[str, Any]]:
    reference = rocheometry.ComponentGeometry(qi, accuracy=rocheometry.Reference())
    companion = rocheometry.ComponentGeometry(1 / qi, accuracy=rocheometry.Reference())
    envelope = cartesian.envelope_estimate(reference.field, _gap(reference))
    total, error = envelope["value"], envelope["error"]
    # The split: the companion's Reference volume, so only the sum is independent.
    part = total - companion._computers.volume(companion.field, _gap(companion))
    fill = (part / (4 * pi / 3 * lobe["value"] ** 3)) ** (1 / 3)
    fill_error = (error * total / part + 3 * lobe["error"]) / 3
    rows = []
    for setting, accuracy in (
        ("fitted", rocheometry.Fitted()),
        ("reference", rocheometry.Reference()),
    ):
        g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
        h = rocheometry.ComponentGeometry(1 / qi, accuracy=accuracy)
        volumes = [x._computers.volume(x.field, _gap(x)) for x in (g, h)]
        rows.append(
            _row(
                "integrals.volume",
                setting,
                fsum(volumes),
                total,
                error,
                dict(qi=qi, kind="L2", oracle="cartesian_envelope"),
                metric="relative (both components)",
            )
        )
        rows.append(
            _row(
                "lobe.max_fill",
                setting,
                g.max_volume_radius_fill(),
                fill,
                fill_error,
                dict(qi=qi, oracle="cartesian_envelope"),
            )
        )
    return rows


def _smooth(
    qi: float,
    settings: dict[str, rocheometry.ComponentGeometry],
    lobe: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    fitted, reference = settings["fitted"], settings["reference"]
    lobe_volume = 4 * pi / 3 * lobe["value"] ** 3
    for level in SMOOTH_FILLS:
        case = dict(qi=qi, kind="fill", level=level, oracle="tensor_rule")
        try:
            surfaces = dict(fitted=select(fitted, "fill", level))
            surfaces["reference"] = reference._surface_at_depth(
                surfaces["fitted"]._depth
            )
            truth = surface_integrals(surfaces["reference"])
        except Exception as exc:
            rows.append(_failed("integrals.volume", case, exc))
            continue
        value, error = truth["value"], truth["error"]
        fill = (value["volume"] / lobe_volume) ** (1 / 3)
        fill_error = (error["volume"] + 3 * lobe["error"]) / 3
        for setting, s in surfaces.items():
            for name in READERS:
                rows.append(
                    _row(
                        INTEGRALS[name],
                        setting,
                        getattr(s, name)(),
                        value[name],
                        error[name],
                        case,
                        bound=scalar_bound(setting, INTEGRALS[name]),
                    )
                )
            rows.append(
                _row(
                    "integrals.fill",
                    setting,
                    s.volume_radius_fill(),
                    fill,
                    fill_error,
                    case,
                )
            )
            q = s.quadrature()
            sums = dict(
                x2=fsum(w * p[0] ** 2 for w, p in zip(q.weights, q.positions)),
                z2=fsum(w * p[2] ** 2 for w, p in zip(q.weights, q.positions)),
                gravity4=fsum(w * v**4 for w, v in zip(q.weights, q.gravities)),
            )
            for name, actual in sums.items():
                rows.append(
                    _row(
                        "quadrature.integrals." + name,
                        setting,
                        actual,
                        value[name],
                        error[name],
                        case,
                    )
                )
        clear_samples(fitted)
        clear_samples(reference)
    return rows
