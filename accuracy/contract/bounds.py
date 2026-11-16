"""Contract bounds: the accuracy the README promises, in one place.

These are user-facing contracts, distinct from numerical controls inside the
library and from test margins. The fitted selection tolerance is defined in
``rocheometry`` and imported from there; the other bounds are stated in the
library's documentation and defined here for every study to import.
"""

from __future__ import annotations

from typing import Any

from rocheometry._computers.fitted import FITTED_FILL_ABSOLUTE_TOLERANCE

#: Relative bound on every scalar, by accuracy setting.
SCALAR_BOUND = {"fitted": 1e-6, "reference": 1e-10}
#: Default-order Reference bound on the two mean-curvature scalars.
REFERENCE_CURVATURE_BOUND = 2e-8
#: Two-way normal distance between an outline and the truth, in R_vol.
PATH_BOUND = {"fitted": 2.5e-4, "reference": 1e-10}
OUTLINE_BOUND = PATH_BOUND["fitted"]
#: Release gate for fitted outlines: 10% inside the public outline contract.
OUTLINE_MARGIN = 2.25e-4
#: Absolute fill error of fitted selection by volume-radius fill.
FILL_BOUND = FITTED_FILL_ABSOLUTE_TOLERANCE

#: Reference curvature scalars: integrals.mean_curvature and its radius.
CURVATURE = "integrals.mean_curvature"


def _is_curvature(quantity: str) -> bool:
    name = quantity.removeprefix("reference_convergence.")
    return name == CURVATURE or name.startswith(CURVATURE + ".")


def scalar_bound(
    setting: str, quantity: str, *, polar_order: int = 128, azimuth_order: int = 64
) -> float:
    """Return the contract for the measured reader, never its truth order.

    Default Reference curvature is 2e-8 relative. The saddle sweep supports
    1e-10 for all scalars at 512x192; other Reference scalars retain 1e-10.
    These are user-specified contracts, not production numerical constants.
    """
    if setting == "fitted":
        return SCALAR_BOUND["fitted"]
    curvature = _is_curvature(quantity)
    high_order = polar_order >= 512 and azimuth_order >= 192
    return (
        REFERENCE_CURVATURE_BOUND
        if curvature and not high_order
        else SCALAR_BOUND["reference"]
    )


def curvature_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the curvature contract to a Reference scalar row.

    In reference_convergence rows, case.polar/azimuth identify the truth
    order; the actual value is the default-order reader. Only explicit
    measured_order metadata establishes a high-order measured reader.
    """
    if (
        row["set"] != "reference"
        or not _is_curvature(row["quantity"])
        or row["bound"] is None
    ):
        return row
    order = row.get("measured_order", {})
    bound = scalar_bound(
        row["set"],
        row["quantity"],
        polar_order=order.get("polar", 128),
        azimuth_order=order.get("azimuth", 64),
    )
    status = row["status"]
    if status in ("pass", "fail") and row["value"] is not None:
        status = "pass" if row["value"] <= bound else "fail"
    return {**row, "bound": bound, "status": status}
