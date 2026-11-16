"""Independent broad and dense near-critical fill-map coverage; no refitting."""

from math import pi

import rocheometry
from accuracy.harness.cases import load
from accuracy.harness.common import clear_samples, relative
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study
from rocheometry._computers.fitted import FITTED_FILL_ABSOLUTE_TOLERANCE
from rocheometry._computers.surface_volume import SampledVolumeComputer


class Selection(Study):
    name = "selection"
    kind = "contract"
    quantities = (
        "selection.by_fill",
        "integrals.volume",
    )
    domain = "The frozen selection grid (selection.json): broad and dense near-critical fills, in batches of 20."

    def jobs(self) -> list[list[tuple[float, float, str]]]:
        cases = [tuple(c) for c in load("selection.json")["cases"]]
        return [cases[i : i + 20] for i in range(0, len(cases), 20)]

    def smoke(self) -> list[tuple[float, float, str]]:
        return self.jobs()[0][:2]

    def measure(self, batch: list[tuple[float, float, str]]) -> list[Row]:
        rows = []
        for qi, fill, region in batch:
            case = dict(qi=qi, kind="fill", level=fill, region=region)
            try:
                fitted = rocheometry.ComponentGeometry(qi)
                reference = rocheometry.ComponentGeometry(
                    qi, accuracy=rocheometry.Reference()
                )
                chosen = fitted.surface(volume_radius_fill=fill)
                volume = SampledVolumeComputer(reference._computers.sample)
                reference_volume = volume(reference.field, chosen._depth)
                measured = (reference_volume / volume(reference.field, 0.0)) ** (1 / 3)
                reported = chosen.volume_radius_fill()
                rows.append(
                    record(
                        "selection.by_fill",
                        "fitted",
                        abs(fill - measured),
                        FITTED_FILL_ABSOLUTE_TOLERANCE,
                        case,
                        metric="absolute in requested fill",
                        potential=chosen.potential.value,
                        map_fill=fill,
                        measured_fill=reported,
                        reference_fill=measured,
                        selection_absolute_error=abs(measured - fill),
                    )
                )
                for name, actual, truth in (
                    ("integrals.volume", chosen.volume(), reference_volume),
                    (
                        "integrals.volume.radius",
                        chosen.volume_radius(),
                        (3 * reference_volume / (4 * pi)) ** (1 / 3),
                    ),
                    ("selection.by_fill.measured", reported, measured),
                ):
                    rows.append(
                        record(
                            name,
                            "fitted",
                            relative(actual, truth),
                            1e-6,
                            case,
                            actual=actual,
                            truth=truth,
                        )
                    )
                clear_samples(reference)
                clear_samples(fitted)
            except Exception as exc:
                rows.append(
                    record(
                        "selection.by_fill",
                        "fitted",
                        None,
                        FITTED_FILL_ABSOLUTE_TOLERANCE,
                        case,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return rows
