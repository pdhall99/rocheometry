"""Independent 40-digit Newton solutions at every decade; no library derivatives."""

import mpmath as mp
from crosschecks.problem import Binary, masses, potential

import rocheometry
from accuracy.contract.bounds import SCALAR_BOUND
from accuracy.harness.common import decade_jobs
from accuracy.harness.record import record
from accuracy.harness.study import Row, Study


class CriticalPoints(Study):
    name = "critical_points"
    kind = "contract"
    quantities = (
        "critical.l1",
        "critical.outer",
        "critical.triangular",
    )
    domain = "L1, the outer saddle and the triangular point at the 17 decades of qi, against 40-digit mpmath solves."

    def jobs(self) -> list[float]:
        return decade_jobs()

    def smoke(self) -> float:
        return 1.0

    def measure(self, qi: float) -> list[Row]:
        rows = []
        with mp.workdps(40):
            m, c = masses(Binary.of(rocheometry.ComponentGeometry(qi).field), mp.mpf)

            # Differentiate the potential numerically, so no hand-derived
            # gradient, the library's or the crosschecks', enters the check.
            def gradient(x: mp.mpf, y: mp.mpf, z: mp.mpf) -> tuple[mp.mpf, ...]:
                return tuple(
                    mp.diff(
                        lambda t: potential(
                            m, c, *((x, y, z)[:j] + (t,) + (x, y, z)[j + 1 :])
                        ),
                        (x, y, z)[j],
                    )
                    for j in range(3)
                )

            for setting, accuracy in (
                ("fitted", rocheometry.Fitted()),
                ("reference", rocheometry.Reference()),
            ):
                g = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
                for label, point in (
                    ("l1", g.l1()),
                    ("outer", g.outer()),
                    ("triangular", g.triangular()),
                ):
                    x0, y0, _ = point.position
                    if label == "triangular":
                        x, y = mp.findroot(
                            lambda x, y: gradient(x, y, mp.mpf(0))[:2],
                            (mp.mpf(x0), mp.mpf(y0)),
                            tol=mp.mpf("1e-37"),
                        )
                    else:
                        x = mp.findroot(
                            lambda x: gradient(x, mp.mpf(0), mp.mpf(0))[0],
                            mp.mpf(x0),
                            tol=mp.mpf("1e-37"),
                        )
                        y = mp.mpf(0)
                    p = (x, y, mp.mpf(0))
                    u = potential(m, c, *p)
                    h = mp.matrix(
                        [
                            [
                                mp.diff(
                                    lambda t: gradient(*(p[:k] + (t,) + p[k + 1 :]))[j],
                                    p[k],
                                )
                                for k in range(3)
                            ]
                            for j in range(3)
                        ]
                    )
                    a, d, b = h[0, 0], h[1, 1], h[0, 1]
                    delta = mp.sqrt((a - d) ** 2 + 4 * b * b)
                    ev = sorted(((a + d - delta) / 2, (a + d + delta) / 2, h[2, 2]))
                    norm = mp.sqrt(sum(v * v for v in p))
                    hn = mp.sqrt(sum(v * v for v in h))
                    values = dict(
                        position=float(
                            mp.sqrt(
                                sum(
                                    (mp.mpf(v) - t) ** 2
                                    for v, t in zip(point.position, p)
                                )
                            )
                            / norm
                        ),
                        potential=float(abs(mp.mpf(point.potential.value) / u - 1)),
                        hessian=float(
                            mp.sqrt(
                                sum(
                                    (mp.mpf(point.hessian[j][k]) - h[j, k]) ** 2
                                    for j in range(3)
                                    for k in range(3)
                                )
                            )
                            / hn
                        ),
                        eigenvalues=float(
                            max(
                                abs(mp.mpf(v) - t)
                                for v, t in zip(point.eigenvalues, ev)
                            )
                            / max(abs(t) for t in ev)
                        ),
                    )
                    for name, value in values.items():
                        rows.append(
                            record(
                                f"critical.{label}.{name}",
                                setting,
                                value,
                                SCALAR_BOUND[setting],
                                dict(qi=qi, point=label),
                                metric="maximum absolute eigenvalue error / spectral norm(H)"
                                if name == "eigenvalues"
                                else "relative",
                                truth=[str(v) for v in ev]
                                if name == "eigenvalues"
                                else None,
                            )
                        )
                    rows.append(
                        record(
                            f"critical.{label}.eigenvalues_relative",
                            setting,
                            float(
                                max(
                                    abs(mp.mpf(v) / t - 1)
                                    for v, t in zip(point.eigenvalues, ev)
                                )
                            ),
                            None,
                            dict(qi=qi, point=label),
                            note="Individual relative eigenvalue error: diagnostic; the contract is norm-wise.",
                        )
                    )
        return rows
