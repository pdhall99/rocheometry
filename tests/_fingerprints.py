"""Record the exact digests that tests/test_accuracy.py compares.

Digests include the platform's libm rounding, so they are recorded on one
machine (the one that makes the release run) and checked bit for bit only
there; elsewhere the test runs exact-field checks instead. Re-record them
when an output changes on purpose, and only then:

    hatch run hatch-test.py3.14:fingerprints

The Python version is part of the recorded environment: use the one you run
the tests with.
"""

from __future__ import annotations

import hashlib
import json
import platform
import struct
import subprocess
import sys
from dataclasses import fields, is_dataclass
from math import pi
from pathlib import Path

import rocheometry

FIXTURE = Path(__file__).parent / "fixtures/accuracy-values.json"
#: Every combination of setting, mass ratio and regime that is fingerprinted.
CASES = tuple(
    (reference, qi, kind)
    for reference in (False, True)
    for qi in (0.001, 1.0, 1000.0)
    for kind in ("detached", "critical", "contact")
)


def digest(value: object) -> str:
    """SHA-256 of a value's binary64 bits and structure."""
    h = hashlib.sha256()

    def visit(v: object) -> None:
        if isinstance(v, float):
            h.update(b"f" + struct.pack(">d", v))
        elif is_dataclass(v) and not isinstance(v, type):
            h.update(type(v).__name__.encode())
            for f in fields(v):
                visit(getattr(v, f.name))
        elif isinstance(v, tuple):
            h.update(b"(" + str(len(v)).encode())
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())

    visit(value)
    return h.hexdigest()


def surface(
    reference: bool, qi: float, kind: str
) -> rocheometry.ComponentSurfaceGeometry:
    accuracy = rocheometry.Reference() if reference else rocheometry.Fitted()
    geometry = rocheometry.ComponentGeometry(qi, accuracy=accuracy)
    if kind == "detached":
        return geometry.surface(volume_radius_fill=0.8)
    if kind == "critical":
        return geometry.lobe()
    return geometry.surface(fill_out_factor=0.5)


def values(s: rocheometry.ComponentSurfaceGeometry) -> dict[str, object]:
    """Everything a fingerprint covers: scalars, nodes and two outlines."""
    plane = rocheometry.CentrelinePlane(pi / 3)
    return {
        "potential": s.potential.value,
        "volume": s.volume(),
        "area": s.area(),
        "mean_curvature": s.integrated_mean_curvature(),
        "mean_gravity": s.mean_gravity(),
        "quadrature": s.quadrature(),
        "section": s.section(plane),
        "silhouette": s.silhouette_envelope(plane),
    }


def environment() -> dict[str, object]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": list(platform.python_version_tuple()[:2]),
    }


def main() -> int:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    record = {
        "provenance": (
            "Exact digests of each accuracy setting's outputs, recorded by "
            "tests/_fingerprints.py. They change only when an output changes."
        ),
        "source_commit": commit or "unknown",
        "environment": environment(),
        "cases": [
            {
                "reference": reference,
                "qi": qi,
                "kind": kind,
                "values": {
                    name: digest(value)
                    for name, value in values(surface(reference, qi, kind)).items()
                },
            }
            for reference, qi, kind in CASES
        ],
    }
    FIXTURE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"recorded {len(CASES)} cases on {environment()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
