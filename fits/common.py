"""What the builders share: paths, hashes and the build environment.

This file is an input of every build (see ``fits/fit.py``), so it holds
only what a build needs.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/rocheometry/_data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folder(name: str) -> Path:
    return ROOT / "fits" / name


def build_inputs(name: str) -> dict[str, str]:
    """Hashes of the files whose content determines a build's output."""
    base = folder(name)
    inputs = {
        f"fits/{name}/{p.name}": sha256(p)
        for p in sorted([base / "config.yaml", base / "build.py"])
    }
    inputs["fits/common.py"] = sha256(ROOT / "fits/common.py")
    return inputs


def environment() -> dict[str, str]:
    """The parts of the build environment that can change numerical output."""
    import numpy

    result = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(platform.python_version_tuple()[:2]),
    }
    result["numpy"] = numpy.__version__
    try:
        import scipy

        result["scipy"] = scipy.__version__
    except ImportError:
        pass
    return result
