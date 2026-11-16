"""Identify the library's code, ignoring changes that cannot affect a result.

A release run records ``code_hash()`` for the ``src/`` it measured, and a
release is published only when its own ``src/`` has the same hash. The hash
covers every committed file under ``src/``: each Python file as its syntax
tree, without docstrings, and every other file byte for byte, each with its
path. A docstring here is any string that stands alone as a statement, such
as a module, class, function or attribute docstring: none of them can change
a result. Comments, formatting, line numbers and docstrings therefore do not
change the hash; any change to code, another string, a file name or a data
file does.

The tree is written out with its empty and unset fields left out, because
those are where the ``ast`` module differs between Python versions; the hash
is the same on every supported Python.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_docstring(statement: object) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _canonical(node: object) -> str:
    if isinstance(node, list):
        items = [item for item in node if not _is_docstring(item)]
        return "[" + ",".join(_canonical(item) for item in items) + "]"
    if not isinstance(node, ast.AST):
        return repr(node)
    parts = [type(node).__name__]
    for name, value in ast.iter_fields(node):
        if isinstance(value, list):
            value = [item for item in value if not _is_docstring(item)]
        if value is None or value == []:
            continue
        parts.append(f"{name}={_canonical(value)}")
    return "(" + " ".join(parts) + ")"


def canonical(source: bytes | str) -> str:
    """Return a Python source's syntax tree without docstrings, as text."""
    return _canonical(ast.parse(source))


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True
    ).stdout


def code_hash(revision: str = "HEAD") -> str:
    """Hash the code of ``src/`` at ``revision``, or say why there is none.

    With the default revision, a checkout whose ``src/`` differs from ``HEAD``
    gives ``"modified"``, because its working files are not what was hashed.
    """
    try:
        if revision == "HEAD" and _git("status", "--porcelain", "src"):
            return "modified"
        names = _git("ls-tree", "-r", "--name-only", revision, "src").decode()
        digest = hashlib.sha256()
        for name in names.splitlines():
            data = _git("show", f"{revision}:{name}")
            text = canonical(data).encode() if name.endswith(".py") else data
            digest.update(name.encode() + b"\0" + text + b"\0")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return digest.hexdigest()
