"""Computers depend on the field layer, never the public geometry/boundary layer."""

import ast
from importlib.util import resolve_name
from pathlib import Path


def test_computers_do_not_import_geometry_or_level_boundaries() -> None:
    root = Path(__file__).parents[1] / "src/rocheometry"
    forbidden = (
        "rocheometry._levels",
        "rocheometry._component_geometry",
        "rocheometry._binary_geometry",
    )
    violations = []
    for path in (root / "_computers").rglob("*.py"):
        package = "rocheometry." + ".".join(path.relative_to(root).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = resolve_name("." * node.level + module, package)
                targets = [module, *[module + "." + a.name for a in node.names]]
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if name in {"__import__", "import_module"}:
                    targets = [node.args[0].value]
            for target in targets:
                if any(target == f or target.startswith(f + ".") for f in forbidden):
                    violations.append(
                        f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}: {target}"
                    )
    assert not violations, "\n".join(violations)
