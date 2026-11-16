"""Independent ways of computing what Rocheometry computes, to check it against.

`problem` states the problem in plain numbers, as `Binary` and `Level`, and
evaluates the exact Roche potential at high precision. The rest build on it:

- Truths compute a quantity by a different method from the library's:
  `radii`, `ray_root`, `cartesian`, `integrals`, `geometry` and
  `cone_deficit`. Some still reuse a library component, such as the
  watershed tracer; the study catalogue says which, and what checks it.
- `distance` measures how far an outline lies from the exact level set or
  from another outline.
- `instrumented` is not a truth: it replays the library's own arithmetic with
  error bounds.

The library's generic numerics (Brent roots, Gauss-Legendre rules) are shared
throughout.
"""
