# Crosschecks

Reference computations: independent ways of computing what Rocheometry computes, so the tests and the accuracy studies have something to check the library against.
The tests use them on chosen cases, and the studies across the whole domain.
None of this code ships in the wheel.

| Module | Contents |
| --- | --- |
| `problem.py` | The problem in plain numbers (`Binary`, `Level`) and the exact Roche potential in `Decimal` or `mpmath` |
| `radii.py`, `ray_root.py` | Surface radii along a ray, by 40- and 100-digit root solves |
| `cartesian.py` | Volumes by Cartesian x-slices instead of rays |
| `integrals.py` | Area, mean curvature and gravity by a plain tensor rule |
| `geometry.py`, `cone_deficit.py` | The L1-plane crossing, and the Gaussian-curvature cone deficit at L1 |
| `distance/` | Distances between outlines, and from an outline to the exact level set |
| `instrumented/` | Not a truth: replays the library's own node arithmetic with running error bounds |

## What belongs here

A module belongs here only if all four hold:

1. **It computes what the library computes:** a reference value for a library quantity, or the distance between a library result and one.
   Helpers, plumbing and wrappers around library code do not qualify; they live where they are used, in `tests/_helpers.py` or `accuracy/harness/`.
2. **It uses a different method.** The library's own method at higher settings is `Reference()`, not a crosscheck.
3. **It declares what it borrows.** It may use the problem statement and the library's generic numerics (Brent roots, Gauss-Legendre rules) freely.
   Any Roche-specific component it borrows, such as the watershed tracer, is listed under `shares` in `accuracy/catalogue.toml`, with the tests that check that component independently, or as a gap.
4. **It is checked itself,** in `crosschecks/tests/`, against analytic or exact cases.

`instrumented/` is the one stated exception: it replays the library's own arithmetic, and says so.

A test may compute a reference value inline when nothing else needs it.
Once a second check needs the same computation, or a study needs it at all, it moves here.

Crosschecks import only `rocheometry`, mpmath and numpy, never `tests/` or `accuracy/`.
Where possible they take a `Level` rather than a library surface; `Level.of(surface)` is the one adapter.
