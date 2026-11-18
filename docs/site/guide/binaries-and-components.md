---
description: "How to build a binary and reach its components in Rocheometry, with mass ratios, frames, units, separations and Kepler's third law."
---

# Binaries and components

Binaries and components are how you reach a [component surface](model.md#component-surfaces): they fix the component, and their `surface()` and `lobe()` methods fix the level.
They also hold what does not depend on a level, such as critical points and the Roche-lobe radius.

Rocheometry has two starting points.
A `BinaryGeometry` is a whole system, in its own frame and length units.
A `ComponentGeometry` is one component in its local frame, in units of the separation.
Both give the same surfaces and quantities.

## A binary

```python
import rocheometry

separation = rocheometry.kepler_separation(period_days=0.5, total_mass_msun=2.0)
binary = rocheometry.BinaryGeometry.from_masses(1.4, 0.6, separation=separation)
binary = rocheometry.BinaryGeometry(
    q=1.4 / 0.6, separation=separation
)  # the same binary
```

`kepler_separation()` returns the separation in nominal solar radii.
`separation`, `frame` (`"binary"` or `"centre_of_mass"`) and `accuracy` are keyword-only, with defaults `1.0`, `"binary"` and `Fitted()`.

The binary itself holds what belongs to both components: `l1()` to `l5()`, `centre_of_mass`, `potential_at(point)`, `convert_potential(...)`, and `lobe_bounds()`, the combined orbital-plane bounds `(xmin, ymin, xmax, ymax)` of both Roche lobes.

Everything else starts by choosing a component:

```python
component2 = binary.component(2)

component2.component_id  # 2
component2.position  # its centre, in the binary's frame
component2.lobe()  # its Roche lobe
component2.surface(volume_radius_fill=0.8)  # a surface inside the lobe
component2.l1(), component2.outer(), component2.triangular()
```

`binary.component(...)` returns a new view each time, so compare views with `==`, not `is`.
A view's `intrinsic_geometry` is the component's `ComponentGeometry`, also available as `binary.geometry1` and `binary.geometry2`; the view transforms and scales its results:

| Result | In a binary view |
| --- | --- |
| Potentials | unchanged |
| Lengths, areas, volumes | times `separation`, `separation**2`, `separation**3` |
| Integrated mean curvature | times `separation` |
| Mean gravity and node gravities | unchanged, in units of `G*(M1+M2)/separation**2` |
| Positions, paths and plane offsets | in the binary's frame and length units |
| Directions passed to `radius()` | in the binary's axes |
| Named radii | keep their component-local meanings |
| Critical-point Hessians | rotated, not rescaled |

So "the pole radius of component 1 at fill 0.8" reads as one chain:

```python
binary.component(1).surface(volume_radius_fill=0.8).named_radius("pole")
```

## One component

```python
geometry = rocheometry.ComponentGeometry(qi=0.4)
surface = geometry.surface(volume_radius_fill=0.8)
```

`qi` is the component's own mass ratio, and everything is in its local frame with the separation as the unit of length.
`geometry.field` is its Roche field.

A `ComponentGeometry(1/q)` built separately uses the rounded ratio, so it need not compare equal to `binary.geometry2`.

## Roche-lobe radius

```python
geometry.lobe_volume_radius()  # R_RL,vol
geometry.lobe_volume_radius_slope()  # d ln(R_RL,vol) / d ln(qi)
geometry.volume_radius_fill_for(0.3)  # 0.3 / lobe_volume_radius()
```

The slope falls from `1/3` in the Hill limit, `qi -> 0`, towards zero as `qi -> infinity`; against the reciprocal ratio it has the opposite sign.
With `Fitted()` these use a packaged model and its analytic derivative, with no sampling, while `lobe().volume_radius()` measures the critical surface; the two agree within the [accuracy contract](accuracy.md).
With `Reference()` both are numerical measurements.

## Sharing and reproducing results

Geometries, surfaces and views are frozen values.
Equality compares their physical inputs and accuracy settings, and equal values share their computed results, held in bounded caches; an evicted result is recomputed when needed.
Constructing a geometry checks its inputs but solves nothing.

Pickles hold only values and settings.
Unpickling reconnects to the receiving process's shared results, so worker tasks in one process reuse each other's work.
A restored object recomputes its results with the installed version of Rocheometry.
