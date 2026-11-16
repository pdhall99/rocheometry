# Surfaces

A [component surface](model.md#component-surfaces) is one component at one level.
A component's `surface()` method picks the level, and everything on this page is then read from the surface it returns.

## Selecting a surface

Select a surface once, then measure it.
Each keyword names the level a different way, and `surface()` takes exactly one of them:

| Keyword | Selects by | Range |
| --- | --- | --- |
| `volume_radius_fill` | `R_vol / R_RL,vol` | above 0, below `max_volume_radius_fill()` |
| `volume_radius` | `R_vol`, divided by `lobe_volume_radius()` to give a fill | above 0, below the part's maximum at L2 |
| `fill_out_factor` | `(U_L1 - U)/(U_L1 - U_L2)` | below 1 |
| `potential` | a `Potential` | above L2 |

```python
import rocheometry

geometry = rocheometry.ComponentGeometry(qi=0.4)
surface = geometry.surface(volume_radius_fill=0.8)
lobe = geometry.lobe()
assert lobe == geometry.surface(volume_radius_fill=1.0)
```

[The model](model.md#sizes-and-fills) defines the fills.
Selections outside these ranges, including L2 itself, raise `ValueError`.
Surfaces are frozen values, and equal selections give equal surfaces.

A fill of one selects the Roche lobe exactly.
For a detached surface, `Fitted()` reads the level from a packaged map from fill to level, for `1e-8 <= qi <= 1e8`.
For a contact part, or outside that range of mass ratios, it finds the level by numerical volume inversion, as `Reference()` always does.
The selected surface's fill is then measured, so it can differ from the request by the selection accuracy on the [accuracy page](accuracy.md).
The map needs no NumPy or SciPy at run time; [its README](https://github.com/pdhall99/rocheometry/tree/main/fits/fill_map) describes how it is built.

A surface is identified by its level's depth below L1.
`surface.potential` reports the level as a Jacobi value, rounded once, so reselecting by it can move the level slightly near L1 or L2.
`fill_out_factor()` is computed from the depth directly, so it avoids that loss of precision, though it can still differ from a requested factor by rounding.
Both components' parts at one level report identical potentials and fill-out factors.

## Level regimes

`surface.regime` says where the surface's level lies.
Since `U` rises towards the components, detached levels are above L1:

| `LevelRegime` | Level | Fill-out factor | Surface |
| --- | --- | --- | --- |
| `DETACHED` | above L1 | below 0 | a closed surface around one component |
| `CRITICAL` | at L1 | 0 | the Roche lobe |
| `CONTACT` | between L1 and L2 | between 0 and 1 | one part of a common envelope |
| `OPEN` | at or below L2 | 1 or more | none: cannot be selected |

`geometry.level_regime(potential)` classifies a level exactly, without constructing a surface.

## Contact parts

Between L1 and L2 each component's surface is its part of the common envelope, ending at the watershed:

```python
binary = rocheometry.BinaryGeometry(q=0.4)
part1 = binary.component(1).surface(fill_out_factor=0.5)
part2 = binary.component(2).surface(fill_out_factor=0.5)  # the same level
part1.regime  # LevelRegime.CONTACT
part1.volume_radius_fill()  # above 1
```

A part's volume and containment use the whole region bounded by its equipotential envelope share and its watershed cut.
The cut is not equipotential.
Area, mean curvature, mean gravity and quadrature cover only the envelope share, without the cut.
[Sections and silhouettes](sections-and-silhouettes.md) describes how parts are drawn.

## Measurements

| Method | Definition | Units |
| --- | --- | --- |
| `volume()` | `V = ∫ dV` over the region | separation³ |
| `volume_radius()` | `(3V/(4π))^(1/3)` | separation |
| `volume_radius_fill()` | `R_vol / R_RL,vol` | — |
| `fill_out_factor()` | `(U_L1 - U)/(U_L1 - U_L2)` | — |
| `area()` | `A = ∫ dA` over the envelope share | separation² |
| `area_radius()` | `sqrt(A/(4π))` | separation |
| `integrated_mean_curvature()` | `M = ∫ H dA` over the envelope share | separation |
| `mean_curvature_radius()` | `M/(4π)` | separation |
| `mean_gravity()` | `∫ abs(∇U) dA / A` over the envelope share | `G(M1+M2)/separation²` |
| `named_radius(name)` | radius towards `"point"`, `"back"`, `"side"` or `"pole"` | separation |
| `radius(direction)` | radius along any nonzero direction | separation |

Units are for a `ComponentGeometry`; [binary views](binaries-and-components.md#a-binary) scale them.
`H` is the mean of the principal curvatures for the outward normal, so a sphere of radius `R` has `H = 1/R` and `M = 4πR`.
Gauss's law gives the mean gravity independently as `(4πm - 2V)/A`, where `m = qi/(1+qi)`, for every regime, since no gravity crosses the cut.

Fills are measured with the same quadrature as the lobe, so `V = V_RL * fill**3` and `R_vol = R_RL,vol * fill` hold exactly for the reported fill.
On the Roche lobe, the point radius is the L1 distance.

The volume, area, curvature and gravity of one surface share one set of radius solves, so measuring several costs about the same as measuring one.

[A recipe](../recipes/surface-gravity.md) converts gravities to `log g` in cgs.

## Point queries

```python
geometry = rocheometry.ComponentGeometry(1.0)
surface = geometry.surface(volume_radius_fill=0.8)
level = geometry.potential_at((0.1, 0.0, 0.0))
assert surface.contains((0.0, 0.0, 0.0))
assert not surface.contains((1.0, 0.0, 0.0))

binary = rocheometry.BinaryGeometry(1.0, separation=2.0)
assert binary.component(2).lobe().contains((2.0, 0.0, 0.0))
```

Points and directions can be any three numbers: a tuple, a list or a NumPy array.
`potential_at(point)` returns the Jacobi potential at a point, rounded once; the two point masses are singular and raise `ValueError`.
`contains(point)` says whether a point is inside the surface's region: a contact part stops at the watershed, and a point on the watershed may belong to either part.
Points within rounding error of the surface can be classified either way.
Don't test membership by comparing `potential_at(point)` with `surface.potential`: both are rounded, and a small difference can be lost.
