---
description: "Exact outlines of component surfaces as cubic Bézier paths: sections in any plane and silhouettes at any inclination, with SVG and Matplotlib export."
---

# Sections and silhouettes

A section is the outline of a component surface's region in a plane, including the watershed cut in contact.
A silhouette is the projected limb of its envelope seen along the plane's normal.
Both are cubic Bézier paths in the plane's 2D coordinates.
Neither is a [contour](model.md#sections-silhouettes-and-contours), which would include every branch of a level curve in the plane.

## Planes

- **`CentrelinePlane(inclination)`** contains the binary centreline, tilted by `inclination` radians in `[0, pi]` (use `math.radians()` for degrees).
  Its coordinates are `(x, u)`, with `u` along `(0, cos(i), -sin(i))`.
  `ORBITAL_PLANE` is `CentrelinePlane(0.0)`, the default everywhere.
- **`AxisNormalPlane(axis, offset)`** is normal to the `x`, `y` or `z` axis at `offset`.
  Its coordinates are `(y, z)` for an x plane, `(x, z)` for y and `(x, y)` for z, with the origin at the foot of the perpendicular from the frame's origin.
  `AxisNormalPlane("y", 0.0)` is `CentrelinePlane(pi/2)` with its second coordinate reflected.

`SectionPlane` is the union of the two.
`plane.project(point3d)` projects a point into the plane's coordinates and `plane.lift(point2d)` puts it back.
A binary view reads planes in its own frame and length units.

## Sections

```python
from math import pi
import rocheometry

binary = rocheometry.BinaryGeometry(q=0.4)
surface = binary.component(2).surface(volume_radius_fill=0.8)
path = surface.section()  # the orbital plane
inclined = surface.section(rocheometry.CentrelinePlane(pi / 6))
slice_ = surface.section(rocheometry.AxisNormalPlane("y", 0.1))  # at y = 0.1
```

**Detached and critical surfaces** give one closed path, starting at the front and running upper half first.
At fill one, the first two segments of each half are degenerate at L1.

**Contact parts** are closed by their watershed cut.
A centreline section starts at L1, follows the upper cut to the rim, goes around the back along the envelope, and returns along the lower cut; the rim is a corner.
`section_envelope()` returns only the envelope, as open arcs between the rims, using exactly the section's segments.
The two components' arcs together draw the common envelope.

Axis-normal planes can cut a contact part in several ways:

- A y or z plane that crosses the watershed includes its curved cut; one that misses it gives a closed loop from both `section()` and `section_envelope()`.
- In an x plane the boundary can switch between envelope and cut, with a corner at each switch.
  `section_envelope()` returns each envelope run, and an empty path when the boundary is all cut.
- A y or z plane near the lighter component's rim can cut the component and a separate collar.
  The component's loop comes first, then the collar, then any further pieces by increasing x.
- A ring-shaped x slice just beyond L1 on the heavier component raises `NotImplementedError`.

A section needs its polar origin, the foot of the perpendicular from the component's centre, strictly inside the surface; otherwise it raises `EmptySectionError`, a `ValueError`.

`Fitted()` centreline sections have a fixed layout of 52 segments, so outlines change smoothly in an animation, and their control points are continuous through L1.
`Fitted()` axis-normal sections, and all `Reference()` sections, are traced adaptively; tracing raises `RuntimeError` in the rare case that refinement cannot meet its target.

## Silhouettes at quadrature

Quadrature here is the orbital phase at which the binary centreline lies in the sky plane; it is unrelated to [surface quadrature](quadrature.md).
Silhouettes take a `CentrelinePlane(i)`, where `i` is the orbital inclination: 0 views the orbit face-on and `pi/2` edge-on.
The centreline stays in the sky plane and the line of sight is the plane's normal, `(0, sin(i), cos(i))`.
An `AxisNormalPlane` raises `TypeError`.

- **`silhouette(plane)`** returns the closed outline of a detached or critical surface, starting at the front and running upper half first.
  Contact levels raise `NotImplementedError`.
- **`silhouette_envelope(plane)`** returns the same closed outline for a detached or critical surface, and for a contact part its open limb arc, which ends where it crosses the watershed.
  The two components' arcs together draw the common envelope.

The cut, occlusion between the components and draw order are left to the caller.
`Fitted()` silhouettes have a fixed layout of 88 segments; `Reference()` traces the exact limb.
If the limb search cannot resolve a degenerate case, a silhouette raises `DegenerateLineOfSightError` or `ComponentOwnershipError`, both `RuntimeError`s.

This example overlays the sections and limbs of a contact binary seen at 60°, using the optional `plot` extra:

```pycon
>>> from math import pi
>>>
>>> import matplotlib.pyplot as plt
>>> import rocheometry
>>> from rocheometry.matplotlib import path_patch
>>>
>>> binary = rocheometry.BinaryGeometry(0.3)
>>> level = binary.component(1).surface(fill_out_factor=0.5).potential
>>> sky = rocheometry.CentrelinePlane(pi / 3)
>>> parts = [binary.component(k).surface(potential=level) for k in (1, 2)]
>>> sections = [part.section(sky) for part in parts]
>>> limbs = [part.silhouette_envelope(sky) for part in parts]
>>> all(not path.closed for limb in limbs for path in limb.paths)
True
>>> fig, ax = plt.subplots()
>>> for limb in limbs:
...     _ = ax.add_patch(path_patch(limb))
>>> for section in sections:
...     _ = ax.add_patch(path_patch(section, linestyle="--"))
>>> x_l1, t_l1 = sky.project(binary.l1().position)
>>> _ = ax.plot(x_l1, t_l1, "o")
>>> ax.autoscale_view()
>>> ax.set_aspect("equal")
>>> plt.close(fig)

```

## Paths and export

Sections and silhouettes return a `CompoundBezierPath`: a frozen tuple of `BezierPath`s in `paths`, which need not connect and each have their own `closed` flag.
`len()` counts the paths, iteration follows their order, and an empty compound is falsy.
`bounds` gives the exact bounds, `transformed(transform)` applies an `AffineTransform2D` and `reversed()` reverses both order and direction.

Sample the individual paths, either evenly in parameter or in arc length, which skips degenerate segments:

```python
points = [piece.sample(401) for piece in path]
uniform = [piece.sample(401, spacing="arc_length") for piece in path]
```

**SVG and Matplotlib have parallel path adapters**, using each backend's own vocabulary: `svg.path_element(path)` returns an SVG `<path>` element string, while `matplotlib.path_patch(path)` returns a Matplotlib `PathPatch`.
Both accept `BezierPath` and `CompoundBezierPath`, preserve their native cubic segments and open/closed subpaths without sampling, and default to an unfilled outline.

`svg.document(binary, *, margin=0.05, linewidth=1.0, precision=12, y_axis="down")` produces a standalone SVG of both critical Roche lobes.
`svg.document_paths(paths, *, margin=0.05, linewidth=1.0, precision=12, classes=None, y_axis="down")` makes a document from any ordered collection of paths, with optional per-path CSS classes.
Both compute exact cubic bounds and pad by a fraction of the larger dimension.
SVG normally displays positive y down; `y_axis="up"` flips only the display so that it matches Matplotlib's upward-positive axes.

Both documents use a `linewidth` in screen pixels, independent of the drawing's scale, with a default of 1 pixel.
Bounds exclude the stroke, so with `margin=0` half the stroke can be clipped.
For callers composing their own SVG, `svg.path_element(path, linewidth=...)` uses SVG coordinate units.

With the optional `plot` extra, `matplotlib.add_binary_lobes(ax, binary, *, margin=0.05, **kwargs)` adds both Roche-lobe patches to caller-provided axes, with shared padded bounds and equal aspect.

```pycon
>>> import rocheometry
>>> from rocheometry import svg
>>> binary = rocheometry.BinaryGeometry(0.25, separation=4.0)
>>> document = svg.document(binary)
>>> document.startswith("<svg ") and document.endswith("</svg>")
True
>>> outline = binary.component(1).surface(volume_radius_fill=0.8).section()
>>> element = svg.path_element(outline, class_name="detached", linewidth=1.5)
>>> element.startswith('<path class="detached"')
True
>>> custom = svg.document_paths((outline,), y_axis="up")
>>> 'transform="scale(1,-1)"' in custom
True

```

For plotting the same path with Matplotlib, use `ax.add_patch(path_patch(outline, linewidth=1.5))`.
SVG returns a string, while Matplotlib returns an artist that can be added to axes.
Both draw the path's own coordinates: neither rescales them, and only `y_axis` changes how an SVG displays them.

The recipes draw a [Roche lobe diagram](../recipes/roche-lobe-diagram.md) from both components' sections at several levels, and a [Roche potential diagram](../recipes/potential-diagram.md) of every branch of the potential.
