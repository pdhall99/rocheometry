---
description: "An example of using Rocheometry: create a binary, select a component surface by how far it fills its Roche lobe, read its radii and draw its outline."
---

# Quickstart

Create a binary from its masses and separation.
Lengths are in the separation's units, here solar radii:

```pycon
>>> import rocheometry
>>>
>>> binary = rocheometry.BinaryGeometry.from_masses(1.4, 0.6, separation=3.3)
>>> component1, component2 = binary.component(1), binary.component(2)
>>> round(component2.lobe().volume_radius(), 3)
1.017

```

If you know the mass ratio rather than the masses, `BinaryGeometry(q=1.4 / 0.6, separation=3.3)` builds the same binary.

Select a surface by what you know about it, such as how far it fills its Roche lobe, and read its quantities:

```pycon
>>> surface2 = component2.surface(volume_radius_fill=0.8)
>>> round(surface2.volume_radius(), 3), round(surface2.named_radius("pole"), 3)
(0.813, 0.787)

```

A surface can also be selected by its volume radius, its potential or its fill-out factor.
Selecting by the radius just measured gives the same surface:

```pycon
>>> same = component2.surface(volume_radius=0.813)
>>> round(same.volume_radius_fill(), 3)
0.8

```

Select component 1's surface at the same level of the potential:

```pycon
>>> surface1 = component1.surface(potential=surface2.potential)
>>> round(surface1.volume_radius_fill(), 3)
0.86

```

Draw a surface's outline.
Sections are cubic Bézier paths, which convert directly to SVG:

```pycon
>>> from rocheometry import svg
>>>
>>> outline = surface2.section()  # the orbital plane by default
>>> svg.path_element(outline).startswith("<path ")  # SVG drawing element
True
>>> document = svg.document(binary)  # both Roche lobes as a standalone SVG

```

Between L1 and L2, the components share an envelope.
Select it by its fill-out factor; each component's part ends at the watershed:

```pycon
>>> part = component1.surface(fill_out_factor=0.5)
>>> part.regime
<LevelRegime.CONTACT: 'contact'>
>>> arcs = part.section_envelope()  # the outline, without the watershed cut

```
