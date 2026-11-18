---
description: "Integrate a gravity-darkened luminosity and a projected area over a component surface, using the quadrature nodes' gravities and normals."
---

# Calculate luminosity and projected area

Integrate a gravity-darkened luminosity and the projected area towards the pole.
Use the quadrature weights, gravities and normals with `T_eff ∝ g^β` and `β = 0.25`:

```pycon
>>> from math import fsum, isclose
>>>
>>> import rocheometry
>>>
>>> surface = rocheometry.ComponentGeometry(0.4).surface(volume_radius_fill=0.8)
>>> nodes = surface.quadrature()
>>> beta = 0.25
>>> luminosity = fsum(w * g ** (4 * beta) for w, g in zip(nodes.weights, nodes.gravities))
>>> luminosity > 0
True
>>> v = (0.0, 0.0, 1.0)
>>> projected_area = fsum(
...     w * max(0.0, fsum(ni * vi for ni, vi in zip(n, v)))
...     for w, n in zip(nodes.weights, nodes.normals)
... )
>>> 0 < projected_area < fsum(nodes.weights)
True
>>> isclose(fsum(nodes.weights), surface.area(), rel_tol=1e-13)
True

```

For [von Zeipel gravity darkening](https://doi.org/10.1093/mnras/84.9.665), luminosity is proportional to `sum(w*g**(4*beta))`; its normalization is left to the caller.
Projected area towards a unit direction `v` is `sum(w*max(0, n·v))`, assuming no self-occultation.
That integrand has a kink at `n·v = 0`, so it converges more slowly than a smooth surface integral.
A [silhouette](../guide/sections-and-silhouettes.md) gives the projected boundary directly.
