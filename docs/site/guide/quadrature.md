---
description: "Surface quadrature nodes with positions, normals, gravities, curvatures and area weights, for your own integrals over a component surface."
---

# Surface quadrature

`surface.quadrature()` returns a frozen `SurfaceQuadrature` for your own surface integrals.
Its equal-length tuples hold, for each node:

- `positions`;
- `normals`, unit and outward;
- `gravities`, the local `abs(∇U)`, in units of `G*(M1+M2)/separation**2`;
- `weights`, area weights that sum to `area()`;
- `mean_curvatures`, `H`, as for `integrated_mean_curvature()`;
- `gaussian_curvatures`, `K = gᵀ adj(Hess U) g / abs(g)**4` with `g = ∇U`, which can be negative.

`len(quadrature)` counts the nodes.
They run polar-major, azimuth-minor, with the four reflections `(y, z)`, `(y, -z)`, `(-y, z)`, `(-y, -z)` of each node consecutive.
A contact part's nodes cover its envelope share only.
In a binary view, positions are in the binary's frame, normals are rotated, and weights scale by `separation**2`, `H` by `1/separation` and `K` by `1/separation**2`; gravities keep their units.

Smooth integrands converge like `area()`.
Node positions, normals and gravities are [checked differently](accuracy.md#quadrature-node-accuracy) from other quantities, because they are sensitive to rounding near L1.

The [gravity darkening recipe](../recipes/gravity-darkening.md) uses the nodes for a gravity-darkened luminosity and a projected area.
