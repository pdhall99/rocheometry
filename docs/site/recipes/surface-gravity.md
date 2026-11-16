# Convert surface gravity to physical units

Calculate the mean effective gravity and the range at quadrature nodes as `log g` in cgs.
For masses in solar masses and separation in nominal solar radii:

```pycon
>>> from math import log10
>>> import rocheometry
>>>
>>> GM_SUN = 1.3271244e26  # cm³ s⁻², IAU nominal
>>> R_SUN = 6.957e10  # cm, IAU nominal
>>>
>>> binary = rocheometry.BinaryGeometry.from_masses(1.4, 0.6, separation=3.3)
>>> surface2 = binary.component(2).surface(volume_radius=0.75)
>>> unit = GM_SUN * (1.4 + 0.6) / (3.3 * R_SUN) ** 2
>>>
>>> round(log10(surface2.mean_gravity() * unit), 3)  # mean effective gravity
4.454
>>> nodes = surface2.quadrature()
>>> (
...     round(log10(min(nodes.gravities) * unit), 3),
...     round(log10(max(nodes.gravities) * unit), 3),
... )
(4.358, 4.5)

```

Gravities are in units of `G(M1+M2)/a²`, so multiply by `unit` before taking the logarithm.
The constants use [IAU nominal solar units](https://doi.org/10.3847/0004-6256/152/2/41), as `kepler_separation()` does.
The node range samples the surface; it need not include the exact extrema.
See [surface quadrature](../guide/quadrature.md) for the nodes and weights.
