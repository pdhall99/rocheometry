---
description: "Calculate the L1 nozzle factor and the cone half-angle of the critical surface at L1 from the potential's Hessian."
---

# Calculate the L1 nozzle factor and cone angle

Calculate the nozzle's geometric factor and the critical surface's cone half-angle at L1.
Use the Hessian of an equal-mass binary and check the angle against the section's tangent, skipping collapsed segments:

```pycon
>>> from math import atan, degrees, isclose, sqrt
>>>
>>> import rocheometry
>>>
>>> donor = rocheometry.ComponentGeometry(1.0)
>>> point = donor.l1()
>>> point.morse_index, point.eigenvalues
(2, (-8.0, -7.0, 17.0))
>>> h = point.hessian
>>> nozzle_factor = 1 / sqrt(h[1][1] * h[2][2])
>>> round(nozzle_factor, 6)
0.133631
>>> tan_half_angle = sqrt(h[0][0] / abs(h[1][1]))
>>> round(degrees(atan(tan_half_angle)), 3)
57.312
>>> path = donor.lobe().section().paths[0]
>>> segment = next(s for s in path.segments if s.p0 == point.position[:2] and s.p1 != s.p0)
>>> dx, dy = segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1]
>>> isclose(abs(dy / dx), tan_half_angle, rel_tol=1e-12)
True
>>> binary = rocheometry.BinaryGeometry(1.0, separation=12, frame="centre_of_mass")
>>> binary.l1().hessian == point.hessian
True
>>> binary.l1().eigenvalues == point.eigenvalues
True

```

The factor is about 0.133631 and the orbital-plane half-angle is 57.312°.
The Hessian and its eigenvalues are unchanged by the binary's separation or a shift to the centre-of-mass frame.

Near L1, $U - U_{\mathrm{L1}} \approx (U_{xx}\,dx^2 + U_{yy}\,y^2 + U_{zz}\,z^2)/2$, with $U_{xx} > 0$ and $U_{yy}, U_{zz} < 0$.
The cone has $\tan^2(\text{half-angle}) = U_{xx}/|U_{yy}|$.
Integrating an isothermal transverse Gaussian gives the nozzle factor $1/\sqrt{U_{yy} U_{zz}}$, as used by [Ritter (1988)](https://ui.adsabs.harvard.edu/abs/1988A%26A...202...93R/abstract).
The effective area is $Q = 2\pi c_s^2 a^3 / (G M_{\mathrm{total}}) / \sqrt{U_{yy} U_{zz}}$, where $c_s$ is the isothermal sound speed; the factor alone does not set a mass-transfer rate.
