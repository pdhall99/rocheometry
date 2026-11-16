# Potentials and critical points

## The Jacobi potential

Rocheometry's potential is the positive Jacobi potential, normalized by the total mass and kept in the binary's centre-of-mass gauge.
In a component's local frame,

$$
U =
\frac{m_i}{r_i}
+
\frac{m_c}{r_c}
+
\frac12\left[(x-m_c)^2+y^2\right],
\qquad
m_i=\frac{q_i}{1+q_i},
\quad
m_c=\frac{1}{1+q_i},
$$

where $r_i^2 = x^2+y^2+z^2$ and $r_c^2 = (x-1)^2+y^2+z^2$.
It is related to the dimensional rotating-frame effective potential by

$$
U = -\frac{a\,\Phi_{\mathrm{eff}}}{G(M_i+M_c)},
\qquad
\omega_{\mathrm{orb}}^2 = \frac{G(M_i+M_c)}{a^3}.
$$

`U` is the rotating-frame pseudo-potential of the restricted three-body problem; the Jacobi constant is $C_J = 2U - v^2$, so a zero-velocity surface has $C_J = 2U$.
The name `U` avoids $\Omega$, which the binary-star literature uses for the differently normalized Kopal potential.

Because the centre-of-mass constant is kept, exchanging the components is an exact identity:

$$
U_{q_i}(x,y,z) = U_{1/q_i}(1-x,\,-y,\,z).
$$

## Potential values

A potential value carries its convention.
A Jacobi value belongs to the binary, so it is the same for both components; a Kopal value is component-centred and must name its component:

```python
import rocheometry

jacobi = rocheometry.Potential.jacobi(1.9)
kopal = rocheometry.Potential.kopal(4.0, component_id=1)

binary = rocheometry.BinaryGeometry(q=0.4)
converted = binary.convert_potential(
    kopal, convention=rocheometry.PotentialConvention.JACOBI
)
same = rocheometry.convert_potential(
    kopal, q=0.4, convention=rocheometry.PotentialConvention.JACOBI
)
```

Bare floats are not accepted as potentials.
Every potential Rocheometry returns, such as `surface.potential` or `geometry.potential_at(point)`, is a Jacobi value rounded once.
A `ComponentGeometry` takes only Jacobi values, because a Kopal value's component number belongs to the binary; binary views take either.

### Kopal potential

For component `i`, with $q_{K,i} = M_{\mathrm{companion}}/M_i = 1/q_i$, the component-centred Kopal potential in that component's local frame is

$$
\Omega_{K,i} =
\frac{1}{r_i}
+
\frac{q_{K,i}}{r_c}
+
\frac{1+q_{K,i}}{2}(x^2+y^2)
-q_{K,i}\,x,
$$

normalized by $GM_i/a$.
It converts exactly to

$$
U =
\frac{\Omega_{K,i}}{1+q_{K,i}}
+
\frac{1}{2}\left(\frac{q_{K,i}}{1+q_{K,i}}\right)^2.
$$

### From Wilson–Devinney or PHOEBE

Wilson–Devinney ([Wilson & Devinney 1971](https://doi.org/10.1086/150986)) and PHOEBE ([Prša et al. 2016](https://doi.org/10.3847/1538-4365/227/2/29)) define the mass ratio as `M2/M1` and quote Kopal potentials in the primary's frame, component 1, with $q_K$ equal to their mass ratio.
So pass the reciprocal mass ratio, and the potential as a Kopal value for component 1; [a recipe](../recipes/wilson-devinney-and-phoebe.md) works through an example.

### Component-scaled potential

Another common form normalizes by the companion's mass:

$$
\Omega_{\mathrm{s}} = \frac{q_i}{r_i} + \frac{1}{r_c} + \frac{1+q_i}{2}(x^2+y^2) - x
= (1 + q_i)\,U - \frac{1}{2(1+q_i)}.
$$

Every convention is affine in `U`, so surfaces, critical points and tangent directions are the same in all of them; gradients and Hessians scale by the same factor, here $1+q_i$.

## Critical points

```python
geometry = rocheometry.ComponentGeometry(qi=0.4)
geometry.l1()  # the inner Lagrange point
geometry.outer()  # the collinear point beyond this component
geometry.triangular()  # the triangular point at +y

binary.l1(), binary.l2(), binary.l3(), binary.l4(), binary.l5()
```

L2 lies beyond the less massive component and L3 beyond the more massive one, so at equal masses L2 is component 2's `outer()`.
On the centreline the three collinear points are the roots of $\partial U/\partial x$ in $x < 0$, $0 < x < 1$ and $x > 1$.
In a component's local frame, L1 solves

$$
(q_i+1)x^5
-(2q_i+3)x^4
+(q_i+3)x^3
-q_i x^2
+2q_i x
-q_i
=0.
$$

At equal masses $x_{\mathrm{L1}} = 1/2$ and $U_{\mathrm{L1}} = 2$; at either extreme of mass ratio $U_{\mathrm{L1}}$ tends to $3/2$.
Critical points keep their relative accuracy very close to a component.
When L1 lies so close to `x = 1` that its coordinate cannot be represented, `l1()` raises `ValueError`.

Each call returns a frozen `CriticalPoint(position, potential, hessian)`.
The potential is a Jacobi value, and the Hessian is the symmetric 3 × 3 Hessian of `U` in the producer's axes, in units of $G(M_1+M_2)/a^3$.
Binary views rotate it, $R H R^\mathsf{T}$, but do not rescale it, so its eigenvalues are the same in every frame.
`eigenvalues` lists them in ascending order, and `morse_index` counts the negative ones: 2 for collinear points, 1 for triangular points.
Since `U` rises towards the components, collinear points rise along the centreline and fall across it, and triangular points rise in the orbital plane and fall out of it.

[A recipe](../recipes/l1-nozzle.md) uses the Hessian at L1 to find the L1 nozzle factor and the cone angle.
