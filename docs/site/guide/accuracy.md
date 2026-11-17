# Accuracy and speed

<!-- generated:summary -->
The release run measured commit `5bbbc38a569b` (source tree `2ca145c399cb`) on macOS-27.2.1-arm64-arm-64bit in 4.8 h with 8 workers.
The failure list has **0** entries.
Summaries: [`accuracy/results/`](https://github.com/pdhall99/rocheometry/tree/main/accuracy/results).
<!-- /generated:summary -->

Every release whose library source differs from the last release run's is measured again before it is published.

## Choosing an accuracy setting

Choose accuracy with a frozen, hashable value, shared by a binary's two components:

```pycon
>>> import rocheometry
>>>
>>> component = rocheometry.ComponentGeometry(0.25, accuracy=rocheometry.Reference())
>>> component.accuracy
Reference(polar_order=128, azimuth_order=64)
>>> binary = rocheometry.BinaryGeometry(
...     0.25, accuracy=rocheometry.Reference(polar_order=128, azimuth_order=96)
... )
>>> binary.geometry1.accuracy == binary.geometry2.accuracy == binary.accuracy
True

```

- **`Fitted()`**, the default, uses packaged fits where it can and numerical methods elsewhere, with 64 × 24 surface quadrature.
  Centreline sections have a fixed layout of 52 segments and silhouettes 88, so these outlines change smoothly in an animation.
  Axis-normal sections are traced adaptively and have variable segment counts.
- **`Reference(polar_order=128, azimuth_order=64)`** computes everything numerically and traces outlines adaptively.
  Its two quadrature orders, integers of at least 4, are its only settings.
  `Reference(polar_order=512, azimuth_order=192)` tightens the two mean-curvature scalars to the same `1e-10` bound as the rest.

`accuracy` is part of a geometry's equality, hash and repr.

## What is verified

The domain is the [classical Roche model](model.md), with component mass ratios `1e-8 <= qi <= 1e8`, at levels above L2.
Other mass ratios are accepted without a warning, but nothing here measures their accuracy.
Distances are relative to the measured volume radius `R_vol`.
Regimes are classified exactly, and volumes and fills are measured, not read from the fitted map.

### Public bounds

<!-- generated:bounds -->
| Quantity | Fitted | Reference |
| --- | ---: | ---: |
| Integrated mean curvature and mean-curvature radius | 10<sup>−6</sup> relative | 2 × 10<sup>−8</sup> relative at default orders |
| Other measured scalars | 10<sup>−6</sup> relative | 10<sup>−10</sup> relative |
| Selection by volume-radius fill | 2 × 10<sup>−6</sup> absolute fill | 10<sup>−13</sup> absolute fill |
| Sections and silhouettes | 2.5 × 10<sup>−4</sup> R_vol, two-way normal distance | 10<sup>−10</sup> R_vol, two-way normal distance |
| Lobe-radius logarithmic slope | 10<sup>−6</sup> absolute | 10<sup>−10</sup> absolute |

Fitted outlines are accepted at **2.25 × 10<sup>−4</sup> R_vol** in the release run.
<!-- /generated:bounds -->

Hessian eigenvalue errors are divided by the Hessian spectral norm.

## Quadrature node accuracy

The nodes of `surface.quadrature()` are checked differently from the quantities above.
Near L1 gravity vanishes and the surface normal is undefined, so a node's normal and gravity there change sharply with its position.
Comparing them with exact values would measure that sensitivity, not the code's error.

Instead, each node is checked to be the right answer for the position it reports:

- its radius along its ray satisfies the surface's equation to within the rounding error of the arithmetic that evaluates it, and each coordinate is that radius times the ray direction, rounded once;
- its normal and gravity are those of the exact potential at that rounded position, again to within the rounding error of the arithmetic that computes them.

This is a backward-error check, and it does not change the bounds above.

## How the contract is measured

Each study in `accuracy/contract/` declares its domain (frozen grids and holdouts in `accuracy/contract/grids/`), its metric, its bound and its truth: `Reference()` at stated orders, or an independent high-precision solve.
A release run measures every study fresh on a recorded machine; its summaries are committed under `accuracy/results/` and the tables below are generated from them.
A sampled maximum is not an analytic certificate for every input.
In the standard terms this is solution verification.

### Contract rows

<details markdown="1">
<summary>Every contract row: bound, worst measured value and case count</summary>

<!-- generated:contract -->
| Quantity | Accuracy setting | Study | Metric / units | Bound | Worst measured | Cases |
| --- | --- | --- | --- | ---: | ---: | ---: |
| binary.join | fitted | joins | distance / separation | 1e-12 | 5.72195849815e-17 | 272 |
| binary.join | reference | guarantees | distance / separation | 1e-12 | 5.59431511414e-17 | 272 |
| critical.l1.eigenvalues | fitted | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-06 | 2.98675530978e-14 | 17 |
| critical.l1.eigenvalues | reference | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-10 | 2.98675530978e-14 | 17 |
| critical.l1.hessian | fitted | critical_points | relative | 1e-06 | 3.19759873351e-14 | 17 |
| critical.l1.hessian | reference | critical_points | relative | 1e-10 | 3.19759873351e-14 | 17 |
| critical.l1.position | fitted | critical_points | relative | 1e-06 | 1.96870524803e-15 | 17 |
| critical.l1.position | reference | critical_points | relative | 1e-10 | 1.96870524803e-15 | 17 |
| critical.l1.potential | fitted | critical_points | relative | 1e-06 | 7.30828190143e-17 | 17 |
| critical.l1.potential | reference | critical_points | relative | 1e-10 | 7.30828190143e-17 | 17 |
| critical.outer.eigenvalues | fitted | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-06 | 3.67895146018e-15 | 17 |
| critical.outer.eigenvalues | reference | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-10 | 3.67895146018e-15 | 17 |
| critical.outer.hessian | fitted | critical_points | relative | 1e-06 | 3.98723419489e-15 | 17 |
| critical.outer.hessian | reference | critical_points | relative | 1e-10 | 3.98723419489e-15 | 17 |
| critical.outer.position | fitted | critical_points | relative | 1e-06 | 1.86142788861e-15 | 17 |
| critical.outer.position | reference | critical_points | relative | 1e-10 | 1.86142788861e-15 | 17 |
| critical.outer.potential | fitted | critical_points | relative | 1e-06 | 6.76762508488e-17 | 17 |
| critical.outer.potential | reference | critical_points | relative | 1e-10 | 6.76762508488e-17 | 17 |
| critical.triangular.eigenvalues | fitted | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-06 | 5.33695878129e-16 | 17 |
| critical.triangular.eigenvalues | reference | critical_points | maximum absolute eigenvalue error / spectral norm(H) | 1e-10 | 5.33695878129e-16 | 17 |
| critical.triangular.hessian | fitted | critical_points | relative | 1e-06 | 4.73655047185e-16 | 17 |
| critical.triangular.hessian | reference | critical_points | relative | 1e-10 | 4.73655047185e-16 | 17 |
| critical.triangular.position | fitted | critical_points | relative | 1e-06 | 5.0175421109e-17 | 17 |
| critical.triangular.position | reference | critical_points | relative | 1e-10 | 5.0175421109e-17 | 17 |
| critical.triangular.potential | fitted | critical_points | relative | 1e-06 | 5.70569453972e-17 | 17 |
| critical.triangular.potential | reference | critical_points | relative | 1e-10 | 5.70569453972e-17 | 17 |
| integrals.area | fitted | scalars | relative | 1e-06 | 1.41580080992e-11 | 1928 |
| integrals.area | reference | saddle_quadrature | relative | 1e-10 | 5.06261699229e-14 | 1137 |
| integrals.area | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 5.97299987248e-14 | 159 |
| integrals.area.radius | fitted | scalars | relative | 1e-06 | 7.07878200501e-12 | 1860 |
| integrals.area.radius | reference | saddle_quadrature | relative | 1e-10 | 2.53130849615e-14 | 1069 |
| integrals.area.radius | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 2.99760216649e-14 | 159 |
| integrals.fill | fitted | joint_holdout | relative | 1e-06 | 3.24096305349e-12 | 468 |
| integrals.fill | reference | independent_truth | relative | 1e-10 | 2.73114864058e-14 | 468 |
| integrals.fill.at_potential | fitted | scalars | absolute | 1e-06 | 3.26161320174e-12 | 1460 |
| integrals.mean_curvature | fitted | joint_holdout | relative | 1e-06 | 2.29919992156e-07 | 1928 |
| integrals.mean_curvature | reference | saddle_quadrature | relative | 2e-08 | 1.45738270341e-08 | 1137 |
| integrals.mean_curvature | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 4.37816449761e-11 | 159 |
| integrals.mean_curvature.radius | fitted | joint_holdout | relative | 1e-06 | 2.29919992156e-07 | 1860 |
| integrals.mean_curvature.radius | reference | saddle_quadrature | relative | 2e-08 | 1.45738270341e-08 | 1069 |
| integrals.mean_curvature.radius | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 4.37816449761e-11 | 159 |
| integrals.mean_gravity | fitted | scalars | relative | 1e-06 | 1.33072441955e-11 | 1928 |
| integrals.mean_gravity | reference | joint_holdout | relative | 1e-10 | 2.45359288442e-14 | 1137 |
| integrals.mean_gravity | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 4.4408920985e-16 | 159 |
| integrals.mean_gravity.gauss_law | fitted | scalars | relative | 1e-06 | 6.59805543535e-13 | 1860 |
| integrals.mean_gravity.gauss_law | reference | joint_holdout | relative | 1e-10 | 1.36557432029e-14 | 1860 |
| integrals.volume | fitted | joint_holdout | relative | 1e-06 | 9.7249985842e-12 | 5199 |
| integrals.volume | fitted | independent_truth | relative (both components) | 1e-06 | 1.72084568817e-14 | 17 |
| integrals.volume | reference | saddle_quadrature | relative | 1e-10 | 5.68434188608e-14 | 1137 |
| integrals.volume | reference | independent_truth | relative (both components) | 1e-10 | 9.99200722163e-16 | 17 |
| integrals.volume | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 6.43929354283e-14 | 159 |
| integrals.volume.radius | fitted | joint_holdout | relative | 1e-06 | 3.2417402096e-12 | 5131 |
| integrals.volume.radius | reference | saddle_quadrature | relative | 1e-10 | 1.88737914186e-14 | 1069 |
| integrals.volume.radius | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 2.15383266777e-14 | 159 |
| integrals.volume.sampled | fitted | joint_holdout | relative | 1e-06 | 9.7249985842e-12 | 1860 |
| integrals.volume.sampled | reference | saddle_quadrature | relative | 1e-10 | 5.68434188608e-14 | 1069 |
| integrals.volume.sampled | reference | saddle_quadrature | relative at 512x192 | 1e-10 | 6.43929354283e-14 | 159 |
| lobe.max_fill | fitted | independent_truth | relative | 1e-06 | 2.36242203577e-09 | 1876 |
| lobe.max_fill | reference | independent_truth | relative | 1e-10 | 5.39901456875e-13 | 922 |
| lobe.slope | fitted | joint_holdout | absolute | 1e-06 | 2.37115360679e-08 | 1877 |
| lobe.slope | reference | independent_truth | absolute | 1e-10 | 3.30957483641e-13 | 927 |
| lobe.volume_radius | fitted | scalars | relative | 1e-06 | 1.50990331349e-13 | 1477 |
| lobe.volume_radius | reference | independent_truth | relative | 1e-10 | 2.45359288442e-14 | 527 |
| lobe.volume_radius.fast | fitted | scalars | relative | 1e-06 | 3.23699400617e-09 | 1877 |
| lobe.volume_radius.fast | reference | independent_truth | relative | 1e-10 | 2.45359288442e-14 | 927 |
| outlines.section | fitted | outlines | two-way normal distance / R_vol | 0.000225 | 0.000186552454672 | 18980 |
| outlines.section | reference | outlines | two-way normal distance / R_vol | 1e-10 | 2.3466174307e-11 | 18980 |
| outlines.section_envelope | fitted | joint_holdout | two-way normal distance / R_vol | 0.000225 | 0.000207017966329 | 19380 |
| outlines.section_envelope | reference | outlines | two-way normal distance / R_vol | 1e-10 | 2.3466174307e-11 | 19380 |
| outlines.shape.L1_continuity | fitted | guarantees | control difference / pole radius | 0.001 | 0.000348767179449 | 136 |
| outlines.shape.L1_rate | fitted | guarantees | ratio | 0.4 | 0.350369480375 | 136 |
| outlines.shape.cone_tangent | fitted | guarantees | radians | 1e-08 | 8.29836199756e-13 | 204 |
| outlines.shape.cone_tangent | reference | guarantees | radians | 1e-08 | 6.46704911844e-13 | 204 |
| outlines.shape.knots | fitted | guarantees | prescribed-ray radius relative | 1e-12 | 1.84297022088e-13 | 816 |
| outlines.shape.knots | reference | guarantees | normal distance / R_vol | 1e-10 | 4.12665816538e-12 | 816 |
| outlines.shape.reference_knots.section | reference | outlines | normal distance / R_vol | 1e-10 | 5.42437783282e-12 | 18980 |
| outlines.shape.reference_knots.section_envelope | reference | outlines | normal distance / R_vol | 1e-10 | 5.42437783282e-12 | 18980 |
| outlines.shape.reference_knots.silhouette | reference | outlines | normal distance / R_vol | 1e-10 | 3.14983170484e-12 | 3761 |
| outlines.shape.reference_knots.silhouette_envelope | reference | outlines | normal distance / R_vol | 1e-10 | 5.42437783282e-12 | 7302 |
| outlines.silhouette | fitted | outlines | two-way normal distance / R_vol | 0.000225 | 0.000166436666251 | 3761 |
| outlines.silhouette | reference | outlines | two-way normal distance / R_vol | 1e-10 | 2.37778797382e-11 | 3761 |
| outlines.silhouette_envelope | fitted | outlines | two-way normal distance / R_vol | 0.000225 | 0.000171739190891 | 8152 |
| outlines.silhouette_envelope | reference | outlines | two-way normal distance / R_vol | 1e-10 | 2.37778797382e-11 | 7702 |
| quadrature.integrals.area_consistency | fitted | quadrature | relative | 1e-13 | 3.99680288865e-15 | 1860 |
| quadrature.integrals.area_consistency | reference | joint_holdout | relative | 1e-13 | 1.37667655054e-14 | 1860 |
| quadrature.integrals.curvature_consistency | fitted | quadrature | relative | 1e-13 | 4.66293670343e-15 | 1860 |
| quadrature.integrals.curvature_consistency | reference | quadrature | relative | 1e-13 | 9.32587340685e-15 | 1860 |
| quadrature.integrals.gravity4 | fitted | quadrature | relative | 1e-06 | 6.85451695404e-13 | 1928 |
| quadrature.integrals.gravity4 | reference | independent_truth | relative | 1e-10 | 4.59632332195e-14 | 978 |
| quadrature.integrals.gravity_consistency | fitted | joint_holdout | relative | 1e-13 | 0 | 1860 |
| quadrature.integrals.gravity_consistency | reference | joint_holdout | relative | 1e-13 | 0 | 1860 |
| quadrature.integrals.x2 | fitted | quadrature | relative | 1e-06 | 5.75683944959e-11 | 1928 |
| quadrature.integrals.x2 | reference | joint_holdout | relative | 1e-10 | 1.01696429056e-13 | 978 |
| quadrature.integrals.z2 | fitted | quadrature | relative | 1e-06 | 9.02389274415e-13 | 1928 |
| quadrature.integrals.z2 | reference | quadrature | relative | 1e-10 | 1.45439216226e-14 | 978 |
| quadrature.nodes.gradient_backward | fitted | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.561390314756 | 400 |
| quadrature.nodes.gradient_backward | fitted | quadrature | represented-gradient error / running component bound | 1 | 0.569191024777 | 1460 |
| quadrature.nodes.gradient_backward | reference | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.67058772176 | 400 |
| quadrature.nodes.gradient_backward | reference | quadrature | represented-gradient error / running component bound | 1 | 0.642340148195 | 1970 |
| quadrature.nodes.gravity_backward | fitted | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.624978043075 | 400 |
| quadrature.nodes.gravity_backward | fitted | quadrature | magnitude error / running hypot bound | 1 | 0.625486924769 | 1460 |
| quadrature.nodes.gravity_backward | reference | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.694640721755 | 400 |
| quadrature.nodes.gravity_backward | reference | quadrature | magnitude error / running hypot bound | 1 | 0.688802134349 | 1970 |
| quadrature.nodes.level_backward | fitted | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.440520038053 | 400 |
| quadrature.nodes.level_backward | fitted | quadrature | exact-ray level offset / running error bound | 1 | 0.508521166329 | 1460 |
| quadrature.nodes.level_backward | reference | joint_holdout | exact-ray/float-gradient error / running bound | 1 | 0.432521380706 | 400 |
| quadrature.nodes.level_backward | reference | quadrature | exact-ray level offset / running error bound | 1 | 0.460781273562 | 1970 |
| quadrature.nodes.unit | fitted | joint_holdout | absolute | 1e-12 | 2.22044604925e-16 | 1860 |
| quadrature.nodes.unit | reference | joint_holdout | absolute | 1e-12 | 2.22044604925e-16 | 2370 |
| radii.direction | fitted | scalars | relative | 1e-06 | 1.23356880266e-12 | 1860 |
| radii.direction | reference | radii | relative | 1e-10 | 1.14186438083e-12 | 655 |
| radii.named.back | fitted | scalars | relative | 1e-06 | 1.23356880266e-12 | 1860 |
| radii.named.back | reference | radii | relative | 1e-10 | 1.14186438083e-12 | 655 |
| radii.named.point | fitted | scalars | relative | 1e-06 | 5.6399329651e-14 | 1860 |
| radii.named.point | reference | radii | relative | 1e-10 | 2.62012633812e-14 | 655 |
| radii.named.pole | fitted | scalars | relative | 1e-06 | 9.54791801178e-14 | 1860 |
| radii.named.pole | reference | radii | relative | 1e-10 | 4.5852210917e-14 | 655 |
| radii.named.side | fitted | scalars | relative | 1e-06 | 1.38666855776e-13 | 1860 |
| radii.named.side | reference | radii | relative | 1e-10 | 3.8413716652e-14 | 655 |
| selection.by_fill | fitted | joint_holdout | absolute | 2e-06 | 4.56458484344e-08 | 100 |
| selection.by_fill | fitted | selection | absolute in requested fill | 2e-06 | 1.70955680279e-06 | 3271 |
| selection.by_fill | reference | joint_holdout | absolute | 1e-13 | 4.04121180964e-14 | 100 |
| selection.by_fill.depth | fitted | joint_holdout | absolute | 1e-06 | 0 | 100 |
| selection.by_fill.depth | reference | joint_holdout | absolute | 1e-10 | 0 | 100 |
| selection.by_fill.measured | fitted | selection | relative | 1e-06 | 1.57873714102e-13 | 3271 |
| selection.by_fill.round_trip | fitted | scalars | absolute | 2e-06 | 4.56410405247e-07 | 752 |
| selection.by_fill.round_trip | reference | scalars | absolute | 1e-13 | 8.65973959208e-14 | 752 |
| selection.by_fill_out_factor | fitted | joint_holdout | absolute | 1e-06 | 1.11022302463e-16 | 200 |
| selection.by_fill_out_factor | reference | joint_holdout | absolute | 1e-10 | 1.11022302463e-16 | 200 |
| selection.by_fill_out_factor.round_trip | fitted | scalars | absolute | 1e-06 | 1.11022302463e-16 | 708 |
| selection.by_fill_out_factor.round_trip | reference | scalars | absolute | 1e-10 | 1.11022302463e-16 | 708 |
| selection.fill_for_radius | fitted | scalars | absolute | 1e-06 | 3.24282600772e-09 | 1460 |
| selection.fill_for_radius | reference | scalars | absolute | 1e-10 | 2.22044604925e-16 | 1460 |
<!-- /generated:contract -->

</details>

### Comparisons with published results

Our solution against values others have published for the same model.
The bound is the publication's rounding, so these rows test definitions and conventions, not accuracy; see [accuracy/literature](https://github.com/pdhall99/rocheometry/tree/main/accuracy/literature).

<!-- generated:literature -->
| Quantity | Accuracy setting | Metric / units | Cases | Closest to its bound | Value / bound | Source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| critical.l1.hessian | reference | half-units of the last printed digit | 46 | 1.54 of 2 | 0.77 | Plavec & Kratochvil 1964, Table 1 |
| critical.l1.position | reference | half-units of the last printed digit | 448 | 0.993 of 1 | 0.99 | Leahy & Leahy 2015, RocheTable.txt |
| critical.l1.position | reference | relative | 17 | 1.97e-15 of 1e-10 | 0.00 | Seidov 2004, ApJ, 603, 283 (2004ApJ...603..283S; arXiv:astro-ph/0311272) |
| critical.l1.potential | reference | half-units of the last printed digit | 382 | 1.49 of 1.5 | 0.99 | Kuiper & Johnson 1956, Table 1 |
| critical.l1.potential | reference | relative | 17 | 1.64e-16 of 1e-10 | 0.00 | Seidov 2004, ApJ, 603, 283 (2004ApJ...603..283S; arXiv:astro-ph/0311272) |
| critical.outer.position | reference | half-units of the last printed digit | 616 | 0.996 of 1 | 1.00 | Leahy & Leahy 2015, RocheTable.txt |
| critical.outer.position | reference | relative | 17 | 1.88e-15 of 1e-10 | 0.00 | Seidov 2004, ApJ, 603, 283 (2004ApJ...603..283S; arXiv:astro-ph/0311272) |
| critical.outer.potential | reference | half-units of the last printed digit | 221 | 1.33 of 1.5 | 0.89 | Kuiper & Johnson 1956, Table 2 |
| critical.outer.potential | reference | relative | 17 | 1.56e-16 of 1e-10 | 0.00 | Seidov 2004, ApJ, 603, 283 (2004ApJ...603..283S; arXiv:astro-ph/0311272) |
| integrals.area | reference | half-units of the last printed digit | 320 | 4.39 of 4.5 | 0.98 | Mochnacki 1984, Table 9 |
| integrals.area.envelope | reference | half-units of the last printed digit | 144 | 2.85 of 4 | 0.71 | Mochnacki 1984, Table 10 |
| integrals.mean_gravity | reference | half-units of the last printed digit | 320 | 2.96 of 4 | 0.74 | Mochnacki 1984, Table 13 |
| integrals.mean_gravity.envelope | reference | half-units of the last printed digit | 144 | 3.41 of 4 | 0.85 | Mochnacki 1984, Table 14 |
| integrals.volume | reference | half-units of the last printed digit | 571 | 1.97 of 2 | 0.98 | Mochnacki 1984, Table 4 |
| integrals.volume.envelope | reference | half-units of the last printed digit | 144 | 1.11 of 1.5 | 0.74 | Mochnacki 1984, Table 6 |
| lobe.max_fill.envelope_volume | reference | half-units of the last printed digit | 16 | 0.854 of 1.5 | 0.57 | Mochnacki 1984, Table 6 |
| lobe.slope | reference | absolute | 10 | 1.13e-07 of 3.68e-07 | 0.31 | Guillochon & Prochaska 2026 (manuscript), eqs. hillexp and logexp |
| lobe.volume_radius | reference | relative | 27 | 0.00751 of 0.01 | 0.75 | Eggleton 1983, ApJ, 268, 368 (1983ApJ...268..368E) |
| lobe.volume_radius | reference | half-units of the last printed digit | 226 | 1.55 of 2.17 | 0.71 | Leahy & Leahy 2015, RocheTable.txt |
| lobe.volume_radius.limit | fitted | relative | 2 | 2.63e-13 of 1e-10 | 0.00 | Guillochon & Prochaska 2026 (manuscript), eqs. hillexp and logexp |
| outlines.section.extent | reference | half-units of the last printed digit | 120 | 12.6 of 13 | 0.97 | Kopal 1959, Table 3-1 |
| potential.conventions.kopal | reference | half-units of the last printed digit | 550 | 1 of 1 | 1.00 | Leahy & Leahy 2015, RocheTable.txt |
| radii.direction | reference | half-units of the last printed digit | 124 | 0.967 of 1 | 0.97 | Leahy & Leahy 2015, Table 1 |
| radii.named | reference | half-units of the last printed digit | 811 | 20.9 of 21 | 1.00 | Leahy & Leahy 2015, RocheLobe.f90 |
| selection.by_fill_out_factor | reference | half-units of the last printed digit | 144 | 1.05 of 1.5 | 0.70 | Mochnacki 1984, Table 3 |
<!-- /generated:literature -->

### Diagnostic rows

These measurements do not gate the public forward contracts.
Each row states what it diagnoses.

<details markdown="1">
<summary>Diagnostic rows</summary>

<!-- generated:diagnostics -->
| Quantity | Accuracy setting | Study | Metric / units | Worst measured | Explanation |
| --- | --- | --- | --- | ---: | --- |
| critical.l1.eigenvalues_relative | fitted | critical_points | relative | 4.47763121437e-14 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| critical.l1.eigenvalues_relative | reference | critical_points | relative | 4.47763121437e-14 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| critical.outer.eigenvalues_relative | fitted | critical_points | relative | 3.64137546942e-07 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| critical.outer.eigenvalues_relative | reference | critical_points | relative | 3.64137546942e-07 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| critical.triangular.eigenvalues_relative | fitted | critical_points | relative | 1.33147693831e-08 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| critical.triangular.eigenvalues_relative | reference | critical_points | relative | 1.33147693831e-08 | Individual near-zero eigenvalues are ill-conditioned; the contract uses the Hessian spectral norm. |
| outlines.shape.corner_rounding | fitted | outline_corners | distance / R_vol | 0.000102816346115 | Corner displacement is included in the two-way outline bound; it has no separate contract. |
| outlines.shape.fill_second_difference | fitted | guarantees | control deviation / R_vol | 0.0260642670116 | Sampled second differences describe continuity on the recorded grid, without a universal bound. |
| outlines.shape.knots_normal | fitted | guarantees | normal distance / R_vol | 3.9341868679e-12 | Normal displacement of a fitted knot near a neck is unbounded; its radial guarantee is checked separately. |
| outlines.shape.q_second_difference | fitted | continuity | control second difference / R_vol | 0.000179364672985 | Sampled second differences describe continuity on the recorded grid, without a universal bound. |
| outlines.shape.q_second_difference | fitted | continuity | relative | None | Sampled second differences describe continuity on the recorded grid, without a universal bound. |
| quadrature.integrals.gauss_bonnet | fitted | scalars | absolute | 6.02726872287e-05 | Integrated Gaussian curvature diagnoses topology and apex conditioning; the critical guard is separate. |
| quadrature.integrals.gauss_bonnet | reference | scalars | absolute | 8.68782283447e-06 | Integrated Gaussian curvature diagnoses topology and apex conditioning; the critical guard is separate. |
| quadrature.nodes.displacement | fitted | joint_holdout | relative | 5.92551949655e-12 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.displacement | fitted | quadrature | normal distance / R_vol | 8.20123455688e-12 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.displacement | reference | joint_holdout | relative | 5.65429567639e-12 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.displacement | reference | quadrature | normal distance / R_vol | 9.79746211807e-12 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.near_displacement | fitted | quadrature | normal distance / R_vol | 5.6639318209e-14 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.near_displacement | reference | quadrature | normal distance / R_vol | 5.76508062516e-14 | Forward node displacement is diagnostic; acceptance uses the exact-ray residual running bound. |
| quadrature.nodes.near_normal_angle | fitted | quadrature | radians | 6.87141126434e-08 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
| quadrature.nodes.near_normal_angle | reference | quadrature | radians | 9.77738363171e-07 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
| quadrature.nodes.normal_angle | fitted | joint_holdout | relative | 5.53920519949e-08 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
| quadrature.nodes.normal_angle | fitted | quadrature | radians | 6.97548139604e-09 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
| quadrature.nodes.normal_angle | reference | joint_holdout | relative | 5.21920220765e-07 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
| quadrature.nodes.normal_angle | reference | quadrature | radians | 6.03875619545e-09 | Forward node-normal error is ill-conditioned near saddles; running backward bounds determine acceptance. |
<!-- /generated:diagnostics -->

</details>

## Speed

Each operation is timed cold, on a fresh geometry; at a new level of an existing geometry; and cached, repeating the same call.
These are medians on one machine, not speed guarantees.

<details markdown="1">
<summary>Timings for every measured mass ratio and level</summary>

<!-- generated:timings -->
Medians of 5 draws in milliseconds on macOS-27.2.1-arm64-arm-64bit. Cold is a fresh geometry with empty caches; new level is a second level at the same mass ratio; cached repeats the new-level call.

| Accuracy setting | qi | Regime | Operation | Cold ms | New level ms | Cached ms |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| fitted | 0.001 | contact | fill | 232.898 | 30.3998 | 0.000540982 |
| fitted | 0.001 | contact | sample | 210.576 | 27.8208 | 0.000500004 |
| fitted | 0.001 | contact | section | 185.529 | 2.79296 | 0.623042 |
| fitted | 0.001 | contact | selection | 328.823 | 90.5132 | 0.00483304 |
| fitted | 0.001 | contact | silhouette_60 | 198.403 | 15.3069 | 7.916 |
| fitted | 0.001 | contact | silhouette_85 | 199.046 | 15.7843 | 8.35021 |
| fitted | 0.001 | contact | volume | 212.474 | 30.0412 | 0.000583008 |
| fitted | 0.001 | detached | fill | 33.5423 | 13.3653 | 0.000583008 |
| fitted | 0.001 | detached | lobe_radius | 0.00245799 | — | 0.000374974 |
| fitted | 0.001 | detached | lobe_slope | 0.00187504 | — | 0.00029098 |
| fitted | 0.001 | detached | sample | 12.989 | 12.6992 | 0.000500004 |
| fitted | 0.001 | detached | section | 0.422375 | 0.41525 | 0.404 |
| fitted | 0.001 | detached | selection | 0.208958 | 0.010375 | 0.00200002 |
| fitted | 0.001 | detached | silhouette_60 | 6.95142 | 6.83079 | 6.82683 |
| fitted | 0.001 | detached | silhouette_85 | 8.75629 | 8.75096 | 8.74104 |
| fitted | 0.001 | detached | volume | 13.5923 | 12.9831 | 0.00083301 |
| fitted | 1 | contact | fill | 260.342 | 24.6464 | 0.000500004 |
| fitted | 1 | contact | sample | 242.283 | 22.3362 | 0.00054104 |
| fitted | 1 | contact | section | 223.095 | 3.00358 | 0.5655 |
| fitted | 1 | contact | selection | 338.222 | 73.2535 | 0.00462501 |
| fitted | 1 | contact | silhouette_60 | 229.39 | 9.54008 | 8.34438 |
| fitted | 1 | contact | silhouette_85 | 230.007 | 10.7375 | 9.56025 |
| fitted | 1 | contact | volume | 244.294 | 24.5898 | 0.000667002 |
| fitted | 1 | detached | fill | 27.6305 | 11.3223 | 0.000500004 |
| fitted | 1 | detached | lobe_radius | 0.00308303 | — | 0.000333006 |
| fitted | 1 | detached | lobe_slope | 0.00204198 | — | 0.00029098 |
| fitted | 1 | detached | sample | 11.114 | 10.7714 | 0.000541972 |
| fitted | 1 | detached | section | 0.40975 | 0.396625 | 0.389791 |
| fitted | 1 | detached | selection | 0.086541 | 0.00895804 | 0.00183395 |
| fitted | 1 | detached | silhouette_60 | 6.31004 | 6.268 | 6.25817 |
| fitted | 1 | detached | silhouette_85 | 8.50304 | 8.48079 | 8.44083 |
| fitted | 1 | detached | volume | 11.207 | 11.2519 | 0.000500004 |
| fitted | 1000 | contact | fill | 228.727 | 29.436 | 0.000500004 |
| fitted | 1000 | contact | sample | 209.091 | 26.9115 | 0.00054203 |
| fitted | 1000 | contact | section | 184.744 | 2.82563 | 0.647417 |
| fitted | 1000 | contact | selection | 287.632 | 58.5029 | 0.00462501 |
| fitted | 1000 | contact | silhouette_60 | 197.627 | 14.4816 | 11.2269 |
| fitted | 1000 | contact | silhouette_85 | 201.726 | 18.8237 | 15.4692 |
| fitted | 1000 | contact | volume | 211.539 | 29.2584 | 0.000583008 |
| fitted | 1000 | detached | fill | 30.4311 | 13.193 | 0.000458036 |
| fitted | 1000 | detached | lobe_radius | 0.00216597 | — | 0.000333006 |
| fitted | 1000 | detached | lobe_slope | 0.00195898 | — | 0.000250002 |
| fitted | 1000 | detached | sample | 12.9857 | 12.7122 | 0.00054104 |
| fitted | 1000 | detached | section | 0.431916 | 0.430375 | 0.417875 |
| fitted | 1000 | detached | selection | 0.198459 | 0.00920799 | 0.00191701 |
| fitted | 1000 | detached | silhouette_60 | 8.41392 | 8.32242 | 8.338 |
| fitted | 1000 | detached | silhouette_85 | 11.5565 | 11.5167 | 11.5094 |
| fitted | 1000 | detached | volume | 13.0914 | 13.0575 | 0.000540982 |
| fitted | 1e+06 | contact | fill | 175.711 | 33.456 | 0.00054203 |
| fitted | 1e+06 | contact | sample | 156.551 | 30.9905 | 0.00054203 |
| fitted | 1e+06 | contact | section | 128.032 | 2.90171 | 0.730667 |
| fitted | 1e+06 | contact | selection | 208.695 | 33.2608 | 0.00487495 |
| fitted | 1e+06 | contact | silhouette_60 | 144.612 | 18.0448 | 14.7522 |
| fitted | 1e+06 | contact | silhouette_85 | 146.7 | 20.2327 | 16.9468 |
| fitted | 1e+06 | contact | volume | 158.84 | 33.3683 | 0.000624976 |
| fitted | 1e+06 | detached | fill | 29.827 | 13.158 | 0.000500004 |
| fitted | 1e+06 | detached | lobe_radius | 0.00200002 | — | 0.000333996 |
| fitted | 1e+06 | detached | lobe_slope | 0.00204204 | — | 0.000250002 |
| fitted | 1e+06 | detached | sample | 13.0268 | 12.8243 | 0.000457978 |
| fitted | 1e+06 | detached | section | 0.452541 | 0.432083 | 0.443834 |
| fitted | 1e+06 | detached | selection | 0.1915 | 0.00887498 | 0.00179204 |
| fitted | 1e+06 | detached | silhouette_60 | 8.33925 | 8.30925 | 8.31704 |
| fitted | 1e+06 | detached | silhouette_85 | 11.5886 | 11.549 | 11.5178 |
| fitted | 1e+06 | detached | volume | 13.0333 | 13.0533 | 0.000500004 |
| fitted | 1e+08 | contact | fill | 140.894 | 36.5376 | 0.00058295 |
| fitted | 1e+08 | contact | sample | 121.47 | 34.0078 | 0.00054203 |
| fitted | 1e+08 | contact | section | 90.6002 | 3.28892 | 0.789625 |
| fitted | 1e+08 | contact | selection | 140.934 | 0.00804104 | 0.00233302 |
| fitted | 1e+08 | contact | silhouette_60 | 107.729 | 19.1775 | 15.6856 |
| fitted | 1e+08 | contact | silhouette_85 | 109.671 | 20.9829 | 17.5147 |
| fitted | 1e+08 | contact | volume | 123.734 | 36.226 | 0.000666012 |
| fitted | 1e+08 | detached | fill | 30.1105 | 13.0801 | 0.000458036 |
| fitted | 1e+08 | detached | lobe_radius | 0.00200002 | — | 0.000333006 |
| fitted | 1e+08 | detached | lobe_slope | 0.00154204 | — | 0.000250002 |
| fitted | 1e+08 | detached | sample | 13.0014 | 12.6757 | 0.000500004 |
| fitted | 1e+08 | detached | section | 0.436 | 0.427333 | 0.42175 |
| fitted | 1e+08 | detached | selection | 0.193459 | 0.00912498 | 0.00187499 |
| fitted | 1e+08 | detached | silhouette_60 | 8.38492 | 8.2785 | 8.26675 |
| fitted | 1e+08 | detached | silhouette_85 | 11.5645 | 11.5132 | 11.4822 |
| fitted | 1e+08 | detached | volume | 13.2468 | 13.1725 | 0.00054203 |
| reference | 0.001 | contact | fill | 742.688 | 153.575 | 0.000583008 |
| reference | 0.001 | contact | sample | 597.086 | 141.291 | 0.000583008 |
| reference | 0.001 | contact | section | 780.847 | 324.09 | 290.804 |
| reference | 0.001 | contact | selection | 1379.96 | 611.874 | 0.00583299 |
| reference | 0.001 | contact | silhouette_60 | 591.97 | 135.546 | 115.393 |
| reference | 0.001 | contact | silhouette_85 | 599.027 | 142.99 | 125.024 |
| reference | 0.001 | contact | volume | 607.927 | 153.325 | 0.00070798 |
| reference | 0.001 | detached | fill | 238.71 | 105.37 | 0.00054203 |
| reference | 0.001 | detached | lobe_radius | 133.231 | — | 0.000333006 |
| reference | 0.001 | detached | lobe_slope | 141.808 | — | 0.000250002 |
| reference | 0.001 | detached | sample | 104.844 | 104.499 | 0.000791973 |
| reference | 0.001 | detached | section | 8.28479 | 8.32554 | 8.36775 |
| reference | 0.001 | detached | selection | 755.732 | 526.209 | 0.00424997 |
| reference | 0.001 | detached | silhouette_60 | 80.0076 | 79.8709 | 80.0328 |
| reference | 0.001 | detached | silhouette_85 | 100.942 | 100.861 | 100.81 |
| reference | 0.001 | detached | volume | 105.297 | 105.1 | 0.000790984 |
| reference | 1 | contact | fill | 778.848 | 123.453 | 0.000540982 |
| reference | 1 | contact | sample | 674.912 | 112.583 | 0.000792032 |
| reference | 1 | contact | section | 564.714 | 12.1392 | 10.9496 |
| reference | 1 | contact | selection | 1293.98 | 488.993 | 0.00562501 |
| reference | 1 | contact | silhouette_60 | 653.352 | 101.644 | 100.356 |
| reference | 1 | contact | silhouette_85 | 664.257 | 115.343 | 113.9 |
| reference | 1 | contact | volume | 673.329 | 123.088 | 0.00070798 |
| reference | 1 | detached | fill | 187.038 | 83.7344 | 0.000583997 |
| reference | 1 | detached | lobe_radius | 103.239 | — | 0.000332948 |
| reference | 1 | detached | lobe_slope | 111.471 | — | 0.000291038 |
| reference | 1 | detached | sample | 83.2901 | 82.6566 | 0.000667002 |
| reference | 1 | detached | section | 8.42871 | 8.38675 | 8.37258 |
| reference | 1 | detached | selection | 604.284 | 335.271 | 0.00429101 |
| reference | 1 | detached | silhouette_60 | 82.3182 | 82.3558 | 82.2969 |
| reference | 1 | detached | silhouette_85 | 107.497 | 107.186 | 107.199 |
| reference | 1 | detached | volume | 83.8585 | 83.8369 | 0.000667002 |
| reference | 1000 | contact | fill | 721.527 | 149.193 | 0.000624976 |
| reference | 1000 | contact | sample | 592.281 | 137.35 | 0.000666012 |
| reference | 1000 | contact | section | 631.312 | 168.311 | 18.8031 |
| reference | 1000 | contact | selection | 1166.97 | 445.287 | 0.00554201 |
| reference | 1000 | contact | silhouette_60 | 733.505 | 277.654 | 259.89 |
| reference | 1000 | contact | silhouette_85 | 756.836 | 301.136 | 285.107 |
| reference | 1000 | contact | volume | 604.201 | 148.819 | 0.000792032 |
| reference | 1000 | detached | fill | 220.494 | 104.438 | 0.000625034 |
| reference | 1000 | detached | lobe_radius | 115.858 | — | 0.000333996 |
| reference | 1000 | detached | lobe_slope | 123.92 | — | 0.000250002 |
| reference | 1000 | detached | sample | 103.854 | 103.493 | 0.000791973 |
| reference | 1000 | detached | section | 8.835 | 8.82329 | 8.79933 |
| reference | 1000 | detached | selection | 737.504 | 416.472 | 0.00395905 |
| reference | 1000 | detached | silhouette_60 | 100.105 | 99.6708 | 100.159 |
| reference | 1000 | detached | silhouette_85 | 129.407 | 129.323 | 129.171 |
| reference | 1000 | detached | volume | 104.135 | 104.121 | 0.000667002 |
| reference | 1e+06 | contact | fill | 602.228 | 170.578 | 0.00054104 |
| reference | 1e+06 | contact | sample | 471.773 | 158.449 | 0.000667002 |
| reference | 1e+06 | contact | section | 380.67 | 65.3788 | 20.2764 |
| reference | 1e+06 | contact | selection | 947.422 | 339.479 | 0.00504102 |
| reference | 1e+06 | contact | silhouette_60 | 813.602 | 499.006 | 495.518 |
| reference | 1e+06 | contact | silhouette_85 | 1413.07 | 1101.16 | 1090.87 |
| reference | 1e+06 | contact | volume | 482.276 | 169.983 | 0.000667002 |
| reference | 1e+06 | detached | fill | 223.544 | 104.746 | 0.000916014 |
| reference | 1e+06 | detached | lobe_radius | 118.555 | — | 0.000333006 |
| reference | 1e+06 | detached | lobe_slope | 126.742 | — | 0.000250002 |
| reference | 1e+06 | detached | sample | 104.476 | 103.589 | 0.000750006 |
| reference | 1e+06 | detached | section | 8.93608 | 8.84808 | 8.8785 |
| reference | 1e+06 | detached | selection | 637.197 | 313.037 | 0.00333303 |
| reference | 1e+06 | detached | silhouette_60 | 99.3343 | 99.4445 | 99.3516 |
| reference | 1e+06 | detached | silhouette_85 | 132.751 | 132.489 | 132.741 |
| reference | 1e+06 | detached | volume | 104.242 | 104.472 | 0.000750006 |
| reference | 1e+08 | contact | fill | 524.693 | 185.869 | 0.000541972 |
| reference | 1e+08 | contact | sample | 389.161 | 173.692 | 0.000500004 |
| reference | 1e+08 | contact | section | 253.006 | 30.8289 | 20.7393 |
| reference | 1e+08 | contact | selection | 897.095 | 369.125 | 0.00483298 |
| reference | 1e+08 | contact | silhouette_60 | 785.232 | 569.548 | 561.91 |
| reference | 1e+08 | contact | silhouette_85 | 1133.36 | 923.299 | 1054.24 |
| reference | 1e+08 | contact | volume | 402.093 | 185.403 | 0.00083301 |
| reference | 1e+08 | detached | fill | 227.754 | 104.581 | 0.00054203 |
| reference | 1e+08 | detached | lobe_radius | 126.015 | — | 0.000333006 |
| reference | 1e+08 | detached | lobe_slope | 131.78 | — | 0.000250002 |
| reference | 1e+08 | detached | sample | 107.262 | 104.279 | 0.00125001 |
| reference | 1e+08 | detached | section | 8.9035 | 8.85429 | 8.85671 |
| reference | 1e+08 | detached | selection | 664.386 | 317.815 | 0.00508298 |
| reference | 1e+08 | detached | silhouette_60 | 99.0371 | 99.1544 | 99.1223 |
| reference | 1e+08 | detached | silhouette_85 | 131.984 | 132.253 | 132.059 |
| reference | 1e+08 | detached | volume | 104.428 | 104.675 | 0.000750006 |
<!-- /generated:timings -->

</details>
