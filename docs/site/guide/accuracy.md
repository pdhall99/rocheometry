# Accuracy and speed

<!-- generated:summary -->
The release run measured commit `d1250649b67b` (source tree `2cfc3eccae32`) on macOS-26.6.1-arm64-arm-64bit in 4.8 h with 8 workers.
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
| radii.direction | reference | absolute, separation units | 2 | 1.55e-08 of 5e-08 | 0.31 | Leahy & Leahy (2015), Table 1 |
| radii.named.point | reference | absolute, separation units | 1 | 0 of 5e-08 | 0.00 | Leahy & Leahy (2015), Table 1 |
| radii.named.side | reference | absolute, separation units | 1 | 4.84e-08 of 5e-08 | 0.97 | Leahy & Leahy (2015), Table 1 |
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
Medians of 5 draws in milliseconds on macOS-26.6.1-arm64-arm-64bit. Cold is a fresh geometry with empty caches; new level is a second level at the same mass ratio; cached repeats the new-level call.

| Accuracy setting | qi | Regime | Operation | Cold ms | New level ms | Cached ms |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| fitted | 0.001 | contact | fill | 237.51 | 30.8662 | 0.000500004 |
| fitted | 0.001 | contact | sample | 215.954 | 28.5082 | 0.000500004 |
| fitted | 0.001 | contact | section | 190.082 | 2.87579 | 0.63625 |
| fitted | 0.001 | contact | selection | 335.566 | 93.0718 | 0.00499998 |
| fitted | 0.001 | contact | silhouette_60 | 203.845 | 15.7076 | 8.12767 |
| fitted | 0.001 | contact | silhouette_85 | 205.195 | 16.1518 | 8.55667 |
| fitted | 0.001 | contact | volume | 217.774 | 30.7763 | 0.000583008 |
| fitted | 0.001 | detached | fill | 33.5566 | 13.3557 | 0.000458036 |
| fitted | 0.001 | detached | lobe_radius | 0.00229204 | — | 0.000415952 |
| fitted | 0.001 | detached | lobe_slope | 0.00191701 | — | 0.00029197 |
| fitted | 0.001 | detached | sample | 13.2565 | 12.8924 | 0.000500004 |
| fitted | 0.001 | detached | section | 0.4225 | 0.421667 | 0.413833 |
| fitted | 0.001 | detached | selection | 0.200292 | 0.0102081 | 0.00208302 |
| fitted | 0.001 | detached | silhouette_60 | 7.06938 | 7.01708 | 6.99846 |
| fitted | 0.001 | detached | silhouette_85 | 8.96983 | 8.96137 | 8.95163 |
| fitted | 0.001 | detached | volume | 13.4335 | 13.2999 | 0.00054203 |
| fitted | 1 | contact | fill | 267.409 | 25.2156 | 0.000500004 |
| fitted | 1 | contact | sample | 247.849 | 22.8572 | 0.000541972 |
| fitted | 1 | contact | section | 227.753 | 2.73717 | 0.574 |
| fitted | 1 | contact | selection | 347.513 | 75.0658 | 0.004542 |
| fitted | 1 | contact | silhouette_60 | 235.005 | 9.72908 | 8.58846 |
| fitted | 1 | contact | silhouette_85 | 236.996 | 10.9905 | 9.82842 |
| fitted | 1 | contact | volume | 250.431 | 25.121 | 0.000666012 |
| fitted | 1 | detached | fill | 28.4758 | 11.5233 | 0.000458036 |
| fitted | 1 | detached | lobe_radius | 0.00279199 | — | 0.000333996 |
| fitted | 1 | detached | lobe_slope | 0.00191596 | — | 0.000250002 |
| fitted | 1 | detached | sample | 11.4021 | 11.1136 | 0.000500004 |
| fitted | 1 | detached | section | 0.408291 | 0.399541 | 0.394042 |
| fitted | 1 | detached | selection | 0.088958 | 0.00891602 | 0.00187499 |
| fitted | 1 | detached | silhouette_60 | 6.46842 | 6.42821 | 6.42237 |
| fitted | 1 | detached | silhouette_85 | 8.70479 | 8.71554 | 8.69692 |
| fitted | 1 | detached | volume | 11.5155 | 11.4836 | 0.000499946 |
| fitted | 1000 | contact | fill | 234.515 | 30.0666 | 0.000500004 |
| fitted | 1000 | contact | sample | 214.242 | 27.6029 | 0.000500004 |
| fitted | 1000 | contact | section | 190.122 | 2.88733 | 0.662833 |
| fitted | 1000 | contact | selection | 295.352 | 59.7875 | 0.00462501 |
| fitted | 1000 | contact | silhouette_60 | 202.776 | 14.8314 | 11.5135 |
| fitted | 1000 | contact | silhouette_85 | 207.204 | 19.1244 | 15.7689 |
| fitted | 1000 | contact | volume | 216.925 | 30.0053 | 0.000583008 |
| fitted | 1000 | detached | fill | 31.1598 | 13.4143 | 0.000459026 |
| fitted | 1000 | detached | lobe_radius | 0.00208302 | — | 0.000333996 |
| fitted | 1000 | detached | lobe_slope | 0.00183296 | — | 0.000250002 |
| fitted | 1000 | detached | sample | 13.3868 | 13.0022 | 0.000458036 |
| fitted | 1000 | detached | section | 0.448292 | 0.440291 | 0.432042 |
| fitted | 1000 | detached | selection | 0.19475 | 0.00945904 | 0.00204198 |
| fitted | 1000 | detached | silhouette_60 | 8.81096 | 8.70846 | 8.69613 |
| fitted | 1000 | detached | silhouette_85 | 12.063 | 12.0117 | 12.0268 |
| fitted | 1000 | detached | volume | 13.412 | 13.396 | 0.000541972 |
| fitted | 1e+06 | contact | fill | 179.879 | 34.2613 | 0.000500004 |
| fitted | 1e+06 | contact | sample | 160.831 | 31.7787 | 0.000500004 |
| fitted | 1e+06 | contact | section | 131.951 | 2.97675 | 0.750041 |
| fitted | 1e+06 | contact | selection | 214.781 | 34.137 | 0.00441598 |
| fitted | 1e+06 | contact | silhouette_60 | 148.382 | 18.4706 | 15.1235 |
| fitted | 1e+06 | contact | silhouette_85 | 150.917 | 20.717 | 17.3558 |
| fitted | 1e+06 | contact | volume | 163.143 | 34.2086 | 0.000583008 |
| fitted | 1e+06 | detached | fill | 30.4703 | 13.37 | 0.000457978 |
| fitted | 1e+06 | detached | lobe_radius | 0.00204198 | — | 0.000333006 |
| fitted | 1e+06 | detached | lobe_slope | 0.00145898 | — | 0.000250002 |
| fitted | 1e+06 | detached | sample | 13.3059 | 13.0181 | 0.000500004 |
| fitted | 1e+06 | detached | section | 0.440958 | 0.437417 | 0.429292 |
| fitted | 1e+06 | detached | selection | 0.198 | 0.00937498 | 0.00191695 |
| fitted | 1e+06 | detached | silhouette_60 | 8.625 | 8.52213 | 8.50758 |
| fitted | 1e+06 | detached | silhouette_85 | 11.882 | 11.8304 | 11.8342 |
| fitted | 1e+06 | detached | volume | 13.5165 | 13.4129 | 0.000540982 |
| fitted | 1e+08 | contact | fill | 144.392 | 37.2367 | 0.000500004 |
| fitted | 1e+08 | contact | sample | 124.761 | 34.7311 | 0.00054203 |
| fitted | 1e+08 | contact | section | 92.7205 | 3.04421 | 0.808208 |
| fitted | 1e+08 | contact | selection | 144.514 | 0.00804197 | 0.00237499 |
| fitted | 1e+08 | contact | silhouette_60 | 110.42 | 19.554 | 16.0499 |
| fitted | 1e+08 | contact | silhouette_85 | 112.425 | 21.3975 | 17.823 |
| fitted | 1e+08 | contact | volume | 127.168 | 37.0815 | 0.000624976 |
| fitted | 1e+08 | detached | fill | 30.8972 | 13.3707 | 0.000500004 |
| fitted | 1e+08 | detached | lobe_radius | 0.00208302 | — | 0.000333006 |
| fitted | 1e+08 | detached | lobe_slope | 0.00162504 | — | 0.000250002 |
| fitted | 1e+08 | detached | sample | 13.3262 | 12.9505 | 0.000500004 |
| fitted | 1e+08 | detached | section | 0.444791 | 0.432583 | 0.430333 |
| fitted | 1e+08 | detached | selection | 0.195375 | 0.00937498 | 0.00199996 |
| fitted | 1e+08 | detached | silhouette_60 | 8.58762 | 8.50021 | 8.51729 |
| fitted | 1e+08 | detached | silhouette_85 | 11.9481 | 11.8751 | 11.8297 |
| fitted | 1e+08 | detached | volume | 13.4498 | 13.4002 | 0.000500004 |
| reference | 0.001 | contact | fill | 760.642 | 156.642 | 0.00054104 |
| reference | 0.001 | contact | sample | 612.724 | 145.512 | 0.00054203 |
| reference | 0.001 | contact | section | 799.363 | 331.453 | 295.325 |
| reference | 0.001 | contact | selection | 1409.12 | 627.109 | 0.00508298 |
| reference | 0.001 | contact | silhouette_60 | 609.802 | 139.101 | 118.595 |
| reference | 0.001 | contact | silhouette_85 | 615.565 | 146.476 | 128.105 |
| reference | 0.001 | contact | volume | 625.51 | 157.012 | 0.000708038 |
| reference | 0.001 | detached | fill | 244.59 | 107.828 | 0.000624976 |
| reference | 0.001 | detached | lobe_radius | 136.031 | — | 0.000375032 |
| reference | 0.001 | detached | lobe_slope | 145.283 | — | 0.00029197 |
| reference | 0.001 | detached | sample | 107.44 | 107.106 | 0.000667002 |
| reference | 0.001 | detached | section | 8.33254 | 8.27967 | 8.28021 |
| reference | 0.001 | detached | selection | 773.648 | 538.066 | 0.004042 |
| reference | 0.001 | detached | silhouette_60 | 81.9736 | 82.0912 | 82.0726 |
| reference | 0.001 | detached | silhouette_85 | 103.66 | 103.652 | 103.561 |
| reference | 0.001 | detached | volume | 107.785 | 107.902 | 0.000708969 |
| reference | 1 | contact | fill | 798.712 | 125.986 | 0.000583008 |
| reference | 1 | contact | sample | 681.325 | 114.831 | 0.000583008 |
| reference | 1 | contact | section | 579.142 | 12.4207 | 11.278 |
| reference | 1 | contact | selection | 1325.98 | 500.687 | 0.00491698 |
| reference | 1 | contact | silhouette_60 | 672.531 | 104.382 | 103.127 |
| reference | 1 | contact | silhouette_85 | 684.387 | 117.987 | 116.743 |
| reference | 1 | contact | volume | 703.087 | 126.039 | 0.000750006 |
| reference | 1 | detached | fill | 191.404 | 85.479 | 0.000709028 |
| reference | 1 | detached | lobe_radius | 105.547 | — | 0.000333996 |
| reference | 1 | detached | lobe_slope | 114.104 | — | 0.00029098 |
| reference | 1 | detached | sample | 85.1001 | 84.8128 | 0.000624976 |
| reference | 1 | detached | section | 8.63183 | 8.61542 | 8.61729 |
| reference | 1 | detached | selection | 618.059 | 343.113 | 0.00316696 |
| reference | 1 | detached | silhouette_60 | 84.7231 | 84.6311 | 84.6519 |
| reference | 1 | detached | silhouette_85 | 110.339 | 110.178 | 110.349 |
| reference | 1 | detached | volume | 85.7705 | 85.8487 | 0.000625034 |
| reference | 1000 | contact | fill | 738.531 | 152.38 | 0.000540982 |
| reference | 1000 | contact | sample | 607.239 | 140.439 | 0.000541972 |
| reference | 1000 | contact | section | 647.54 | 172.844 | 19.3653 |
| reference | 1000 | contact | selection | 1195.01 | 454.53 | 0.00504096 |
| reference | 1000 | contact | silhouette_60 | 752.991 | 284.571 | 266.309 |
| reference | 1000 | contact | silhouette_85 | 776.968 | 308.667 | 292.455 |
| reference | 1000 | contact | volume | 620.391 | 152.376 | 0.000667002 |
| reference | 1000 | detached | fill | 225.371 | 106.56 | 0.000500004 |
| reference | 1000 | detached | lobe_radius | 118.105 | — | 0.000333996 |
| reference | 1000 | detached | lobe_slope | 126.836 | — | 0.000250002 |
| reference | 1000 | detached | sample | 106.49 | 105.835 | 0.000624976 |
| reference | 1000 | detached | section | 9.03813 | 9.0465 | 9.00563 |
| reference | 1000 | detached | selection | 755.302 | 427.102 | 0.00599999 |
| reference | 1000 | detached | silhouette_60 | 109.909 | 105.861 | 102.777 |
| reference | 1000 | detached | silhouette_85 | 132.469 | 132.48 | 132.54 |
| reference | 1000 | detached | volume | 106.914 | 106.842 | 0.000624976 |
| reference | 1e+06 | contact | fill | 617.215 | 174.578 | 0.000583008 |
| reference | 1e+06 | contact | sample | 483.85 | 162.838 | 0.00054203 |
| reference | 1e+06 | contact | section | 390.56 | 66.7627 | 20.8892 |
| reference | 1e+06 | contact | selection | 966.188 | 347.59 | 0.00466697 |
| reference | 1e+06 | contact | silhouette_60 | 836.745 | 498.43 | 492.642 |
| reference | 1e+06 | contact | silhouette_85 | 1449.79 | 1130.73 | 1119.46 |
| reference | 1e+06 | contact | volume | 496.431 | 174.875 | 0.000666012 |
| reference | 1e+06 | detached | fill | 228.52 | 106.998 | 0.000540982 |
| reference | 1e+06 | detached | lobe_radius | 121.019 | — | 0.000333006 |
| reference | 1e+06 | detached | lobe_slope | 130.052 | — | 0.000250002 |
| reference | 1e+06 | detached | sample | 106.807 | 106.34 | 0.000708038 |
| reference | 1e+06 | detached | section | 9.17454 | 9.1245 | 9.14604 |
| reference | 1e+06 | detached | selection | 652.828 | 323.726 | 0.00595901 |
| reference | 1e+06 | detached | silhouette_60 | 102.003 | 101.982 | 102.017 |
| reference | 1e+06 | detached | silhouette_85 | 135.913 | 135.803 | 135.789 |
| reference | 1e+06 | detached | volume | 107.194 | 107.195 | 0.000791973 |
| reference | 1e+08 | contact | fill | 539.126 | 190.418 | 0.000583008 |
| reference | 1e+08 | contact | sample | 400.213 | 178.275 | 0.000583008 |
| reference | 1e+08 | contact | section | 260.41 | 32.0834 | 21.2863 |
| reference | 1e+08 | contact | selection | 919.624 | 378.5 | 0.00491703 |
| reference | 1e+08 | contact | silhouette_60 | 808.074 | 584.995 | 576.925 |
| reference | 1e+08 | contact | silhouette_85 | 1164.51 | 941.511 | 935.667 |
| reference | 1e+08 | contact | volume | 413.24 | 190.322 | 0.000750006 |
| reference | 1e+08 | detached | fill | 234.055 | 107.218 | 0.000624976 |
| reference | 1e+08 | detached | lobe_radius | 125.794 | — | 0.000333996 |
| reference | 1e+08 | detached | lobe_slope | 134.753 | — | 0.000291038 |
| reference | 1e+08 | detached | sample | 106.829 | 106.274 | 0.000625034 |
| reference | 1e+08 | detached | section | 9.19883 | 9.13617 | 9.14088 |
| reference | 1e+08 | detached | selection | 659.624 | 322.252 | 0.00329199 |
| reference | 1e+08 | detached | silhouette_60 | 102.079 | 102.019 | 102.011 |
| reference | 1e+08 | detached | silhouette_85 | 135.89 | 136.245 | 141.686 |
| reference | 1e+08 | detached | volume | 107.401 | 107.675 | 0.000624976 |
<!-- /generated:timings -->

</details>
