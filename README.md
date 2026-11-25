# Rocheometry

<img src="https://pdhall99.github.io/rocheometry/assets/favicon.svg" alt="Rocheometry logo: the two Roche lobes of a binary system" width="96" align="right">

[![CI](https://img.shields.io/github/actions/workflow/status/pdhall99/rocheometry/ci.yml?branch=main)](https://github.com/pdhall99/rocheometry/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rocheometry)](https://pypi.org/project/rocheometry/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://pdhall99.github.io/rocheometry/)
[![Python](https://img.shields.io/pypi/pyversions/rocheometry)](https://pypi.org/project/rocheometry/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24325522.svg)](https://doi.org/10.5281/zenodo.24325522)
[![Runtime dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)](https://github.com/pdhall99/rocheometry/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/pdhall99/rocheometry)](https://github.com/pdhall99/rocheometry/blob/main/LICENSE)

Compute the equipotential geometry of binary systems in the classical Roche model.

Rocheometry computes the geometry of equipotential surfaces around the components of a binary system in the classical Roche model: their radii, volumes, areas, curvatures and gravities, and their outlines as cubic Bézier paths for visualisation.
The Roche model is widely used in astrophysics to describe close binary systems, from interacting binary stars and contact binaries to planets close to their host stars.
A surface can be at any level from detached surfaces through the critical Roche lobes to contact envelopes approaching the L2 opening.
A surface can be selected by its potential, its volume radius, how far it fills its lobe, or its fill-out factor.

## Table of contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Accuracy and speed](#accuracy-and-speed)
- [Citation](#citation)
- [Compatibility and versioning](#compatibility-and-versioning)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Background

In the classical Roche model of a binary system, each component's gravity is that of a point mass at its centre, the orbit is circular, and each component rotates synchronously with it ([Kopal 1959](https://scixplorer.org/abs/1959cbs..book.....K), [full text](https://archive.org/details/closebinarysyste00kopa/page/125/mode/1up)).
In the co-rotating frame, gravity and the centrifugal force derive from one potential, the **Roche potential**.
A component in hydrostatic equilibrium has its surface on an **equipotential**: a surface on which the potential takes one value, its **level**.
In units of the separation, the geometry depends only on the mass ratio `q`, so the surfaces form a family set by `q` and the level.

As the level falls, each component's surface grows until the two meet at the inner Lagrange point, L1; these critical surfaces are the **Roche lobes**.
Below L1, the components share a contact envelope, which opens at the outer Lagrange point, L2.
The contact envelope can be divided between the two components at the watershed, the surface through L1 made of field lines of effective gravity ([Fabry, Marchant & Sana 2022](https://doi.org/10.1051/0004-6361/202243094)).

Many applications need only one member of the family, the critical Roche lobe, and a fit to its radius such as that of [Eggleton (1983)](https://doi.org/10.1086/160960) is enough.
Anything more, such as the shape of a component that underfills its lobe, its surface area, or the envelope of a contact binary, usually requires a solver.
Selecting a surface by a physical quantity, such as its radius, rather than by its potential needs a root finder on top.
Because of this, Roche-lobe figures are often sketched or approximated rather than computed.

Rocheometry is a dependency-free Python library that computes these surfaces directly, selects them by whatever you know about them, and provides tools for visualising them.

### Related tools

- [roche-radius-calculator](https://github.com/denisleahy/roche-radius-calculator) computes radii, areas and volumes of the Roche lobe and of surfaces inside it, with a Java graphical interface and a Fortran command-line version; it also handles non-synchronous rotation and, with some restrictions, eccentric orbits ([Leahy & Leahy 2015](https://doi.org/10.1186/s40668-015-0008-8))
- [`roche`](https://github.com/trm-astro/roche-rs) is a Python wrapper around a Rust translation of [cpp-roche](https://github.com/trmrsh/cpp-roche), covering Lagrange points, potentials, Roche-lobe sections, eclipses and gas streams
- [`phoebe`](https://github.com/phoebe-project/phoebe2) models eclipsing binaries, including contact binaries, on Roche meshes, and computes light curves and radial velocities from them ([Prša et al. 2016](https://doi.org/10.3847/1538-4365/227/2/29))
- [`PyAstronomy`](https://github.com/sczesla/PyAstronomy) is a collection of astronomy routines that includes the dimensionless Roche potential and its derivatives, Lagrange points, Eggleton's approximation to the Roche-lobe radius, and Roche-lobe volumes and effective radii from Monte Carlo integration ([Czesla et al. 2019](https://ascl.net/1906.010))

## Installation

`rocheometry` requires Python 3.11 or later (see [Compatibility and versioning](#compatibility-and-versioning)).
It has no runtime dependencies by default.

`rocheometry` is available from [PyPI](https://pypi.org/project/rocheometry/).
Install it with your preferred Python package manager.
For example, using `pip`:

```shell
python -m pip install rocheometry
```

Install with the optional `plot` extra to use the helpers in `rocheometry.matplotlib`:

```shell
python -m pip install "rocheometry[plot]"
```

## Usage

### Quickstart

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

### Conventions

- **Mass ratio:** `q = M1 / M2`, with no assumption about which component is more massive: `q` can be greater or less than 1.
  Each component also has its own ratio `qi`, its mass over its companion's.
- **Lengths:** in units of the separation, or in the binary's length units when it is given a `separation`.
- **Frame:** co-rotating and right-handed, with component 1 at the origin and component 2 at `(1, 0, 0)`.
  The `x` axis is the binary centreline, traditionally called the line of centres; `y` lies in the orbital plane and `z` along the orbital angular momentum.
- **Potential:** the positive Jacobi potential `U`, normalized by the total mass, which rises towards the components.
  Potential values carry their convention, and a Kopal value also names its component; see [potentials and critical points](https://pdhall99.github.io/rocheometry/guide/potentials/).

See [The Roche model](https://pdhall99.github.io/rocheometry/guide/model/) for further definitions.

### Quantities

Every quantity is a method on one of these objects:

| Object | Create with | Gives |
| :-- | :-- | :-- |
| `BinaryGeometry` | `BinaryGeometry(q, separation=...)`, `BinaryGeometry.from_masses(...)` | Lagrange points, components, potential conversion |
| component | `binary.component(1)`, `binary.component(2)` | position, Roche lobe, surfaces |
| surface | `component.surface(...)`, `component.lobe()` | potential, radii, volume, area, curvature, gravity, quadrature, sections, silhouettes |
| `ComponentGeometry` | `ComponentGeometry(qi)` | one component on its own, in units of the separation |

`q = M1/M2` is the binary's mass ratio; `qi` is a component's own ratio, its mass over its companion's.

Each quantity is available on a single component (`ComponentGeometry`) and on either component of a binary (`BinaryGeometry.component(1)` or `(2)`), except where its notes say otherwise.

| Quantity | Call | Notes |
| --- | --- | --- |
| Surface by fill, radius, potential or fill-out factor | `surface(...)` | Pick exactly one; a binary's components take Jacobi or Kopal potentials, `ComponentGeometry` takes Jacobi only |
| Critical Roche lobe | `lobe()` | |
| Lobe volume radius | `lobe_volume_radius()` | Fast fitted model; `lobe().volume_radius()` measures the lobe itself |
| Level regime | `surface.regime`, `level_regime(potential)` | Detached, critical, contact or open |
| Volume and volume radius | `volume()`, `volume_radius()` | |
| Volume-radius fill | `volume_radius_fill()` | (V / V_lobe)^(1/3); above 1 in contact |
| Fill-out factor | `fill_out_factor()` | 0 at L1, 1 at L2 |
| Area | `area()` | A contact component's share, excluding the cut |
| Integrated mean curvature | `integrated_mean_curvature()` | Integral of H dA |
| Mean effective gravity | `mean_gravity()` | Area-weighted mean of the field gradient |
| Point, back, side and pole radii | `named_radius(name)` | |
| Radius in any direction | `radius(direction)` | |
| Planar section | `section(plane)` | Closed cubic Bézier paths |
| Silhouette at quadrature | `silhouette(plane)`, `silhouette_envelope(plane)` | Any inclination, at the orbital phase where the centreline lies in the sky plane; contact levels need `silhouette_envelope()` |
| Surface quadrature | `quadrature()` | Nodes and area weights for your own integrals |
| Inside test and potential at a point | `contains(point)`, `potential_at(point)` | `potential_at()` is on `BinaryGeometry` and `ComponentGeometry` |
| Lagrange points with Hessians | `l1()` … `l5()` on a binary | Components have `l1()`, `outer()`, `triangular()` |

Selection by volume radius divides by the fitted lobe radius, so a radius measured from `lobe()` can select a detached surface just inside the lobe.
Use `lobe()` or `volume_radius_fill=1` when you need the critical surface itself.

Every path can be exported to SVG or to Matplotlib.

## Documentation

See the [project documentation](https://pdhall99.github.io/rocheometry).

## Accuracy and speed

### Accuracy settings

The accuracy and speed of `BinaryGeometry` and `ComponentGeometry` can be configured by setting `accuracy`.

By default, both classes have `accuracy=rocheometry.Fitted()`, with which they use fitted formulae to compute some quantities.
This makes computation faster while still maintaining sufficient accuracy for many applications.

The alternative is to set `accuracy=rocheometry.Reference()`, which does not use any fitted formulae.
This is slower but more precise than the default.

The [accuracy page](https://pdhall99.github.io/rocheometry/guide/accuracy/) gives the measured bounds and timings for both settings.

### Measured accuracy

<!-- generated:readme-contract -->
Rocheometry's accuracy is measured for component mass ratios from 10<sup>−8</sup> to 10<sup>8</sup>, at every level above L2.

With the default `Fitted()`:

- measured quantities, such as volumes, areas and radii, have relative errors of at most **10<sup>−6</sup>**;
- a surface selected by its fill has the requested fill to within **2 × 10<sup>−6</sup>**;
- sections and silhouettes lie within **2.5 × 10<sup>−4</sup>** of the true outline, in units of the surface's volume radius.

With `Reference()`, quantities and outlines are within **10<sup>−10</sup>**.
The exceptions are the integrated mean curvature and mean-curvature radius, within **2 × 10<sup>−8</sup>** at the default orders and **10<sup>−10</sup>** at `Reference(polar_order=512, azimuth_order=192)`.

The [accuracy page](https://pdhall99.github.io/rocheometry/guide/accuracy/) gives every bound and the measurements behind it.
<!-- /generated:readme-contract -->

### Measured speed

<!-- generated:readme-speed -->
Typical times with `Fitted()` for a detached surface, across the measured mass ratios:

- selecting a surface: **0.0089–0.01 ms**;
- measuring its volume: **11–13 ms**;
- a section: **0.4–0.43 ms**;
- a silhouette at 60° inclination: **6.3–8.3 ms**;
- reading a volume already measured: **0.0005–0.00083 ms**.

These are medians on one machine.
[The timing table](https://pdhall99.github.io/rocheometry/guide/accuracy/#speed) also gives the costs with `Reference()`, for contact surfaces, and of building a geometry for the first time.
<!-- /generated:readme-speed -->

### Verification

Rocheometry's solution of the classical Roche model is verified against independent calculations.

Before each release whose code has changed, a release run compares every quantity across the whole verified domain with high-precision references: 40- and 100-digit root solves, volumes by Cartesian slicing, and integrals by a separate quadrature rule.
On every push, the tests check that the code implements the model's definitions and satisfies exact identities, such as the Gauss–Bonnet theorem.
Its radii also agree with the published values of [Leahy & Leahy (2015)](https://doi.org/10.1186/s40668-015-0008-8) to their stated precision.

## Citation

If you use this project in published work, please cite it as detailed on [the citation page](https://pdhall99.github.io/rocheometry/project/citation/).

## Compatibility and versioning

### Package versioning

This project follows [Semantic Versioning](https://semver.org/).
Releases have version numbers of the form `MAJOR.MINOR.PATCH`:

- **MAJOR** releases may contain backwards-incompatible changes to the public API
- **MINOR** releases may add functionality and deprecate public APIs while remaining backwards compatible
- **PATCH** releases contain backwards-compatible fixes

APIs explicitly documented as experimental are not covered by the same backwards-compatibility guarantees.

For releases before `1.0.0`, the public API should be considered under development and may change between minor releases, as permitted by Semantic Versioning.

### Python-version compatibility

This project supports Python feature releases from their official final release until their [official end-of-life](https://devguide.python.org/versions/).

Support for a new Python feature release is generally introduced in the first minor release of this project following the upstream Python release.
Python feature releases may be dropped once they reach the end of their upstream support cycle.
The currently supported Python versions are declared in the package metadata.

Dropping an end-of-life Python version is considered a change to the supported runtime environment rather than a backwards-incompatible change to this project's public API, and therefore does not by itself require a new major release.

## Acknowledgements

This project is developed with assistance from AI coding tools.
All code included in the project is reviewed and tested by [@pdhall99](https://github.com/pdhall99), who takes responsibility for its quality and maintenance.

## Contributing

See the [contributor guide](https://github.com/pdhall99/rocheometry/blob/main/docs/CONTRIBUTING.md).

## License

[MIT © PD Hall](https://github.com/pdhall99/rocheometry/blob/main/LICENSE)
