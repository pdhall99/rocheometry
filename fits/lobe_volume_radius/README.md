# Lobe volume radius

`src/rocheometry/_data/volume-radius-model.json` gives the volume-equivalent radius of the critical Roche lobe, `R_RL,vol / a`, for every positive mass ratio `qi`.

## Model

```
s = qi^(1/6) / (1 + qi^(1/6))
F = Horner(4s - 1, low)   if qi <= 1
    Horner(4s - 3, high)  otherwise
R = s^2 F
```

| Choice | Value | Reason |
| --- | --- | --- |
| coordinate | `s = qi^(1/6) / (1 + qi^(1/6))` | maps every `qi > 0` to (0, 1); `R / s^2` is finite at both ends |
| endpoint constraints | `F(0) = H0`, `F'(0) = 2 H0`, `F(1) = H1`, `F'(1) = -2 H1` | the Hill limit and the `qi -> inf` limit hold exactly |
| pieces | split at `qi = 1`, degrees 17 and 19, C2 at the join | best held-out accuracy and scalar runtime of the compact whole-domain forms compared; no roundoff sensitivity |
| shape | `dR/ds >= 0` on a 1200-point grid | the radius is monotone in `qi` |
| training | 257 Chebyshev–Lobatto nodes in `s`; `Reference()` inside `1e-12 <= qi <= 1e12` | outside that range the binary64 mass pair loses accuracy and the limits carry the fit |

38 numbers in all.
The alternatives compared were a one-piece degree-36 Horner form, rational baselines with corrections, and cube-root forms on a finite domain; the two pieces gave the best held-out accuracy and scalar runtime.

## Holdout

253 mass ratios over `1e-12 <= qi <= 1e12` at interval midpoints of three spacings, with `Reference()` radii (`holdout.json`).
The packaged model has a largest relative error of 5.02e-9 on it, against a bound of 1e-8.

## Reproducing it

```shell
hatch run fits:build lobe_volume_radius --output src/rocheometry/_data/volume-radius-model.json
hatch run fits:check lobe_volume_radius
```

The build reads only `config.yaml` and `Reference()`.
`provenance.json` records the hashes of its inputs and the environment that made the packaged file; in that environment (platform, Python, NumPy, SciPy) a rebuild is bit-identical, and `fits:check` requires it.
Elsewhere libm rounding can change the last bits, and the rebuild is compared for information only.
