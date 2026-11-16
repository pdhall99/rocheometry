# Fill map

`src/rocheometry/_data/surface-potential-map.json` gives the potential of a detached surface from its component mass ratio `qi` and its volume-radius fill `fill = R_vol / R_RL,vol`.
It lets `Fitted()` select a surface by fill without inverting a numerical volume.

## Model

The map stores the residual `t` of an endpoint-constrained form,

```
fill * t = fill * (1 + (t_inf - 1)(1 - fill^3) + (1 - fill) fill^3 R(z, w))
```

where `t_inf` comes from the lobe volume radius, so the form is exact at `fill = 0` and `fill = 1` and the tensor `R` only carries the interior.

| Choice | Value | Reason |
| --- | --- | --- |
| `qi` coordinate | `z`: compact fourth root `qi^(1/4) / (1 + qi^(1/4))`, mapped to [-1, 1] | spreads both extremes of `1e-8 <= qi <= 1e8` evenly |
| fill coordinate | `w = 2 sqrt(1 - fill) - 1` | the potential has a square-root behaviour at the critical surface |
| degrees | 24 in `z`, 14 in `w` (25 × 15 coefficients) | 13 × 9 misses the 1e-6 held-out target on the wide `qi` domain |
| domain | `1e-8 <= qi <= 1e8`, `0 < fill <= 1` | the contract domain |
| training | 49 Lobatto nodes in `qi`, 25 intervals in `sqrt(1 - fill)` | `Reference()` inverts its volume at each node; no packaged fit but the lobe radius is read |

## Holdout

450 Reference measurements at random `(qi, fill)`, seed 11, independent of the training nodes (`holdout.json`).
The packaged map has a largest relative fill error of 1.81e-7 on it, against a bound of 1e-6.

## Reproducing it

```shell
hatch run fits:build fill_map --output src/rocheometry/_data/surface-potential-map.json
hatch run fits:check fill_map
```

Build the lobe radius first: the map's endpoint form reads it.
In the environment recorded in `provenance.json` a rebuild is bit-identical, and `fits:check` requires it.
The builder inverts `Reference()` at each training node, so the training data never depend on the packaged map.
