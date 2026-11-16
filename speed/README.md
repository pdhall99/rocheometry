# Speed

How fast Rocheometry is.
`grid.py` is the standard timing grid, whose medians the docs publish; `profile.py` runs cProfile on one call from it.
The published timings are in `results/speed.json`, and `report.py` generates the speed sections of the docs from them.
The verbs mean the same as in `accuracy/`.

```shell
hatch run speed:sample                 # build/speed/<commit>.json
hatch run speed:sample --quick         # Fitted only, qi = 1, one draw
hatch run speed:compare published A.json  # or two timing files
hatch run speed:profile silhouette_85 --qi 1e6 --setting reference
hatch run speed:run                    # the release timing: publish and regenerate the docs
hatch run speed:check                  # published timings and generated docs vs this checkout
```
