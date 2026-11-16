# Fits

How the packaged fits are built.
Each artefact in `src/rocheometry/_data/` has a folder here holding its configuration, its builder, its frozen holdout and the record of how the packaged file was built; its README states the model, its choices and the holdout result.
The verbs mean the same as in `accuracy/` and `speed/`.

| Path | Contents |
| --- | --- |
| `lobe_volume_radius/` | the critical lobe's volume radius from `qi` |
| `fill_map/` | the potential of a detached surface from `qi` and its fill; depends on the lobe radius |
| `fit.py` | what a fit is: `Fit`, which checks a packaged file's hash, holdout and rebuild |
| `common.py` | shared build code, hashed as an input of every build |

In each fit's folder, `config.yaml`, `build.py` and `common.py` are the build inputs: `provenance.json` records their hashes, so editing them needs a rebuild.
`fit.py` and `holdout.json` only check the result.

The Rocheometry rename changed only package names, paths, schema identifiers and build-input hashes.
The fit coefficients and holdout values were retained exactly, and the provenance hashes were updated for that metadata migration.
The release run's manifest in `accuracy/results/` predates the rename and was kept unchanged.
Renaming the selection's fill-to-depth computer from `potential` to `depth`, which `fill_map/build.py` calls, was migrated the same way.

```shell
hatch run fits:build lobe_volume_radius   # build/fits/
hatch run fits:build fill_map
hatch run fits:check                      # hashes, holdouts and rebuilds
hatch run fits:check --quick              # hashes only: the check CI runs
```
