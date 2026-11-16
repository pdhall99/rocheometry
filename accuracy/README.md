# Accuracy

How accurate Rocheometry is, and how we know.
Each check here is a study: it measures the library over a domain and records the result in each release run, and the docs publish what it finds.
In standard terms the studies are solution verification (how large is the numerical error?) and comparisons with published results (do our definitions match the model as others state it?).

A check whose result the docs publish belongs here.
A check that only gates a change is a test, and belongs in the top-level `tests/`; the same comparison often exists as both.
Studies measure against the reference computations in `crosschecks/`.

Every study, contract or literature, is a `Study` class (`accuracy/harness/study.py`) in its own file, registered in `accuracy/harness/run.py`, and reports the same rows.

| Directory | Contents |
| --- | --- |
| `contract/` | Accuracy studies over the contract domain, one study per file; frozen inputs in `grids/`; contract bounds in `bounds.py` |
| `literature/` | Comparisons with published results for the classical Roche model |
| `harness/` | What a study is (`study.py`), running, recording (`record.py`) and summarizing studies |
| `tests/` | Pytest for this folder's own code: the registry, the grids and the bookkeeping |
| `results/` | Summaries of the latest release run; git keeps the earlier ones |
| `catalogue.toml` | Every public numerical quantity and every check of it |

| Command | Does |
| --- | --- |
| `hatch run accuracy:run` | the release run: every study, then the generated docs |
| `hatch run accuracy:smoke` | one small job of every study |
| `hatch run accuracy:sample --every K` | every K-th job of every study |
| `hatch run accuracy:study NAME ...` | chosen studies over their full domain |
| `hatch run accuracy:check` | committed results, generated docs and catalogue against this checkout |
| `hatch run accuracy:report` | regenerate the generated docs |

Every row in the results is named for the catalogue quantity it measures.
