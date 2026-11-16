"""The studies harness must keep working.

This checks the machinery, not the accuracy numbers. The fast tests confirm
that every study is registered where it lives, that its frozen inputs load and match
the committed manifest, and that the harness's bookkeeping behaves; the slow
tests run each study on one small job. CI also runs ``hatch run accuracy:smoke``.
"""

import inspect
import json
from pathlib import Path

import pytest

from accuracy.harness import cases, coverage, run
from accuracy.harness.contracted import contracted_rows

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "accuracy/results"


def test_catalogue_resolves_and_covers_every_measured_quantity() -> None:
    from accuracy.harness import coverage

    entries, problems = coverage.load()
    assert not problems, "\n".join(problems)
    assert len({e.key for e in entries}) == len(entries)


def test_every_study_is_registered_under_its_file_name() -> None:
    # The base classes enforce the methods; this checks what they cannot: that
    # each study lives where results and coverage look for it, measures
    # catalogue keys, and has jobs.
    keys = {e.key for e in coverage.load()[0]}
    for name, study in run.REGISTRY.items():
        path = Path(inspect.getfile(type(study)))
        assert path == ROOT / f"accuracy/{study.kind}/{name}.py", name
        assert study.domain, name
        assert set(study.quantities) <= keys, name
        if name in run.STUDIES:
            assert run.study(name).jobs(), name


def test_frozen_grids_decode_to_finite_floats() -> None:
    for name in cases.manifest():
        data = cases.load(name)
        assert data["cases"] if "cases" in data else data["decades"]
    grid = cases.load("permanent.json")
    assert len(grid["decades"]) == 17 and len(grid["random_q"]) == 200
    assert all(isinstance(q, float) and q > 0 for q in grid["sweep"])


def test_full_job_lists_match_the_recorded_release_counts() -> None:
    if not (RESULTS / "summary.json").exists():
        pytest.skip("no release run recorded yet")
    recorded = json.loads((RESULTS / "summary.json").read_text())["studies"]
    observed = {r["name"]: r.get("requested_jobs") for r in recorded}
    renamed = dict(
        critical="critical_points",
        fill_map="selection",
        freeze_saddles="saddle_quadrature",
        freeze_quadrature="quadrature",
        paths="outlines",
        depth_d4="outline_corners",
        depth_d4_holdouts="outline_holdouts",
        release_joint="joint_holdout",
    )
    counts = {renamed.get(k, k): v for k, v in observed.items() if v is not None}
    for name in run.STUDIES:
        if name not in run.UNRECORDED:
            assert len(run.jobs(name)) == counts[name], name
    assert set(run.UNRECORDED) <= set(run.STUDIES)


@pytest.mark.parametrize("name", run.STUDIES)
def test_each_study_has_one_small_smoke_job_unless_it_is_to_do(name: str) -> None:
    jobs = run.smoke_jobs(name)
    assert len(jobs) == (1 if run.jobs(name) else 0)


@pytest.mark.slow
@pytest.mark.parametrize("name", run.STUDIES)
def test_each_study_runs_without_a_harness_error(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROCHEOMETRY_STUDY_OUTPUT", str(tmp_path))
    if not run.smoke_jobs(name):
        pytest.skip(f"{name} is still to do")
    result = run._measure((name, run.smoke_jobs(name)[0]))
    assert not result.get("harness_error"), result.get("traceback")
    assert result["rows"]


def test_corner_diagnostic_cannot_generate_a_separate_strict_marker() -> None:
    row = {
        "quantity": "outlines.shape.corner_rounding",
        "set": "fitted",
        "value": 1.5e-4,
        "bound": 1.18e-4,
        "status": "fail",
    }
    actual = contracted_rows("example_holdout", [row])[0]
    assert actual["bound"] is None
    assert actual["status"] == "note"
    assert row["bound"] == 1.18e-4


def test_code_hash_ignores_comments_formatting_and_docstrings() -> None:
    from accuracy.harness.code import canonical

    original = (
        '"""Module."""\n\ndef f(x):\n    """Square."""\n    return x * 2  # double\n'
    )
    cosmetic = "def f( x ):\n    '''Twice x.'''\n\n\n    return x*2\n"
    assert canonical(original) == canonical(cosmetic)
    for changed in (
        "def f(x):\n    return x * 3\n",
        "def f(y):\n    return y * 2\n",
        "def f(x):\n    return x * 2.0\n",
        'def f(x):\n    "Square."\n    raise ValueError("other")\n',
    ):
        assert canonical(changed) != canonical(original)


def test_code_hash_ignores_attribute_docstrings_but_not_other_strings() -> None:
    from accuracy.harness.code import canonical

    assert canonical('X = 1\n"""One."""\n') == canonical('X = 1\n"""Unity."""\n')
    assert canonical('x = "a"\n') != canonical('x = "b"\n')
    assert canonical('f("a")\n') != canonical('f("b")\n')
