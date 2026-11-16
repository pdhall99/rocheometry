"""Generate the accuracy map and exhaustive failure list from study summaries."""

import gzip
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from accuracy.harness.common import COMMIT, OUTLINES, read_rows, script, summarize
from accuracy.harness.contracted import case_key, contracted_rows, test_cases

logger = logging.getLogger(__name__)


def generate(output: Path) -> None:
    """Write the accuracy map, failure list and contract cases into ``output``."""
    summaries = []
    failures = []
    notes = []
    studies = {}
    observations = []
    triage_path = output / "construction-triage.json"
    triage = (
        json.loads(triage_path.read_text())
        if triage_path.exists()
        else {"classifications": []}
    )
    classifications = {r["case_key"]: r for r in triage["classifications"]}
    unsupported_path = output / "unsupported.json"
    excluded = (
        {r["case_key"] for r in json.loads(unsupported_path.read_text())["cases"]}
        if unsupported_path.exists()
        else set()
    )
    for path in sorted(output.glob("*.summary.json")):
        name = path.name.removesuffix(".summary.json")
        if name.endswith("_smoke"):
            continue
        d = json.loads(path.read_text())
        measured_commit = d["commit"]
        rows = [r for r in read_rows(name, output) if case_key(r) not in excluded]
        # Summaries are derived views of the observations. Refresh schema and
        # aggregation without rerunning measurements or replacing wall times.
        d = summarize(name, rows, d["runtime_seconds"], d["metadata"])
        # Reaggregating a study is not a new measurement. Keep its provenance
        # when regenerating markers after a change in another study.
        d["commit"] = measured_commit
        for quantity in d["quantities"]:
            quantity["commit"] = measured_commit
        path.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
        studies[name] = dict(
            runtime_seconds=d["runtime_seconds"],
            metadata=d["metadata"],
            commit=d["commit"],
        )
        rows = contracted_rows(name, rows)
        d = summarize(name, rows, d["runtime_seconds"], d["metadata"])
        d["commit"] = measured_commit
        for r in d["quantities"]:
            r["commit"] = measured_commit
            if r["quantity"].startswith("reference_convergence."):
                continue
            r["study"] = name
            summaries.append(r)
        observations.extend({**r, "study": name} for r in rows)
        failures.extend(
            {
                **r,
                "study": name,
                "script": script(name),
                "commit": d["commit"],
            }
            for r in rows
            if r["status"] in ("fail", "error")
        )
        notes.extend({**r, "study": name} for r in rows if r["status"] == "note")
    # The same named quantity can be observed in a full default-truth study
    # and an independent reference-only study. Merge their measured domains.
    pairs = defaultdict(dict)
    for row in observations:
        if row["quantity"] in OUTLINES:
            key = (
                row["quantity"],
                row["metric"],
                json.dumps(row["case"], sort_keys=True),
            )
            pairs[key][row["set"]] = row
    unmeasured = []
    for group in pairs.values():
        if len(group) != 1:
            continue
        blocking = next(iter(group.values()))
        if blocking["value"] is not None:
            # A one-sided measured distance uses a reference curve; absence of
            # a separate exact-field audit is not a construction failure.
            continue
        missing = "fitted" if blocking["set"] == "reference" else "reference"
        unmeasured.append(
            {
                "quantity": blocking["quantity"],
                "set": missing,
                "metric": blocking["metric"],
                "case": blocking["case"],
                "status": "unmeasured",
                "classification": "defect",
                "diagnosis": blocking.get("diagnosis")
                or classifications.get(case_key(blocking), {}).get("diagnosis"),
                "reason": f"{blocking['set']} construction failed; no comparison curve",
                "blocking_case_key": case_key(blocking),
            }
        )
    missing_counts = Counter((r["quantity"], r["set"]) for r in unmeasured)
    from accuracy.harness.run import LITERATURE

    merged = {}
    for row in summaries:
        # A comparison with a publication never merges into a contract row.
        key = (row["quantity"], row["set"], row["metric"], row["study"] in LITERATURE)
        if key not in merged:
            merged[key] = row
            continue
        old = merged[key]
        old["failures"] += row["failures"]
        old["domain"]["cases"] += row["domain"]["cases"]
        old["domain"]["measured"] += row["domain"]["measured"]
        old["domain"]["contracted_cases"] += row["domain"]["contracted_cases"]
        for end, operation in (("qi_min", min), ("qi_max", max)):
            old["domain"][end] = operation(old["domain"][end], row["domain"][end])
        old["domain"]["grid"] = "See the combined study metadata in accuracy.json."
        old["notes"] = sorted(set(old["notes"] + row["notes"]))
        a, b = old.get("closest"), row.get("closest")
        if b and (not a or b["value"] / b["bound"] > a["value"] / a["bound"]):
            old["closest"] = b
        a, b = old["known_note_worst"], row["known_note_worst"]
        if b and (not a or b["value"] > a["value"]):
            old["known_note_worst"] = b
        if row["measured_worst"] is not None and (
            old["measured_worst"] is None
            or row["measured_worst"] > old["measured_worst"]
        ):
            old.update(
                {
                    k: row[k]
                    for k in (
                        "measured_worst",
                        "worst_case",
                        "contracted_bound",
                        "script",
                        "study",
                    )
                }
            )
    summaries = sorted(merged.values(), key=lambda r: (r["quantity"], r["set"]))
    for row in summaries:
        row["domain"]["unmeasured_cases"] = missing_counts[row["quantity"], row["set"]]
    (output / "accuracy.json").write_text(
        json.dumps(
            dict(
                commit=COMMIT,
                studies=studies,
                quantities=summaries,
                unmeasured_comparisons=len(unmeasured),
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    failures = sorted({case_key(r): r for r in failures}.values(), key=case_key)
    for index, row in enumerate(failures, 1):
        classified = classifications.get(case_key(row))
        if classified:
            row.update(
                classification=classified["classification"],
                diagnosis=classified["diagnosis"],
                geometry=classified["geometry"],
                public_docstring=classified["public_docstring"],
            )
        row["id"] = f"FL-{index:05d}"
    failure_ids = {case_key(r): r["id"] for r in failures}
    for row in unmeasured:
        row["blocking_failure_id"] = failure_ids.get(row.pop("blocking_case_key"))
    (output / "unmeasured.json").write_text(json.dumps(unmeasured, indent=2) + "\n")
    (output / "contract-cases.json.gz").write_bytes(
        gzip.compress(
            (
                json.dumps(
                    test_cases(observations, failures),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode(),
            mtime=0,
        )
    )
    (output / "failures.json").write_text(
        json.dumps(
            dict(commit=COMMIT, count=len(failures), failures=failures),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (output / "known-limitations.json").write_text(
        json.dumps(dict(commit=COMMIT, notes=notes), indent=2, allow_nan=False) + "\n"
    )
    logger.info(
        "accuracy map: %d quantities, %d failures, %d notes",
        len(summaries),
        len(failures),
        len(notes),
    )
