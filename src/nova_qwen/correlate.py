"""Cross-classifier view of one validation run (Decision Tree vs Random Forest).

record_id is the one-based row position in the frame each export was produced from (schema 1.1).
Two exports can only be correlated if they come from the same run and dataset, and if every
record_id they share carries identical evidence, the necessary condition for "same row". Rows that
only one classifier alerted on cannot be cross-checked; that limit is stated in the report.
Explanations are never merged: each classifier's answer stays attached to its own alert.
"""
from __future__ import annotations

from .alerts import AlertExport
from .errors import IncompatibleExports


def check_compatible(dt: AlertExport, rf: AlertExport) -> dict:
    problems = []
    if (dt.detector, rf.detector) != ("decision_tree", "random_forest"):
        problems.append(f"expected a decision_tree and a random_forest export, got {dt.detector} and {rf.detector}")
    if dt.schema_version != rf.schema_version:
        problems.append(f"schema versions differ: {dt.schema_version} vs {rf.schema_version}")
    if dt.run_id != rf.run_id:
        problems.append(f"run_id differs: {dt.run_id!r} vs {rf.run_id!r} (different validation executions)")
    if dt.dataset != rf.dataset:
        problems.append(f"dataset differs: {dt.dataset!r} vs {rf.dataset!r}")
    rf_by_record = {a.record_id: a for a in rf.alerts}
    shared = [a for a in dt.alerts if a.record_id in rf_by_record]
    mismatched = [a.record_id for a in shared if a.evidence != rf_by_record[a.record_id].evidence]
    if mismatched:
        problems.append(f"{len(mismatched)} shared record_id(s) carry different evidence, so the two files were not "
                        f"produced from the same rows in the same order (e.g. {mismatched[:5]})")
    if problems:
        raise IncompatibleExports("exports cannot be correlated: " + "; ".join(problems),
                                  details={"problems": problems})
    return {"run_id": dt.run_id, "dataset": dt.dataset, "shared_records_checked": len(shared)}


def correlate(dt: AlertExport, rf: AlertExport, dt_results: dict | None = None, rf_results: dict | None = None) -> dict:
    compat = check_compatible(dt, rf)
    for name, res, exp in (("decision_tree", dt_results, dt), ("random_forest", rf_results, rf)):
        if res is not None and res.get("source", {}).get("sha256") != exp.sha256:
            raise IncompatibleExports(f"{name} results were not produced from the supplied {name} export",
                                      code="RESULTS_EXPORT_MISMATCH")
    status = {}
    for name, res in (("decision_tree", dt_results), ("random_forest", rf_results)):
        status[name] = {r["record_id"]: r["status"] for r in (res or {}).get("results", [])}
    d = {a.record_id: a for a in dt.alerts}
    r = {a.record_id: a for a in rf.alerts}
    rows = []
    for rec in sorted(d.keys() | r.keys(), key=lambda x: int(x.split("-")[1])):
        a, b = d.get(rec), r.get(rec)
        kind = ("both_same_class" if a and b and a.predicted_class == b.predicted_class else
                "both_different_class" if a and b else "decision_tree_only" if a else "random_forest_only")
        rows.append({"record_id": rec, "agreement": kind,
                     "decision_tree": None if a is None else {"alert_id": a.alert_id, "predicted_class": a.predicted_class,
                                                              "explanation_status": status["decision_tree"].get(rec)},
                     "random_forest": None if b is None else {"alert_id": b.alert_id, "predicted_class": b.predicted_class,
                                                              "explanation_status": status["random_forest"].get(rec)}})
    counts = {k: sum(x["agreement"] == k for x in rows)
              for k in ("both_same_class", "both_different_class", "decision_tree_only", "random_forest_only")}
    warnings = []
    if compat["shared_records_checked"] == 0 and (dt.alerts or rf.alerts):
        warnings.append("the exports share no record_id, so row alignment could not be verified at all; treat the "
                        "correlation as unverified")
    return {
        "report": "nova-qwen-correlation", "report_version": "0.1.0-provisional",
        "compatibility": compat,
        "warnings": warnings,
        "limitations": [
            "record_id is a row position; correlation assumes both exports used the complete validation frame in "
            "its original order. Identical evidence on every shared record_id was verified; single-classifier "
            "alerts cannot be cross-checked.",
            "Explanations are not merged; each stays attached to its own classifier's alert.",
        ],
        "counts": counts,
        "disagreements": [x for x in rows if x["agreement"] == "both_different_class"],
        "records": rows,
    }
