"""Human-review worksheet for explanation quality.

The automated checks are structural and deterministic only. Whether an explanation is faithful,
sensible and useful must be judged by people; this produces a CSV with the evidence, the answer,
the automated findings, and empty scoring columns. The scores map onto the team's criteria in
docs/qwen_evaluation_criteria.md (accuracy, evidence use, uncertainty, usefulness).
"""
from __future__ import annotations

import csv
import io

HUMAN_COLUMNS = [
    "H1_faithful_to_evidence_0_2", "H2_units_and_values_correct_0_1", "H3_no_invented_facts_0_1",
    "H4_prediction_treated_as_unverified_0_1", "H5_limitations_adequate_0_1", "H6_useful_to_analyst_0_2",
    "reviewer", "review_notes",
]
WORKSHEET_MARKER = HUMAN_COLUMNS[0]
RUBRIC = """Scoring guide (fill one row per alert; leave blank if not reviewed). Maps to docs/qwen_evaluation_criteria.md.
H1 faithful to evidence: 0 = contradicts or ignores the evidence, 1 = partly, 2 = fully consistent
H2 units and values correct: 1 = every value and unit matches the normalised evidence, else 0
H3 no invented facts: 1 = nothing beyond the supplied evidence (no ports, IPs, intent, thresholds, typical values), else 0
H4 prediction treated as unverified: 1 = never presented as a confirmed attack, else 0
H5 limitations adequate: 1 = states what the three features cannot show, else 0
H6 useful to an analyst: 0 = no value, 1 = some, 2 = clearly helps judge the alert
Automated columns come from deterministic checks only; they do not establish correctness."""
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _cell(value) -> str:
    """Text for a CSV cell, neutralised against spreadsheet formula injection."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA_PREFIXES) else text


def _join(items, sep=" | ") -> str:
    return sep.join(str(x) for x in items) if isinstance(items, list) else ("" if items is None else str(items))


def rows_from_results(docs: list) -> list:
    rows = []
    for doc in docs:
        for r in doc.get("results", []):
            resp = r.get("response") or r.get("rejected_response") or {}
            resp = resp if isinstance(resp, dict) else {}
            v = r.get("validation") or {}
            evidence_shown = "; ".join(f"{e.get('feature')} = {e.get('display')} {e.get('unit')}"
                                       for e in r.get("evidence_normalized") or [] if isinstance(e, dict))
            row = {
                "detector": r.get("detector"), "run_id": r.get("run_id"), "alert_id": r.get("alert_id"),
                "record_id": r.get("record_id"), "predicted_class": r.get("predicted_class"),
                "status": r.get("status"), "evidence_as_shown_to_model": evidence_shown,
                "evidence_raw": "; ".join(f"{k} = {val}" for k, val in (r.get("evidence") or {}).items()),
                "data_quality_notes": _join(r.get("data_quality_notes")),
                "observations": _join(resp.get("observations")), "explanation": resp.get("explanation"),
                "limitations": _join(resp.get("limitations")), "abstain": resp.get("abstain"),
                "automated_errors": " ".join(str(e.get("code")) for e in v.get("errors") or [] if isinstance(e, dict)),
                "automated_warnings": " ".join(str(w.get("code")) for w in v.get("warnings") or []
                                               if isinstance(w, dict)),
                "model_error": (r.get("error") or {}).get("code", "") if isinstance(r.get("error"), dict) else "",
            }
            rows.append({**{k: _cell(val) for k, val in row.items()}, **{c: "" for c in HUMAN_COLUMNS}})
    return rows


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    fields = list(rows[0].keys()) if rows else ["alert_id", *HUMAN_COLUMNS]
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
