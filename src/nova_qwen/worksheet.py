"""Human-review worksheet for explanation quality.

The automated checks are structural and deterministic only. Whether an explanation is faithful,
sensible and useful must be judged by people; this produces a CSV with the evidence, the answer,
the automated findings, and empty scoring columns. The scoring columns are the five criteria in
docs/qwen_evaluation_criteria.md (0-2 each), plus analyst usefulness, which is reported separately.
Guidance and examples: docs/qwen_review_rubric.md.
"""
from __future__ import annotations

import csv
import io
import json

PRIORITIZATION_COLUMN = "prioritization_0_2_or_NA"
HUMAN_COLUMNS = [
    "explanation_accuracy_0_2", "evidence_grounding_0_2", "uncertainty_handling_0_2", "consistency_0_2",
    PRIORITIZATION_COLUMN, "analyst_usefulness_0_2_not_in_total", "reviewer", "review_notes",
]
WORKSHEET_MARKER = HUMAN_COLUMNS[0]
EARLIER_MARKERS = ("H1_faithful_to_evidence_0_2",)  # sheets from version 0.1 may hold human scores too
NO_PRIORITY_POLICY = "N/A"
RARE_CLASS_NOTE = ("Rare class with weaker classifier recall in the preliminary results: review closely. This alert "
                   "shows only what was flagged, never what the classifier missed.")
RARE_CLASSES = ("Bots", "Web Attacks")
RUBRIC = """Scoring guide (fill one row per alert; leave blank if not reviewed). Full guide: docs/qwen_review_rubric.md.
Criteria are from docs/qwen_evaluation_criteria.md: 0 = does not meet, 1 = partially meets, 2 = fully meets.
Explanation accuracy: explains the prediction correctly, and never presents it as a confirmed attack.
Evidence grounding: uses only the supplied values and units, and invents nothing (ports, IPs, intent, typical values).
Uncertainty handling: says what is missing, or what the three features cannot show. An abstention can score 2.
Consistency (a proposed reading of that criterion): score rows that share a consistency_group (the same flow seen by
  both classifiers). Answers should not contradict each other on the facts, though the predicted classes may differ.
  Leave blank if the row has no group.
Prioritization: N/A. No priority policy is approved, so Qwen always returns priority null. Report totals out of 8
  for rows with a consistency score, and out of 6 for rows where consistency is blank.
Analyst usefulness (0-2): recorded separately, not part of the total.
Automated columns come from deterministic checks only; they do not establish correctness."""
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def is_worksheet_header(head: str) -> bool:
    """Is this file's first line a worksheet header (current or earlier version)?"""
    first = head.splitlines()[0] if head else ""
    return any(marker in first for marker in (WORKSHEET_MARKER, *EARLIER_MARKERS))


def _cell(value) -> str:
    """Text for a CSV cell, neutralised against spreadsheet formula injection."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA_PREFIXES) else text


def _join(items, sep=" | ") -> str:
    return sep.join(str(x) for x in items) if isinstance(items, list) else ("" if items is None else str(items))


def _same_row_key(r: dict) -> tuple:
    """record_id is only a position, so "same flow" needs the same run, dataset and evidence as well."""
    return (r.get("dataset"), r.get("run_id"), r.get("record_id"), json.dumps(r.get("evidence"), sort_keys=True))


def rows_from_results(docs: list) -> list:
    seen_in = {}
    for doc in docs:
        for r in doc.get("results", []):
            seen_in.setdefault(_same_row_key(r), set()).add(r.get("detector"))
    several_runs = len({(r.get("dataset"), r.get("run_id")) for doc in docs for r in doc.get("results", [])}) > 1
    rows = []
    for doc in docs:
        provenance = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        for r in doc.get("results", []):
            resp = r.get("response") or r.get("rejected_response") or {}
            resp = resp if isinstance(resp, dict) else {}
            v = r.get("validation") or {}
            evidence_shown = "; ".join(f"{e.get('feature')} = {e.get('display')} {e.get('unit')}"
                                       for e in r.get("evidence_normalized") or [] if isinstance(e, dict))
            row = {
                "detector": r.get("detector"), "run_id": r.get("run_id"), "dataset": r.get("dataset"),
                "dataset_version": provenance.get("dataset_version", "not recorded"),
                "dataset_verified": provenance.get("verified", False),
                "alert_id": r.get("alert_id"),
                "record_id": r.get("record_id"), "predicted_class": r.get("predicted_class"),
                "consistency_group": ("" if len(seen_in.get(_same_row_key(r), ())) < 2 else
                                      f"{r.get('run_id')}/{r.get('record_id')}" if several_runs else r.get("record_id")),
                "class_review_note": RARE_CLASS_NOTE if r.get("predicted_class") in RARE_CLASSES else "",
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
            human = {c: NO_PRIORITY_POLICY if c == PRIORITIZATION_COLUMN else "" for c in HUMAN_COLUMNS}
            rows.append({**{k: _cell(val) for k, val in row.items()}, **human})
    return rows


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    fields = list(rows[0].keys()) if rows else ["alert_id", *HUMAN_COLUMNS]
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
