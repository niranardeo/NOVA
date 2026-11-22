"""Independent check of a results file against the alert export it claims to come from.

Nothing in the results file is trusted: identifiers, classes, evidence, normalisation and data-quality
notes are re-derived from the export; accepted answers are re-validated and compared with the model's
recorded raw output; the alerts covered must be exactly the recorded selection.
"""
from __future__ import annotations

from .alerts import AlertExport
from .consumer import PREDICTION_STATUS, base_result, select_alerts
from .errors import NovaQwenError
from .features import normalize_evidence
from .output import OUTPUT_SCHEMA_VERSION, RUN_STATUSES, STATUSES
from .response import parse_content, validate_response

DERIVED_KEYS = ("record_id", "record_position", "detector", "run_id", "dataset", "schema_version",
                "predicted_class", "prediction_status", "evidence", "evidence_normalized", "data_quality_notes")


def verify_results(doc: dict, export: AlertExport, *, allow_incomplete: bool = False) -> dict:
    """Return {"passed", "complete", "problems", "checked"}. A run that is not complete fails unless
    allow_incomplete=True (then only what was processed so far is checked)."""
    problems = []
    try:
        _verify(doc, export, problems, allow_incomplete)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:  # malformed structure: report, never crash
        problems.append(f"malformed results file: {type(exc).__name__}: {exc}")
    results = doc.get("results") if isinstance(doc, dict) and isinstance(doc.get("results"), list) else []
    return {"passed": not problems, "complete": isinstance(doc, dict) and doc.get("run_status") == "complete",
            "problems": problems, "checked": len(results)}


def _verify(doc, export, problems, allow_incomplete):
    bad = problems.append
    if doc.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        bad(f"unexpected output_schema_version {doc.get('output_schema_version')!r}")
    run_status = doc.get("run_status")
    if run_status not in RUN_STATUSES:
        bad(f"invalid run_status {run_status!r}")
    elif run_status != "complete" and not allow_incomplete:
        bad(f"run_status is {run_status!r}, not 'complete' (resume it, or verify with --allow-incomplete)")
    src = doc.get("source") or {}
    for key, expected in export.source_record().items():
        if key != "file_name" and src.get(key) != expected:
            bad(f"source.{key} is {src.get(key)!r}, the export has {expected!r}")

    results = doc["results"]
    selection = doc.get("selection") or {}
    try:
        expected_alerts, _ = select_alerts(export, limit=selection.get("limit"), offset=selection.get("offset") or 0,
                                           alert_ids=selection.get("alert_ids"))
        expected_ids = [a.alert_id for a in expected_alerts]
    except NovaQwenError as exc:
        bad(f"selection {selection} cannot be applied to the export: {exc}")
        expected_ids = None
    actual_ids = [r.get("alert_id") for r in results]
    if expected_ids is not None:
        if selection.get("selected") != len(expected_ids):
            bad(f"selection.selected is {selection.get('selected')}, the selection covers {len(expected_ids)} alerts")
        if run_status == "complete" and actual_ids != expected_ids:
            bad(f"results do not cover exactly the selected alerts in order "
                f"(missing {sorted(set(expected_ids) - set(actual_ids))[:5]}, extra {sorted(set(actual_ids) - set(expected_ids))[:5]})")
        elif actual_ids != expected_ids[:len(actual_ids)]:
            bad("results are not a prefix of the selected alerts in order")

    by_id = {a.alert_id: a for a in export.alerts}
    seen = set()
    for r in results:
        aid = r.get("alert_id")
        if aid in seen:
            bad(f"{aid}: duplicate result")
        seen.add(aid)
        alert = by_id.get(aid)
        if alert is None:
            bad(f"{aid}: not present in the source export (fabricated or wrong file)")
            continue
        fresh = base_result(export, alert)
        for k in DERIVED_KEYS:
            if r.get(k) != fresh[k]:
                bad(f"{aid}: {k} differs from the source alert ({r.get(k)!r} != {fresh[k]!r})"[:300])
        status = r.get("status")
        if status not in STATUSES:
            bad(f"{aid}: invalid status {status!r}")
            continue
        attempts = r.get("attempts") or []
        if status in ("explained", "abstained"):
            resp = r.get("response")
            if not isinstance(resp, dict):
                bad(f"{aid}: status {status} without a response")
                continue
            if (status == "abstained") != (resp.get("abstain") is True):
                bad(f"{aid}: status {status} disagrees with response.abstain")
            last = attempts[-1] if attempts else {}
            raw, _ = parse_content(last.get("raw_response")) if not last.get("raw_truncated") else (None, None)
            if raw != resp:
                bad(f"{aid}: the accepted response is not the model's recorded output (attempts[-1].raw_response)")
            if last.get("done_reason") != "stop":
                bad(f"{aid}: accepted answer did not end with done_reason 'stop'")
            recheck = validate_response(resp, normalize_evidence(alert.evidence), fresh["data_quality_notes"],
                                        alert.predicted_class)
            if not recheck.passed:
                bad(f"{aid}: stored response fails validation on re-check: {[e['code'] for e in recheck.errors]}")
        else:
            if r.get("response") is not None:
                bad(f"{aid}: status {status} must not carry an accepted response")
            if status == "rejected" and not (r.get("validation") or {}).get("errors"):
                bad(f"{aid}: rejected without validation errors")
            if status == "failed" and not (r.get("error") or {}).get("code"):
                bad(f"{aid}: failed without an error code")
    summary = doc.get("summary") or {}
    if summary.get("processed") != len(results):
        bad(f"summary.processed {summary.get('processed')} != {len(results)} results")
    for s in STATUSES:
        if summary.get(s) != sum(r.get("status") == s for r in results):
            bad(f"summary.{s} does not match the results")
    model = doc.get("model") or {}
    if not export.alerts and model.get("contacted"):
        bad("the model was contacted for an empty export")
    if any(r.get("attempts") for r in results):
        if not model.get("contacted") or not isinstance(model.get("digest"), str):
            bad("results contain model answers but the model block says it was not contacted / has no digest")
        if model.get("expected_digest") and not (model.get("digest_verified") and
                                                 model.get("digest") == model.get("expected_digest")):
            bad("model digest is not the verified expected digest")
    if any(r.get("prediction_status") != PREDICTION_STATUS for r in results):
        bad("prediction_status must stay 'unverified classifier prediction'")
