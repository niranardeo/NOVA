"""Turn a run's counts into one honest verdict, shared by the end-to-end and smoke scripts.

Only a complete, verified run with at least one explanation and no rejected or failed answers is PASS. Abstaining
is a valid answer, so it does not fail a run, but it is always counted separately, and a run that explained nothing
does not count as a pass.

Alert-level meanings:
  explained   validated answer
  abstained   validated answer with abstain: true (the model's own flag, not a judgement of the evidence)
  rejected    the model answered, and the answer failed the checks (after the corrective re-ask)
  failed      no answer: transport, model or budget error
  incomplete  selected, but not processed (interrupted run)
  skipped     in the export, deliberately not selected (a bounded sample, not a problem)
"""
from __future__ import annotations

PASS = "PASS"
NO_EXPLANATIONS = "NO_EXPLANATIONS"
REJECTED_ANSWERS = "REJECTED_ANSWERS"
INCOMPLETE = "INCOMPLETE"
FAILED = "FAILED"
SKIPPED_NO_ALERTS = "SKIPPED_NO_ALERTS"
# Worst first: the overall verdict is the worst classifier verdict.
SEVERITY = (FAILED, INCOMPLETE, REJECTED_ANSWERS, NO_EXPLANATIONS, SKIPPED_NO_ALERTS, PASS)


def classifier_outcome(doc: dict | None, verify_report: dict | None, alerts_in_export: int,
                       selected: int | None = None) -> dict:
    """Verdict for one classifier. ``doc`` None means the classifier was never reached; ``selected`` is then how many
    alerts had been planned for it, so the overall totals still add up."""
    if doc is None:
        counts = None if selected is None else {
            "explained": 0, "abstained": 0, "rejected": 0, "failed": 0, "incomplete": selected,
            "skipped": alerts_in_export - selected, "selected": selected, "alerts_in_export": alerts_in_export}
        return {"verdict": INCOMPLETE, "counts": counts, "reasons": ["this classifier was not processed"],
                "verified": None}
    s = doc.get("summary") or {}
    counts = {"explained": s.get("explained", 0), "abstained": s.get("abstained", 0), "rejected": s.get("rejected", 0),
              "failed": s.get("failed", 0), "incomplete": s.get("pending", 0),
              "skipped": alerts_in_export - s.get("selected", 0), "selected": s.get("selected", 0),
              "alerts_in_export": alerts_in_export}
    verified = bool(verify_report and verify_report.get("passed"))
    reasons = []
    if counts["failed"]:
        reasons.append(f"{counts['failed']} alerts failed with a technical error")
    if doc.get("run_status") != "complete" or counts["incomplete"]:
        reasons.append(f"run_status {doc.get('run_status')!r}, {counts['incomplete']} selected alerts not processed")
    if counts["rejected"]:
        reasons.append(f"{counts['rejected']} answers were rejected and remain unresolved")
    if not verified:
        reasons.append("independent verification did not pass: "
                       + "; ".join((verify_report or {}).get("problems", ["no report"]))[:400])

    complete = doc.get("run_status") == "complete" and not counts["incomplete"]
    if counts["failed"] or (complete and not verified):
        verdict = FAILED
    elif not complete:
        verdict = INCOMPLETE
    elif counts["rejected"]:
        verdict = REJECTED_ANSWERS
    elif alerts_in_export == 0:
        verdict, reasons = SKIPPED_NO_ALERTS, ["the export has no alerts, so nothing was sent to the model"]
    elif not counts["explained"]:
        verdict = NO_EXPLANATIONS
        reasons.append(f"all {counts['abstained']} processed alerts abstained; nothing was explained")
    else:
        verdict = PASS
    return {"verdict": verdict, "counts": counts, "reasons": reasons, "verified": verified}


def overall_outcome(per_classifier: dict) -> dict:
    """Overall verdict: the worst classifier verdict. PASS needs every classifier to pass."""
    verdicts = {name: o["verdict"] for name, o in per_classifier.items()}
    overall = next(v for v in SEVERITY if v in verdicts.values())
    totals: dict = {}
    for o in per_classifier.values():
        for key, value in (o.get("counts") or {}).items():
            totals[key] = totals.get(key, 0) + value
    return {"verdict": overall, "classifiers": verdicts, "totals": totals, "success": overall == PASS}


def counts_line(counts: dict | None) -> str:
    if not counts:
        return "not processed"
    return ("explained={explained} abstained={abstained} rejected={rejected} failed={failed} "
            "incomplete={incomplete} skipped={skipped}").format(**counts)
