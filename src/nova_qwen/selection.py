"""Choose a small, deterministic, representative set of alerts to send to Qwen.

A real export holds tens of thousands of alerts and the first few are not representative. This picks the cases the
team's criteria ask about: common and rare classes, a class disagreement between the classifiers (both alerted, on
different classes), and unusual flow durations. Where possible both classifiers explain the same row, so a reviewer can
compare the answers. No randomness: the same exports always give the same selection. Anything not covered is reported,
never dropped silently.
"""
from __future__ import annotations

from .alerts import AlertExport

FIRST_CLASSES = ("DoS", "DDoS", "Bots", "Web Attacks")
LATER_CLASSES = ("Port Scanning", "Brute Force")
DISAGREEMENT = "dt_rf_class_disagreement"  # both alerted, on different classes (correlate: both_different_class)
ONE_CLASSIFIER_ONLY = "one_classifier_only"  # the other classifier raised no alert on this row
LONGEST = "longest_flow_duration"
NEGATIVE = "negative_flow_duration"
FILL = "row_order_fill"
REQUESTED = "requested_by_alert_id"


def _duration(alert):
    return alert.evidence.get("Flow Duration")


def _targets(export: AlertExport, other: AlertExport) -> list:
    """Ordered (name, matches, preferred) tests. ``preferred`` picks among matches first."""
    other_by_record = {a.record_id: a for a in other.alerts}

    def shared_same_class(a):
        b = other_by_record.get(a.record_id)
        return b is not None and b.predicted_class == a.predicted_class

    def different_class(a):
        b = other_by_record.get(a.record_id)
        return b is not None and b.predicted_class != a.predicted_class

    def class_target(cls):
        return f"class:{cls}", (lambda a: a.predicted_class == cls), shared_same_class

    durations = [a for a in export.alerts if isinstance(_duration(a), (int, float))]
    longest = max(durations, key=_duration).alert_id if durations else None
    return [
        *[class_target(c) for c in FIRST_CLASSES],
        (DISAGREEMENT, different_class, None),
        (LONGEST, (lambda a: a.alert_id == longest), None),
        (NEGATIVE, (lambda a: isinstance(_duration(a), (int, float)) and _duration(a) < 0), None),
        (ONE_CLASSIFIER_ONLY, (lambda a: a.record_id not in other_by_record), None),
        *[class_target(c) for c in LATER_CLASSES],
    ]


def _plan_one(export: AlertExport, other: AlertExport, limit: int) -> dict:
    chosen: dict = {}
    credited: dict = {}  # alert_id -> the targets this row was selected or counted for
    covered, unavailable, over_limit = [], [], []
    for name, matches, preferred in _targets(export, other):
        hit = next((a for a in chosen.values() if matches(a)), None)
        if hit is not None:
            credited[hit.alert_id].append(name)
            covered.append(name)
            continue
        candidates = [a for a in export.alerts if matches(a)]
        if not candidates:
            unavailable.append(name)
        elif len(chosen) >= limit:
            over_limit.append(name)
        else:
            best = next((a for a in candidates if preferred and preferred(a)), candidates[0])
            chosen[best.alert_id] = best
            credited[best.alert_id] = [name]
            covered.append(name)
    for alert in export.alerts:
        if len(chosen) >= limit:
            break
        chosen.setdefault(alert.alert_id, alert)
    ordered = sorted(chosen.values(), key=lambda a: a.record_position)
    alerts = [{"alert_id": a.alert_id, "record_id": a.record_id, "predicted_class": a.predicted_class,
               "reasons": credited.get(a.alert_id) or [FILL]} for a in ordered]
    return {"alert_ids": [a.alert_id for a in ordered], "alerts": alerts, "alerts_in_export": len(export.alerts),
            "covered": covered, "not_available_in_export": unavailable, "not_covered_because_of_limit": over_limit}


def plan_representative(dt: AlertExport, rf: AlertExport, limit: int) -> dict:
    return {"method": "representative", "limit": limit, "decision_tree": _plan_one(dt, rf, limit),
            "random_forest": _plan_one(rf, dt, limit)}


def plan_first(dt: AlertExport, rf: AlertExport, limit: int) -> dict:
    """The first alerts in row order (the earlier default, kept for exact comparisons)."""
    plan = {"method": "first", "limit": limit}
    for export in (dt, rf):
        ids = [a.alert_id for a in export.alerts[:limit]]
        plan[export.detector] = {
            "alert_ids": ids, "alerts_in_export": len(export.alerts),
            "alerts": [{"alert_id": a.alert_id, "record_id": a.record_id, "predicted_class": a.predicted_class,
                        "reasons": [FILL]} for a in export.alerts[:limit]],
            "covered": [], "not_available_in_export": [], "not_covered_because_of_limit": []}
    return plan


def add_requested(plan: dict, export: AlertExport, other: AlertExport, requested: list) -> None:
    """Append explicitly requested alert ids (already checked to exist) to one classifier's plan."""
    entry = plan[export.detector]
    by_id = {a.alert_id: a for a in export.alerts}
    known = {a["alert_id"]: a for a in entry["alerts"]}
    for alert_id in dict.fromkeys(requested):
        if alert_id in known:
            known[alert_id]["reasons"].append(REQUESTED)
            continue
        a = by_id[alert_id]
        record = {"alert_id": a.alert_id, "record_id": a.record_id, "predicted_class": a.predicted_class,
                  "reasons": [REQUESTED]}
        entry["alerts"].append(record)
        entry["alert_ids"].append(alert_id)
        known[alert_id] = record
    tests = {name: matches for name, matches, _ in _targets(export, other)}
    added = [by_id[i] for i in dict.fromkeys(requested)]
    now_covered = [n for n in entry["not_covered_because_of_limit"] if any(tests[n](a) for a in added)]
    entry["covered"].extend(now_covered)
    entry["not_covered_because_of_limit"] = [n for n in entry["not_covered_because_of_limit"] if n not in now_covered]
