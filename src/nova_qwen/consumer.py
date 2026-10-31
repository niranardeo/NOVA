"""The Qwen consumer: schema 1.1 alert export -> validated explanations file.

Flow per alert: normalise evidence in Python -> build prompt -> context-budget check -> Ollama chat
(one alert per request, JSON-schema constrained) -> validate -> on validation errors, one corrective
re-ask (configurable) -> record. Identifiers, predicted class and raw evidence are copied from the
export in code, never from the model. Empty exports never contact the model.
"""
from __future__ import annotations

from pathlib import Path

from . import __version__
from .alerts import AlertExport
from .config import QwenSettings
from .errors import NovaQwenError, OllamaError, PromptBudgetExceeded, ResumeMismatch
from .features import data_quality_notes, normalize_evidence
from .output import OUTPUT_SCHEMA, OUTPUT_SCHEMA_VERSION, load_results, utc_now, write_json_atomic
from .prompt import (PROMPT_VERSION, RESPONSE_SCHEMA_VERSION, build_messages, check_budget, feedback_message,
                     response_schema)
from .response import parse_content, validate_response

MAX_RAW_CHARS = 32000  # far above any answer the validator accepts (~6 KB); see verify
PREDICTION_STATUS = "unverified classifier prediction"
TRANSPORT_CODES = ("OLLAMA_UNAVAILABLE", "OLLAMA_TIMEOUT", "OLLAMA_HTTP_ERROR", "OLLAMA_PROTOCOL_ERROR")
FINGERPRINT_SETTINGS = ("model", "think", "temperature", "seed", "num_ctx", "num_predict", "max_attempts")


class RunAborted(OllamaError):
    code = "RUN_ABORTED"


def normalize_selection(*, limit: int | None, offset: int = 0, alert_ids=None) -> dict:
    """Canonical description of which alerts a run covers (stored in, and compared on, resume)."""
    if alert_ids:
        if offset:
            raise NovaQwenError("--offset cannot be combined with --alert-id", code="BAD_SELECTION")
        return {"mode": "alert_ids", "alert_ids": list(dict.fromkeys(alert_ids)), "offset": None, "limit": None}
    if offset < 0 or (limit is not None and limit < 1):
        raise NovaQwenError("offset must be >= 0 and limit >= 1", code="BAD_SELECTION")
    return {"mode": "all" if limit is None and offset == 0 else "range", "alert_ids": None,
            "offset": offset, "limit": limit}


def select_alerts(export: AlertExport, *, limit: int | None, offset: int = 0, alert_ids=None) -> tuple:
    """Return (alerts, selection record). ``limit=None`` means all alerts (explicit opt-in)."""
    selection = normalize_selection(limit=limit, offset=offset, alert_ids=alert_ids)
    if selection["mode"] == "alert_ids":
        by_id = {a.alert_id: a for a in export.alerts}
        missing = [i for i in selection["alert_ids"] if i not in by_id]
        if missing:
            raise NovaQwenError(f"alert ids not found in {export.file_name}: {missing}", code="ALERT_ID_NOT_FOUND")
        return [by_id[i] for i in selection["alert_ids"]], selection
    if export.alerts and offset >= len(export.alerts):
        raise NovaQwenError(f"--offset {offset} is past the last of {len(export.alerts)} alerts", code="BAD_SELECTION")
    end = None if limit is None else offset + limit
    return list(export.alerts[offset:end]), selection


def base_result(export: AlertExport, alert) -> dict:
    normalized = normalize_evidence(alert.evidence)
    return {
        "alert_id": alert.alert_id,
        "record_id": alert.record_id,
        "record_position": alert.record_position,
        "detector": export.detector,
        "run_id": export.run_id,
        "dataset": export.dataset,
        "schema_version": export.schema_version,
        "predicted_class": alert.predicted_class,
        "prediction_status": PREDICTION_STATUS,
        "evidence": dict(alert.evidence),
        "evidence_normalized": [f.to_record() for f in normalized],
        "data_quality_notes": data_quality_notes(normalized),
        "status": None,
        "response": None,
        "validation": None,
        "attempts": [],
        "error": None,
    }


def explain_alert(client, export: AlertExport, alert, settings: QwenSettings, *, send_think: bool = True) -> dict:
    result = base_result(export, alert)
    normalized = normalize_evidence(alert.evidence)
    notes = result["data_quality_notes"]
    messages = build_messages(export.detector, alert.predicted_class, normalized, notes)
    schema = response_schema([f.feature for f in normalized])
    last_validation, last_parsed = None, None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            size = check_budget(messages, schema, settings.num_ctx, settings.num_predict)
        except PromptBudgetExceeded as exc:
            if attempt == 1:
                result.update(status="failed", error=exc.to_dict())
                return result
            break  # a corrective re-ask would not fit: never send it, keep the last rejected answer
        try:
            chat = client.chat(messages, schema=schema, send_think=send_think)
        except OllamaError as exc:
            result.update(status="failed", error=exc.to_dict())
            result["attempts"].append({"attempt": attempt, "request_bytes": size, "error": exc.code})
            return result
        parsed, parse_error = parse_content(chat.content)
        validation = validate_response(parsed, normalized, notes, alert.predicted_class, chat.done_reason)
        if parsed is None:
            validation.errors = [{"code": "RESPONSE_NOT_JSON", "message": parse_error}] + \
                [e for e in validation.errors if e["code"] != "RESPONSE_NOT_OBJECT"]
        if chat.thinking:
            validation.warn("THINKING_RETURNED", "the model returned thinking text although think=false")
        result["attempts"].append({
            "attempt": attempt, "request_bytes": size, **chat.metadata(),
            "validation_errors": [e["code"] for e in validation.errors],
            "raw_response": chat.content[:MAX_RAW_CHARS],
            "raw_truncated": len(chat.content) > MAX_RAW_CHARS,
        })
        last_validation, last_parsed = validation, parsed
        if validation.passed:
            result.update(status="abstained" if parsed["abstain"] else "explained", response=parsed,
                          validation=validation.to_record())
            return result
        messages = messages + [{"role": "assistant", "content": chat.content[:MAX_RAW_CHARS]},
                               {"role": "user", "content": feedback_message(validation.errors)}]
    result.update(status="rejected", response=None, validation=last_validation.to_record(),
                  rejected_response=last_parsed)
    return result


def _summary(results: list, selected: int) -> dict:
    counts = {s: sum(r["status"] == s for r in results) for s in ("explained", "abstained", "rejected", "failed")}
    counts.update(selected=selected, processed=len(results), pending=selected - len(results))
    return counts


def fingerprint(doc: dict) -> dict:
    model = doc.get("model") or {}
    settings = model.get("settings") or {}
    generator = doc.get("generator") or {}
    return {"digest": model.get("digest"), "component_version": generator.get("version"),
            "prompt_version": generator.get("prompt_version"),
            "response_schema_version": generator.get("response_schema_version"),
            **{k: settings.get(k) for k in FINGERPRINT_SETTINGS}}


def _reusable(old_result: dict, fresh: dict) -> dict | None:
    """Reuse an earlier accepted answer only if the alert is unchanged and the answer still validates."""
    if old_result.get("status") not in ("explained", "abstained") or not isinstance(old_result.get("response"), dict):
        return None
    for key in ("record_id", "record_position", "detector", "run_id", "dataset", "predicted_class", "evidence",
                "evidence_normalized", "data_quality_notes"):
        if old_result.get(key) != fresh[key]:
            return None
    attempts = old_result.get("attempts") or []
    last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    raw, _ = parse_content(last.get("raw_response"))
    if raw != old_result["response"] or last.get("done_reason") != "stop":
        return None  # the stored answer is not what the model returned: never reuse it
    recheck = validate_response(old_result["response"], normalize_evidence(fresh["evidence"]),
                                fresh["data_quality_notes"], fresh["predicted_class"])
    if not recheck.passed:
        return None
    return {**fresh, "status": "abstained" if old_result["response"].get("abstain") else "explained",
            "response": old_result["response"], "validation": recheck.to_record(),
            "attempts": old_result.get("attempts") or []}


def run(export: AlertExport, settings: QwenSettings, output_path, *, client=None, limit: int | None = 10,
        offset: int = 0, alert_ids=None, resume: bool = False, selection_from_previous: bool = False,
        progress=None) -> dict:
    """Process the selected alerts of one export and write the results file (checkpointed per alert).

    resume=True continues ``output_path`` for the SAME export, selection and settings. With
    selection_from_previous=True the selection is taken from that file (what the CLI does when no
    selection flag is given); otherwise a different selection is refused, never silently narrowed."""
    old = None
    if resume and Path(output_path).exists():
        old = load_results(output_path)
        if old["source"].get("sha256") != export.sha256:
            raise ResumeMismatch("cannot resume: the results were produced from a different alert export")
        if selection_from_previous:
            prev = old["selection"]
            limit, offset, alert_ids = prev.get("limit"), prev.get("offset") or 0, prev.get("alert_ids")
    alerts, selection = select_alerts(export, limit=limit, offset=offset, alert_ids=alert_ids)
    if old is not None:
        prev = {k: old["selection"].get(k) for k in ("mode", "alert_ids", "offset", "limit")}
        if prev != selection:
            raise ResumeMismatch(f"cannot resume with a different selection: previous {prev}, now {selection}. "
                                 f"Run --resume without selection flags to continue the previous one.")
    selection["selected"] = len(alerts)
    doc = {
        "output_schema": OUTPUT_SCHEMA,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "generator": {"component": "nova_qwen", "version": __version__, "prompt_version": PROMPT_VERSION,
                      "response_schema_version": RESPONSE_SCHEMA_VERSION},
        "run_status": "in_progress",
        "created_utc": (old or {}).get("created_utc") or utc_now(),
        "updated_utc": None,
        "source": export.source_record(),
        "selection": selection,
        "model": {"contacted": False, "name": settings.model, "digest": None, "settings": settings.to_record()},
        "summary": None,
        "results": [],
    }
    previous = {}
    if old is not None:
        previous = {r.get("alert_id"): r for r in old["results"] if isinstance(r, dict)}
        doc["resumed_from"] = {"previous_updated_utc": old.get("updated_utc"), "reused_results": 0,
                               "previous_fingerprint": fingerprint(old)}

    if not alerts:
        # Nothing to explain: never contact Ollama, never invent an alert.
        doc.update(run_status="complete", updated_utc=utc_now(), summary=_summary([], 0))
        write_json_atomic(output_path, doc)
        return doc

    if client is None:
        from .ollama import OllamaClient
        client = OllamaClient(settings)
    info = client.check_model()
    doc["model"].update(info.to_record(), contacted=True)
    if old is not None and any(r.get("status") in ("explained", "abstained") for r in previous.values()):
        if fingerprint(doc) != doc["resumed_from"]["previous_fingerprint"]:
            raise ResumeMismatch(f"cannot resume with a different model, version or settings: previous "
                                 f"{doc['resumed_from']['previous_fingerprint']}, now {fingerprint(doc)}")
    send_think = (not info.capabilities) or ("thinking" in info.capabilities)

    def keep_unreached_previous():
        """On interruption, keep earlier results for alerts this resumed run had not reached yet (in order)."""
        tail = []
        for alert in alerts[len(doc["results"]):]:
            if alert.alert_id not in previous:
                break
            tail.append(previous[alert.alert_id])
        return doc["results"] + tail

    consecutive_transport_failures = 0
    try:
        for index, alert in enumerate(alerts, start=1):
            reused = _reusable(previous[alert.alert_id], base_result(export, alert)) \
                if alert.alert_id in previous else None
            if reused is not None:
                result = reused
                doc["resumed_from"]["reused_results"] += 1
            else:
                result = explain_alert(client, export, alert, settings, send_think=send_think)
            doc["results"].append(result)
            doc.update(updated_utc=utc_now(), summary=_summary(doc["results"], len(alerts)))
            write_json_atomic(output_path, doc)
            if progress:
                progress(index, len(alerts), result)
            failed_transport = result["status"] == "failed" and (result.get("error") or {}).get("code") in TRANSPORT_CODES
            consecutive_transport_failures = consecutive_transport_failures + 1 if failed_transport else 0
            if consecutive_transport_failures >= settings.abort_after_consecutive_failures and index < len(alerts):
                doc["results"] = keep_unreached_previous()
                doc.update(run_status="interrupted", updated_utc=utc_now(), summary=_summary(doc["results"], len(alerts)))
                write_json_atomic(output_path, doc)
                raise RunAborted(f"stopped after {consecutive_transport_failures} consecutive alerts failed to reach "
                                 f"the model ({result['error']['code']}); fix Ollama, then rerun with --resume",
                                 details={"processed": len(doc["results"]), "selected": len(alerts)})
    except KeyboardInterrupt:
        doc["results"] = keep_unreached_previous()
        doc.update(run_status="interrupted", updated_utc=utc_now(), summary=_summary(doc["results"], len(alerts)))
        write_json_atomic(output_path, doc)
        raise
    doc.update(run_status="complete", updated_utc=utc_now(), summary=_summary(doc["results"], len(alerts)))
    write_json_atomic(output_path, doc)
    return doc
