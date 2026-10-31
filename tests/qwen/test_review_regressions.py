"""Regression tests for every finding of the independent adversarial review (evidence in the release
report). Each test reproduces the reviewed defect and asserts the fixed behaviour."""
import copy
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from qwen_test_support import (DIGEST, DT_EMPTY, DT_FIXTURE, MODEL, RF_FIXTURE, ROOT, chat_body, evidence_from_request,
                               load_json, make_settings, valid_answer, write_json)
from src.nova_qwen.alerts import load_export, parse_export
from src.nova_qwen.config import QwenSettings
from src.nova_qwen.consumer import FINGERPRINT_SETTINGS, MAX_RAW_CHARS, RunAborted, run, select_alerts
from src.nova_qwen.correlate import correlate
from src.nova_qwen.errors import (AlertContractError, AlertFileInvalidJSON, ConfigError, ModelDigestMismatch,
                                  NovaQwenError, OllamaHTTPError, OllamaProtocolError, OllamaTimeout, OutputError,
                                  ResultsFileError, ResumeMismatch)
from src.nova_qwen.ollama import OllamaClient
from src.nova_qwen.output import check_output_path, load_results
from src.nova_qwen.prompt import request_bytes, response_schema
from src.nova_qwen.verify import verify_results
from src.nova_qwen.worksheet import rows_from_results, to_csv


def cli(*args, env_extra=None, cwd=ROOT, timeout=120):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "src.nova_qwen", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


# =========================================================================== resume (High)
def test_resume_without_flags_continues_the_original_all_selection(fake_ollama, tmp_path):
    """Review: resuming an interrupted --all run without repeating --all silently dropped results."""
    out = tmp_path / "r.json"
    export = load_export(DT_FIXTURE)
    s = make_settings(fake_ollama.url)
    with pytest.raises(KeyboardInterrupt):
        run(export, s, out, limit=None, progress=lambda i, n, r: (_ for _ in ()).throw(KeyboardInterrupt) if i == 2 else 0)
    assert len(load_json(out)["results"]) == 2
    r = cli("explain", DT_FIXTURE, "--out", out, "--resume", "--quiet", env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr
    doc = load_json(out)
    assert doc["selection"]["mode"] == "all" and len(doc["results"]) == 5 and doc["resumed_from"]["reused_results"] == 2
    assert verify_results(doc, export)["passed"]


def test_resume_with_a_different_selection_is_refused(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=None)
    with pytest.raises(ResumeMismatch):
        run(load_export(DT_FIXTURE), settings, out, limit=2, resume=True)
    r = cli("explain", DT_FIXTURE, "--out", out, "--resume", "--limit", "2", env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESUME_MISMATCH"
    assert len(load_json(out)["results"]) == 5  # nothing was lost


def test_resume_on_missing_file_starts_fresh(fake_ollama, settings, tmp_path):
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "new.json", limit=1, resume=True)
    assert doc["run_status"] == "complete" and "resumed_from" not in doc


def test_resume_does_not_reuse_a_tampered_or_now_invalid_answer(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=2)
    doc = load_json(out)
    doc["results"][0]["response"]["explanation"] = "This confirms the attack lasted 3 years."
    write_json(out, doc)
    before = len(fake_ollama.chat_requests())
    new = run(load_export(DT_FIXTURE), settings, out, limit=2, resume=True)
    assert len(fake_ollama.chat_requests()) - before == 1 and new["resumed_from"]["reused_results"] == 1
    assert "3 years" not in json.dumps(new["results"][0]["response"])


@pytest.mark.parametrize("key", list(FINGERPRINT_SETTINGS) + ["digest", "prompt_version", "component_version",
                                                              "response_schema_version"])
def test_every_fingerprint_component_blocks_resume(fake_ollama, settings, tmp_path, key):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=1)
    doc = load_json(out)
    if key in FINGERPRINT_SETTINGS:
        doc["model"]["settings"][key] = "changed"
    elif key == "digest":
        doc["model"]["digest"] = "f" * 64
    else:
        doc["generator"][{"prompt_version": "prompt_version", "component_version": "version",
                          "response_schema_version": "response_schema_version"}[key]] = "changed"
    write_json(out, doc)
    with pytest.raises(ResumeMismatch):
        run(load_export(DT_FIXTURE), settings, out, limit=1, resume=True)


def test_resume_keeps_created_utc(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    first = run(load_export(DT_FIXTURE), settings, out, limit=1)
    time.sleep(0.01)
    again = run(load_export(DT_FIXTURE), settings, out, limit=1, resume=True)
    assert again["created_utc"] == first["created_utc"] and again["updated_utc"] != first["updated_utc"]


# =========================================================================== selection
def test_selection_edge_cases():
    export = load_export(DT_FIXTURE)
    for kw in ({"limit": None, "offset": 1, "alert_ids": ["dt-000002"]}, {"limit": 1, "offset": 5},
               {"limit": 1, "offset": 50}):
        with pytest.raises(NovaQwenError) as e:
            select_alerts(export, **kw)
        assert e.value.code == "BAD_SELECTION"
    assert select_alerts(load_export(DT_EMPTY), limit=1, offset=0)[0] == []


def test_cli_rejects_offset_with_alert_id(tmp_path):
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "r.json", "--alert-id", "dt-000002", "--offset", "1")
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "BAD_SELECTION"


# =========================================================================== run robustness
def test_run_stops_after_consecutive_transport_failures(fake_ollama, tmp_path):
    fake_ollama.default_chat = lambda b: ("json", 500, {"error": "model runner has unexpectedly stopped"})
    s = make_settings(fake_ollama.url, transport_retries=0, abort_after_consecutive_failures=3)
    out = tmp_path / "r.json"
    with pytest.raises(RunAborted) as e:
        run(load_export(DT_FIXTURE), s, out, limit=None)
    assert e.value.code == "RUN_ABORTED"
    doc = load_json(out)
    assert doc["run_status"] == "interrupted" and len(doc["results"]) == 3
    assert len(fake_ollama.chat_requests()) == 3
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "c.json", "--all", "--retries", "0", "--quiet",
            env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 3 and json.loads(r.stderr)["error"]["code"] == "RUN_ABORTED"


def test_nan_in_model_answer_is_rejected_not_a_crash(fake_ollama, tmp_path):
    fake_ollama.chat_actions.append(("json", 200, chat_body('{"observations": [NaN]}')))
    s = make_settings(fake_ollama.url, max_attempts=1)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", limit=2)
    assert doc["results"][0]["status"] == "rejected"
    assert doc["results"][0]["attempts"][0]["validation_errors"][0] == "RESPONSE_NOT_JSON"
    assert doc["results"][1]["status"] == "explained" and doc["run_status"] == "complete"


def test_reask_that_does_not_fit_is_never_sent(fake_ollama, tmp_path):
    """Review P05: the first request fits, the corrective re-ask would not: keep the rejection, send nothing."""
    export = load_export(DT_FIXTURE)
    huge = "x" * 3000
    fake_ollama.chat_actions.append(lambda b: ("json", 200, chat_body(huge)))
    first_bytes = request_bytes([{"role": "user", "content": "x"}], response_schema(["Flow Duration"]))
    from src.nova_qwen.prompt import TEMPLATE_OVERHEAD_TOKENS, build_messages
    from src.nova_qwen.features import normalize_evidence
    n = normalize_evidence(export.alerts[0].evidence)
    size = request_bytes(build_messages("decision_tree", "DoS", n, []), response_schema([f.feature for f in n]))
    s = make_settings(fake_ollama.url, num_ctx=size + TEMPLATE_OVERHEAD_TOKENS + 512 + 100, num_predict=512)
    doc = run(export, s, tmp_path / "r.json", alert_ids=["dt-000002"])
    r = doc["results"][0]
    assert first_bytes > 0 and r["status"] == "rejected" and len(fake_ollama.chat_requests()) == 1


def test_think_parameter_omitted_when_model_lacks_thinking(fake_ollama, settings, tmp_path):
    fake_ollama.capabilities = ["completion"]
    run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", limit=1)
    assert "think" not in fake_ollama.chat_requests()[0]


def test_raw_response_is_truncated_for_audit(fake_ollama, tmp_path):
    fake_ollama.chat_actions.append(("json", 200, chat_body("y" * (MAX_RAW_CHARS + 50))))
    doc = run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, max_attempts=1), tmp_path / "r.json", limit=1)
    a = doc["results"][0]["attempts"][0]
    assert len(a["raw_response"]) == MAX_RAW_CHARS and a["raw_truncated"] is True


def test_answer_from_another_model_is_refused(fake_ollama, tmp_path):
    fake_ollama.chat_actions.append(lambda b: ("json", 200, dict(chat_body(valid_answer(evidence_from_request(b))),
                                                                 model="llama3:8b")))
    doc = run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, transport_retries=0), tmp_path / "r.json", limit=1)
    assert doc["results"][0]["status"] == "failed" and doc["results"][0]["error"]["code"] == "OLLAMA_PROTOCOL_ERROR"


# =========================================================================== client hardening
def test_trickling_server_hits_the_overall_deadline(fake_ollama):
    fake_ollama.chat_actions.append(("trickle", 10_000, 0.2))
    c = OllamaClient(make_settings(fake_ollama.url, timeout_seconds=1.0, transport_retries=0))
    start = time.monotonic()
    with pytest.raises(OllamaTimeout):
        c.chat([{"role": "user", "content": "u"}], schema=None)
    assert time.monotonic() - start < 4.0


@pytest.mark.parametrize("action, exc", [
    (("raw", 200, b"[" * 200_000 + b"]" * 200_000, "application/json"), OllamaProtocolError),
    (("raw", 400, b"[" * 200_000 + b"]" * 200_000, "application/json"), OllamaHTTPError),
    (("json", 200, {"model": MODEL, "message": {"content": "{}", "thinking": 5}, "done": True}), OllamaProtocolError),
    (("json", 200, {"model": MODEL, "message": {"content": "{}"}, "done": True, "done_reason": 7}), OllamaProtocolError),
    (("json", 200, {"model": MODEL, "message": {"content": "{}"}, "prompt_eval_count": "3"}), OllamaProtocolError),
    (("json", 200, {"error": "boom", "message": {"content": "{}"}}), OllamaProtocolError),
    (("raw", 200, b'{"message": {"content": "x"}, "eval_count": NaN}', "application/json"), OllamaProtocolError),
])
def test_hostile_response_shapes_become_typed_errors(fake_ollama, action, exc):
    fake_ollama.chat_actions.append(action)
    c = OllamaClient(make_settings(fake_ollama.url, transport_retries=0))
    with pytest.raises(exc):
        c.chat([{"role": "user", "content": "u"}], schema=None)


def test_hostile_tags_and_show_shapes(fake_ollama):
    c = OllamaClient(make_settings(fake_ollama.url))
    fake_ollama.model_entry = dict(fake_ollama.model_entry, details="not a dict")
    fake_ollama.capabilities = 7
    info = c.check_model()
    assert info.quantization is None and info.capabilities == ()


def test_huge_http_error_text_is_capped(fake_ollama):
    fake_ollama.chat_actions.append(("json", 400, {"error": "E" * 900_000}))
    c = OllamaClient(make_settings(fake_ollama.url, transport_retries=0))
    with pytest.raises(OllamaHTTPError) as e:
        c.chat([{"role": "user", "content": "u"}], schema=None)
    assert len(str(e.value)) < 600


def test_digest_must_match_in_full(fake_ollama):
    fake_ollama.model_entry = dict(fake_ollama.model_entry, digest=DIGEST[:12] + "0" * 52)
    with pytest.raises(ModelDigestMismatch):
        OllamaClient(make_settings(fake_ollama.url)).check_model()


@pytest.mark.parametrize("url", ["http://127.0.0.1:99999", "http://127.0.0.1:abc", "http://127.0.0.1:0",
                                 "http://127.0.0.1:-1"])
def test_invalid_ports_are_config_errors(url):
    with pytest.raises(ConfigError) as e:
        QwenSettings(url=url).validate()
    assert e.value.code == "CONFIG_URL"


@pytest.mark.parametrize("kw", [{"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
                                {"retry_backoff_seconds": float("nan")}, {"temperature": float("nan")}])
def test_non_finite_settings_rejected(kw):
    with pytest.raises(ConfigError) as e:
        QwenSettings(**kw).validate()
    assert e.value.code == "CONFIG_NOT_FINITE"


def test_cli_accepts_uppercase_digest(fake_ollama):
    r = cli("check-ollama", "--expected-digest", DIGEST.upper(), env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr


def test_cli_timeout_nan_is_an_input_error():
    assert cli("check-ollama", "--timeout", "nan").returncode == 2


# =========================================================================== reader hardening
VALID = load_json(DT_FIXTURE)


@pytest.mark.parametrize("ids, code", [
    (("dt-0000002", "val-0000002"), "CONTRACT_ID_NOT_CANONICAL"),
    (("dt-0000002", "val-000002"), "CONTRACT_ID_NOT_CANONICAL"),
    (("dt-000002", "val-0000002"), "CONTRACT_ID_NOT_CANONICAL"),
    (("dt-٠٠٠٠٠٢", "val-000002"), "CONTRACT_ALERT_ID"),
    (("dt-000002", "val-٠٠٠٠٠٢"), "CONTRACT_RECORD_ID"),
])
def test_non_canonical_ids_rejected(ids, code):
    doc = copy.deepcopy(VALID)
    doc["alerts"][0].update(alert_id=ids[0], record_id=ids[1])
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert e.value.code == code


def test_seven_digit_canonical_ids_accepted():
    doc = copy.deepcopy(VALID)
    doc["alerts"] = [dict(VALID["alerts"][0], alert_id="dt-1000001", record_id="val-1000001")]
    assert parse_export(doc).alerts[0].record_position == 1_000_001


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 20), 1e16])
def test_out_of_range_evidence(value):
    doc = copy.deepcopy(VALID)
    doc["alerts"][0]["evidence"]["Flow Duration"] = value
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert e.value.code == "CONTRACT_EVIDENCE_RANGE"


def test_deeply_nested_export_is_invalid_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("[" * 200_000 + "]" * 200_000)
    with pytest.raises(AlertFileInvalidJSON):
        load_export(p)


@pytest.mark.parametrize("bad", ["validation x", "validation x", "validation﻿x"])
def test_line_and_paragraph_separators_rejected(bad):
    doc = copy.deepcopy(VALID)
    doc["run_id"] = bad
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert e.value.code == "CONTRACT_RUN_ID"


# =========================================================================== results files and paths
def test_load_results_errors_are_typed_input_errors(tmp_path):
    with pytest.raises(ResultsFileError) as e:
        load_results(tmp_path / "missing.json")
    assert e.value.code == "RESULTS_MISSING"
    write_json(tmp_path / "other.json", {"x": 1})
    with pytest.raises(ResultsFileError) as e:
        load_results(tmp_path / "other.json")
    assert e.value.code == "RESULTS_NOT_OURS"
    write_json(tmp_path / "bad.json", {"output_schema": "nova-qwen-explanations", "results": [{"alert_id": "x"}],
                                       "source": {}, "selection": {}, "model": {}})
    with pytest.raises(ResultsFileError) as e:
        load_results(tmp_path / "bad.json")
    assert e.value.code == "RESULTS_MALFORMED"
    r = cli("verify", tmp_path / "missing.json", "--export", DT_FIXTURE)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESULTS_MISSING"


def test_output_is_input_detected_through_other_spellings(tmp_path):
    export = tmp_path / "a.json"
    export.write_text("{}")
    (tmp_path / "sub").mkdir()
    for spelled in (tmp_path / "sub" / ".." / "a.json", tmp_path / "link.json"):
        if spelled.name == "link.json":
            if os.name != "posix":
                continue
            spelled.symlink_to(export)
        with pytest.raises(OutputError) as e:
            check_output_path(spelled, [export], overwrite=True, resume=False)
        assert e.value.code == "OUTPUT_IS_INPUT"


# =========================================================================== verify
@pytest.fixture
def results(fake_ollama, settings, tmp_path):
    fake_ollama.chat_actions.append(("json", 500, {"error": "x"}))
    s = make_settings(fake_ollama.url, transport_retries=0)
    return run(load_export(DT_FIXTURE), s, tmp_path / "r.json", limit=None), load_export(DT_FIXTURE)


def _hide_failed(d):
    d["results"] = [r for r in d["results"] if r["status"] != "failed"]
    d["summary"].update(failed=0, processed=len(d["results"]))
    d["selection"]["selected"] = len(d["results"])


TAMPER = {
    "detector": (lambda d: d["results"][1].update(detector="random_forest"), "detector differs"),
    "record_position": (lambda d: d["results"][1].update(record_position=3), "record_position differs"),
    "dataset": (lambda d: d["results"][1].update(dataset="other"), "dataset differs"),
    "schema_version": (lambda d: d["results"][1].update(schema_version="1.0"), "schema_version differs"),
    "dq_notes_removed": (lambda d: d["results"][2].update(data_quality_notes=[]), "data_quality_notes differs"),
    "prediction_status": (lambda d: d["results"][1].update(prediction_status="confirmed attack"), "prediction_status"),
    "source_alert_count": (lambda d: d["source"].update(alert_count=4), "source.alert_count"),
    "source_size": (lambda d: d["source"].update(size_bytes=1), "source.size_bytes"),
    "source_run_id": (lambda d: d["source"].update(run_id="x"), "source.run_id"),
    "source_dataset": (lambda d: d["source"].update(dataset="x"), "source.dataset"),
    "source_detector": (lambda d: d["source"].update(detector="random_forest"), "source.detector"),
    "source_schema_version": (lambda d: d["source"].update(schema_version="1.0"), "source.schema_version"),
    "order": (lambda d: d["results"].reverse(), "in order"),
    "processed": (lambda d: d["summary"].update(processed=9), "summary.processed"),
    "schema_version_doc": (lambda d: d.update(output_schema_version="9.9"), "output_schema_version"),
    "run_status": (lambda d: d.update(run_status="done"), "invalid run_status"),
    "hide_failed_alert": (_hide_failed, "do not cover exactly"),
    "edited_but_valid_answer": (lambda d: d["results"][1]["response"].update(
        explanation="These values could fit the prediction but cannot establish anything further here."),
        "not the model's recorded output"),
    "model_not_contacted": (lambda d: d["model"].update(contacted=False), "not contacted"),
    "digest_unverified": (lambda d: d["model"].update(digest_verified=False), "verified expected digest"),
    "failed_without_code": (lambda d: d["results"][0].update(error=None), "failed without an error code"),
    "rejected_without_errors": (lambda d: d["results"][0].update(status="rejected", error=None,
                                                                 validation={"errors": []}), "rejected without"),
    "malformed": (lambda d: d["results"].__setitem__(1, "garbage"), "malformed results file"),
}


@pytest.mark.parametrize("name", list(TAMPER))
def test_verify_detects(results, name):
    doc, export = results
    assert verify_results(doc, export)["passed"]
    doc = copy.deepcopy(doc)
    change, fragment = TAMPER[name]
    change(doc)
    report = verify_results(doc, export)
    assert not report["passed"] and any(fragment in p for p in report["problems"]), report["problems"]


def test_verify_flags_contact_for_empty_export(fake_ollama, settings, tmp_path):
    doc = run(load_export(DT_EMPTY), settings, tmp_path / "r.json")
    doc["model"]["contacted"] = True
    assert any("empty export" in p for p in verify_results(doc, load_export(DT_EMPTY))["problems"])


def test_verify_incomplete_needs_explicit_flag(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    with pytest.raises(KeyboardInterrupt):
        run(load_export(DT_FIXTURE), settings, out, limit=None,
            progress=lambda i, n, r: (_ for _ in ()).throw(KeyboardInterrupt))
    assert cli("verify", out, "--export", DT_FIXTURE).returncode == 1
    assert cli("verify", out, "--export", DT_FIXTURE, "--allow-incomplete").returncode == 0


# =========================================================================== correlate / worksheet guards
def test_correlate_warns_when_no_rows_are_shared(tmp_path):
    rf = copy.deepcopy(load_json(RF_FIXTURE))
    rf["alerts"] = [a for a in rf["alerts"] if a["record_id"] == "val-000003"]  # RF-only row
    dt = copy.deepcopy(load_json(DT_FIXTURE))
    dt["alerts"] = [a for a in dt["alerts"] if a["record_id"] == "val-000005"]  # DT-only row
    report = correlate(load_export(write_json(tmp_path / "dt.json", dt)), load_export(write_json(tmp_path / "rf.json", rf)))
    assert report["warnings"] and report["compatibility"]["shared_records_checked"] == 0


def test_correlate_out_guards(tmp_path):
    dt_copy, rf_copy = tmp_path / "dt.json", tmp_path / "rf.json"  # never point --out at the real fixtures
    dt_copy.write_bytes(DT_FIXTURE.read_bytes())
    rf_copy.write_bytes(RF_FIXTURE.read_bytes())
    base = ["correlate", "--dt-export", dt_copy, "--rf-export", rf_copy, "--out"]
    r = cli(*base, dt_copy)
    assert dt_copy.read_bytes() == DT_FIXTURE.read_bytes()
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_IS_INPUT"
    other = tmp_path / "notes.txt"
    other.write_text("my notes")
    r = cli(*base, other)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_NOT_OURS"
    assert other.read_text() == "my notes"
    assert cli(*base, tmp_path / "c.json").returncode == 0 and cli(*base, tmp_path / "c.json").returncode == 0


def test_worksheet_never_overwrites_inputs_or_review_work(fake_ollama, settings, tmp_path):
    res = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, res, limit=2)
    export_copy = tmp_path / "export.json"
    export_copy.write_bytes(DT_FIXTURE.read_bytes())
    r = cli("worksheet", res, "--out", res)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_IS_INPUT"
    r = cli("worksheet", res, "--out", export_copy)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_NOT_OURS"
    assert export_copy.read_bytes() == DT_FIXTURE.read_bytes()
    ws = tmp_path / "ws.csv"
    assert cli("worksheet", res, "--out", ws).returncode == 0
    scored = ws.read_text().replace(",,,,,,,", ",2,1,1,1,1,2,paul,ok", 1)
    ws.write_text(scored)
    r = cli("worksheet", res, "--out", ws)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_EXISTS" and ws.read_text() == scored
    assert cli("worksheet", res, "--out", ws, "--overwrite").returncode == 0


@pytest.mark.skipif(os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs non-root POSIX")
def test_worksheet_write_failure_is_exit_4(fake_ollama, settings, tmp_path):
    res = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, res, limit=1)
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    try:
        r = cli("worksheet", res, "--out", d / "ws.csv")
        assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_WRITE_FAILED"
    finally:
        d.chmod(0o755)


def test_worksheet_neutralises_formulas_and_tolerates_odd_values(fake_ollama, settings, tmp_path):
    def formula(body):
        a = valid_answer(evidence_from_request(body))
        a["explanation"] = "=HYPERLINK(\"http://x\") these values could fit the prediction but cannot establish it."
        return ("json", 200, chat_body(a))
    fake_ollama.chat_actions.append(formula)
    doc = run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, max_attempts=1), tmp_path / "r.json", limit=1)
    doc["results"].append(dict(copy.deepcopy(doc["results"][0]), status="rejected", response=None,
                               rejected_response={"observations": [1, None], "limitations": 5}))
    text = to_csv(rows_from_results([doc]))
    assert ",'=HYPERLINK" in text or "\"'=HYPERLINK" in text


# =========================================================================== CLI exit codes
def test_exit_1_when_an_alert_failed(fake_ollama, tmp_path):
    fake_ollama.chat_actions.append(("json", 500, {"error": "x"}))
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "r.json", "--limit", "2", "--retries", "0", "--quiet",
            env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 1 and json.loads(r.stdout)["summary"]["failed"] == 1


@pytest.mark.parametrize("content, done", [('{"ready": true}', "length"), ("not json", "stop"), ("[1]", "stop")])
def test_check_ollama_inference_failures_exit_3(fake_ollama, content, done):
    fake_ollama.chat_actions.append(("json", 200, {"model": MODEL, "message": {"content": content}, "done": True,
                                                   "done_reason": done}))
    r = cli("check-ollama", "--inference", env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 3 and json.loads(r.stdout)["inference_test"]["ok"] is False


@pytest.mark.skipif(os.name != "posix", reason="SIGINT delivery is POSIX-specific")
def test_ctrl_c_exits_130_and_keeps_partial_results(fake_ollama, tmp_path):
    fake_ollama.default_chat = lambda b: ("sleep", 0.6, ("json", 200, chat_body(valid_answer(evidence_from_request(b)))))
    out = tmp_path / "r.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", NOVA_OLLAMA_URL=fake_ollama.url)
    proc = subprocess.Popen([sys.executable, "-m", "src.nova_qwen", "explain", str(DT_FIXTURE), "--out", str(out),
                             "--all", "--quiet"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 20
    while time.time() < deadline and not (out.exists() and load_json(out)["results"]):
        time.sleep(0.1)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=30)
    assert proc.returncode == 130
    assert load_json(out)["run_status"] == "interrupted"


def test_budget_counts_the_schema():
    small = request_bytes([{"role": "user", "content": "a"}], response_schema(["Flow Duration"]))
    large = request_bytes([{"role": "user", "content": "a"}],
                          response_schema(["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]))
    assert large > small
