"""The full consumer pipeline against FakeOllama: export -> validation -> normalisation -> request ->
answer -> validation -> results file. Checks traceability, empty input, re-asks, rejections,
transport failures, budget refusal, interruption and resume."""
import copy
import json

import pytest

from qwen_test_support import (DIGEST, DT_EMPTY, DT_FIXTURE, RF_EMPTY, RF_FIXTURE, chat_body, evidence_from_request,
                      load_json, make_settings, valid_answer, write_json)
from src.nova_qwen.alerts import load_export
from src.nova_qwen.consumer import run, select_alerts
from src.nova_qwen.errors import (ModelDigestMismatch, NovaQwenError, OllamaUnavailable, ResumeMismatch)
from src.nova_qwen.output import OUTPUT_SCHEMA, OUTPUT_SCHEMA_VERSION
from src.nova_qwen.verify import verify_results


def assert_traceable(doc, export):
    """Every identifier, prediction and raw evidence value is exactly the source alert's."""
    src = {a.alert_id: a for a in export.alerts}
    for r in doc["results"]:
        a = src[r["alert_id"]]
        assert (r["record_id"], r["record_position"], r["predicted_class"], r["evidence"]) == \
            (a.record_id, a.record_position, a.predicted_class, a.evidence)
        assert (r["run_id"], r["dataset"], r["detector"], r["schema_version"]) == \
            (export.run_id, export.dataset, export.detector, "1.1")
    assert verify_results(doc, export)["passed"]


@pytest.mark.parametrize("fixture", [DT_FIXTURE, RF_FIXTURE], ids=["decision_tree", "random_forest"])
def test_full_pipeline_for_each_classifier(fake_ollama, settings, tmp_path, fixture):
    export = load_export(fixture)
    out = tmp_path / "results.json"
    doc = run(export, settings, out, limit=None)
    assert load_json(out) == doc
    assert doc["output_schema"] == OUTPUT_SCHEMA and doc["output_schema_version"] == OUTPUT_SCHEMA_VERSION
    assert doc["run_status"] == "complete" and doc["summary"]["explained"] == 5 and doc["summary"]["pending"] == 0
    assert doc["model"]["contacted"] and doc["model"]["digest"] == DIGEST and doc["model"]["digest_verified"]
    assert doc["source"]["sha256"] == export.sha256 and doc["source"]["detector"] == export.detector
    assert [r["alert_id"] for r in doc["results"]] == [a.alert_id for a in export.alerts]
    assert len(fake_ollama.chat_requests()) == 5  # one request per alert
    assert_traceable(doc, export)
    for r, body in zip(doc["results"], fake_ollama.chat_requests()):
        prompt = evidence_from_request(body)
        assert prompt["predicted_class"] == r["predicted_class"]
        assert [e["value"] for e in prompt["evidence"]] == [e["display"] for e in r["evidence_normalized"]]
        assert r["alert_id"] not in json.dumps(body) and r["run_id"] not in json.dumps(body)


def test_flow_duration_reaches_the_model_in_seconds(fake_ollama, settings, tmp_path):
    run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", alert_ids=["dt-000002"])
    prompt = evidence_from_request(fake_ollama.chat_requests()[0])
    assert prompt["evidence"][0] == {"feature": "Flow Duration", "value": "101.041626", "unit": "seconds"}


@pytest.mark.parametrize("fixture", [DT_EMPTY, RF_EMPTY], ids=["decision_tree", "random_forest"])
def test_empty_export_never_contacts_ollama(fake_ollama, settings, tmp_path, fixture):
    doc = run(load_export(fixture), settings, tmp_path / "r.json")
    assert fake_ollama.requests == []  # not even /api/version
    assert doc["results"] == [] and doc["summary"]["selected"] == 0 and doc["model"]["contacted"] is False
    assert doc["run_status"] == "complete"
    assert verify_results(doc, load_export(fixture))["passed"]


def test_selection_limit_offset_and_ids():
    export = load_export(DT_FIXTURE)
    assert [a.alert_id for a in select_alerts(export, limit=2)[0]] == ["dt-000002", "dt-000004"]
    assert [a.alert_id for a in select_alerts(export, limit=2, offset=3)[0]] == ["dt-000006", "dt-000008"]
    assert len(select_alerts(export, limit=None)[0]) == 5
    chosen, sel = select_alerts(export, limit=10, alert_ids=["dt-000008", "dt-000002", "dt-000008"])
    assert [a.alert_id for a in chosen] == ["dt-000008", "dt-000002"] and sel["mode"] == "alert_ids"
    with pytest.raises(NovaQwenError) as e:
        select_alerts(export, limit=1, alert_ids=["dt-000003"])  # row 3 was Normal for DT
    assert e.value.code == "ALERT_ID_NOT_FOUND"
    for bad in ({"limit": 0}, {"limit": 1, "offset": -1}):
        with pytest.raises(NovaQwenError) as e:
            select_alerts(export, **bad)
        assert e.value.code == "BAD_SELECTION"


def test_batch_limit_bounds_model_calls(fake_ollama, settings, tmp_path):
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", limit=2)
    assert len(fake_ollama.chat_requests()) == 2 and doc["summary"]["selected"] == 2


def invalid_then(valid_after: bool):
    def first(body):
        bad = valid_answer(evidence_from_request(body))
        bad["explanation"] = "The flow lasted 101041626.0 seconds (approximately 3.18 years), confirming the attack."
        return ("json", 200, chat_body(bad))
    return [first] + ([] if valid_after else [first])


def test_invalid_answer_is_re_asked_once_with_feedback(fake_ollama, settings, tmp_path):
    fake_ollama.chat_actions += invalid_then(valid_after=True)
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", alert_ids=["dt-000002"])
    r = doc["results"][0]
    assert r["status"] == "explained" and len(r["attempts"]) == 2
    assert set(r["attempts"][0]["validation_errors"]) == {"TIME_UNIT_MISMATCH", "CONFIRMATION_CLAIM"}
    second = fake_ollama.chat_requests()[1]["messages"]
    assert second[-2]["role"] == "assistant" and "TIME_UNIT_MISMATCH" in second[-1]["content"]
    assert "3.18 years" not in json.dumps(r["response"])


def test_answer_still_invalid_is_rejected_never_accepted(fake_ollama, settings, tmp_path):
    fake_ollama.chat_actions += invalid_then(valid_after=False)
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", alert_ids=["dt-000002"])
    r = doc["results"][0]
    assert r["status"] == "rejected" and r["response"] is None and r["validation"]["passed"] is False
    assert "3.18 years" in r["rejected_response"]["explanation"]  # kept for audit only
    assert doc["summary"]["rejected"] == 1 and verify_results(doc, load_export(DT_FIXTURE))["passed"]


@pytest.mark.parametrize("action, code", [
    (lambda b: ("json", 200, chat_body('{"observations": ["Flow Dur', done_reason="length")), "RESPONSE_NOT_JSON"),
    (lambda b: ("json", 200, chat_body(valid_answer(evidence_from_request(b)), done_reason="length")),
     "RESPONSE_TRUNCATED"),
    (lambda b: ("json", 200, chat_body("Sure! Here is the explanation.")), "RESPONSE_NOT_JSON"),
    (lambda b: ("json", 200, chat_body(dict(valid_answer(evidence_from_request(b)), priority="high"))),
     "PRIORITY_NOT_NULL"),
    (lambda b: ("json", 200, chat_body(dict(valid_answer(evidence_from_request(b)), alert_id="dt-999999"))),
     "RESPONSE_KEYS"),
])
def test_incomplete_or_invalid_outputs_are_rejected(fake_ollama, tmp_path, action, code):
    fake_ollama.chat_actions.append(action)
    s = make_settings(fake_ollama.url, max_attempts=1)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", alert_ids=["dt-000002"])
    r = doc["results"][0]
    assert r["status"] == "rejected" and code in r["attempts"][0]["validation_errors"]
    assert r["response"] is None


def test_model_cannot_change_identifiers_or_prediction(fake_ollama, settings, tmp_path):
    def hostile(body):
        a = valid_answer(evidence_from_request(body))
        a["explanation"] = "These values could fit the prediction but cannot establish it; DDoS is also possible."
        return ("json", 200, chat_body(a))
    fake_ollama.default_chat = hostile
    export = load_export(DT_FIXTURE)
    doc = run(export, settings, tmp_path / "r.json", limit=None)
    assert_traceable(doc, export)
    assert all(r["status"] == "explained" for r in doc["results"])
    assert any(w["code"] == "OTHER_CLASS_MENTIONED" for r in doc["results"] for w in r["validation"]["warnings"])


def test_transport_failure_on_one_alert_is_recorded_and_the_run_continues(fake_ollama, tmp_path):
    fake_ollama.chat_actions += [("json", 500, {"error": "runner stopped"})] * 3
    s = make_settings(fake_ollama.url, transport_retries=2)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", limit=2)
    first, second = doc["results"]
    assert first["status"] == "failed" and first["error"]["code"] == "OLLAMA_HTTP_ERROR" and first["response"] is None
    assert second["status"] == "explained" and doc["summary"]["failed"] == 1


def test_timeout_is_recorded_as_failure(fake_ollama, tmp_path):
    fake_ollama.chat_actions += [("sleep", 1.0, ("json", 200, chat_body("{}")))]
    s = make_settings(fake_ollama.url, timeout_seconds=0.3, transport_retries=0)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", alert_ids=["dt-000002"])
    assert doc["results"][0]["status"] == "failed" and doc["results"][0]["error"]["code"] == "OLLAMA_TIMEOUT"


def test_unavailable_service_fails_before_any_alert(unused_port, tmp_path):
    s = make_settings(f"http://127.0.0.1:{unused_port}", transport_retries=0)
    out = tmp_path / "r.json"
    with pytest.raises(OllamaUnavailable):
        run(load_export(DT_FIXTURE), s, out, limit=1)
    assert not out.exists()


def test_wrong_model_digest_fails_before_any_alert(fake_ollama, settings, tmp_path):
    fake_ollama.model_entry = dict(fake_ollama.model_entry, digest="2" * 64)
    with pytest.raises(ModelDigestMismatch):
        run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", limit=1)
    assert fake_ollama.chat_requests() == []


def test_prompt_budget_refusal_never_sends(fake_ollama, tmp_path):
    s = make_settings(fake_ollama.url, num_ctx=1024, num_predict=512)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", alert_ids=["dt-000002"])
    r = doc["results"][0]
    assert r["status"] == "failed" and r["error"]["code"] == "PROMPT_BUDGET_EXCEEDED"
    assert fake_ollama.chat_requests() == []


def test_thinking_text_is_flagged(fake_ollama, settings, tmp_path):
    fake_ollama.chat_actions.append(lambda b: ("json", 200, chat_body(valid_answer(evidence_from_request(b)),
                                                                      thinking="let me think")))
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", alert_ids=["dt-000002"])
    assert {w["code"] for w in doc["results"][0]["validation"]["warnings"]} == {"THINKING_RETURNED"}


def test_data_quality_alert_end_to_end(fake_ollama, settings, tmp_path):
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", alert_ids=["dt-000005"])
    r = doc["results"][0]
    assert r["evidence"]["Flow Duration"] == -1.0 and len(r["data_quality_notes"]) == 1
    assert "negative" in evidence_from_request(fake_ollama.chat_requests()[0])["data_quality_notes"][0]


def test_interrupt_leaves_a_valid_partial_file_and_resume_finishes(fake_ollama, settings, tmp_path):
    export = load_export(DT_FIXTURE)
    out = tmp_path / "r.json"

    def stop_after_two(i, n, r):
        if i == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(export, settings, out, limit=None, progress=stop_after_two)
    partial = load_json(out)
    assert partial["run_status"] == "interrupted" and len(partial["results"]) == 2
    assert verify_results(partial, export, allow_incomplete=True)["passed"]
    assert not verify_results(partial, export)["passed"]  # an unfinished run must not pass by default
    before = len(fake_ollama.chat_requests())
    doc = run(export, settings, out, limit=None, resume=True)
    assert doc["run_status"] == "complete" and len(doc["results"]) == 5
    assert len(fake_ollama.chat_requests()) - before == 3  # only the 3 missing alerts were sent
    assert doc["resumed_from"]["reused_results"] == 2
    assert verify_results(doc, export)["passed"]


def test_resume_reprocesses_failed_alerts(fake_ollama, tmp_path):
    export = load_export(DT_FIXTURE)
    out = tmp_path / "r.json"
    fake_ollama.chat_actions += [("json", 500, {"error": "x"})]
    run(export, make_settings(fake_ollama.url, transport_retries=0), out, limit=2)
    doc = run(export, make_settings(fake_ollama.url, transport_retries=0), out, limit=2, resume=True)
    assert [r["status"] for r in doc["results"]] == ["explained", "explained"]
    assert doc["resumed_from"]["reused_results"] == 1


def test_resume_refuses_a_different_export(fake_ollama, settings, work):
    out = work / "r.json"
    run(load_export(work / "decision_tree_alerts.json"), settings, out, limit=1)
    doc = load_json(work / "decision_tree_alerts.json")
    doc["alerts"] = doc["alerts"][:3]
    write_json(work / "changed.json", doc)
    with pytest.raises(ResumeMismatch):
        run(load_export(work / "changed.json"), settings, out, limit=1, resume=True)


def test_resume_refuses_different_settings(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    export = load_export(DT_FIXTURE)
    run(export, settings, out, limit=1)
    with pytest.raises(ResumeMismatch):
        run(export, make_settings(fake_ollama.url, seed=7), out, limit=1, resume=True)


def test_checkpoint_after_every_alert(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    seen = []
    run(load_export(DT_FIXTURE), settings, out, limit=3,
        progress=lambda i, n, r: seen.append((load_json(out)["summary"]["processed"], load_json(out)["run_status"])))
    assert seen == [(1, "in_progress"), (2, "in_progress"), (3, "in_progress")]
    assert load_json(out)["run_status"] == "complete"


def test_conflicting_classifier_results_stay_separate(fake_ollama, settings, tmp_path):
    dt = run(load_export(DT_FIXTURE), settings, tmp_path / "dt.json", alert_ids=["dt-000004"])
    rf = run(load_export(RF_FIXTURE), settings, tmp_path / "rf.json", alert_ids=["rf-000004"])
    d, r = dt["results"][0], rf["results"][0]
    assert d["record_id"] == r["record_id"] == "val-000004"
    assert (d["detector"], d["predicted_class"]) == ("decision_tree", "DDoS")
    assert (r["detector"], r["predicted_class"]) == ("random_forest", "DoS")
    assert d["evidence"] == r["evidence"]


def test_a_model_answer_that_cannot_be_stored_is_rejected_not_a_crash(fake_ollama, settings, tmp_path):
    """A lone surrogate escape is valid JSON, but UTF-8 cannot hold it, so writing it would fail on every resume too."""
    export = load_export(DT_FIXTURE)
    fake_ollama.default_chat = lambda body: ("json", 200, chat_body('{"explanation": "cut off \\ud83d"}'))
    doc = run(export, settings, tmp_path / "r.json", limit=1)
    assert doc["results"][0]["status"] == "rejected"
    assert load_json(tmp_path / "r.json")["results"][0]["status"] == "rejected"


def test_check_only_refuses_or_returns_without_writing_or_sending(fake_ollama, settings, tmp_path):
    export = load_export(DT_FIXTURE)
    out = tmp_path / "r.json"
    run(export, settings, out, check_only=True, resume=True, limit=1)  # no earlier file: nothing to refuse
    assert not out.exists() and fake_ollama.chat_requests() == []
    run(export, settings, out, limit=1)
    before = out.read_bytes()
    fake_ollama.requests.clear()
    run(export, settings, out, check_only=True, resume=True, limit=1)
    assert out.read_bytes() == before and fake_ollama.chat_requests() == []
    with pytest.raises(ResumeMismatch):
        run(export, settings, out, check_only=True, resume=True, limit=2)
    assert out.read_bytes() == before and fake_ollama.chat_requests() == []


def test_check_only_on_an_empty_export_writes_nothing(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    run(load_export(DT_EMPTY), settings, out, check_only=True, resume=True)
    assert not out.exists() and fake_ollama.requests == []
