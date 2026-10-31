"""Regression tests for the second independent review round (fix confirmation, mutation survivors and
new-defect hunt). Each test pins one verified finding or one surviving mutant."""
import copy
import http.client
import json
import os
import time

import pytest

from qwen_test_support import (DIGEST, DT_FIXTURE, MODEL, RF_FIXTURE, chat_body, evidence_from_request, load_json,
                               make_settings, valid_answer, write_json)
from src.nova_qwen.alerts import load_export
from src.nova_qwen.config import QwenSettings
from src.nova_qwen.consumer import FINGERPRINT_SETTINGS, RunAborted, run
from src.nova_qwen.correlate import correlate
from src.nova_qwen.errors import ConfigError, ModelNotAvailable, OllamaProtocolError, ResumeMismatch
from src.nova_qwen.features import data_quality_notes, normalize_evidence, normalize_value
from src.nova_qwen.ollama import OllamaClient
from src.nova_qwen.prompt import response_schema
from src.nova_qwen.response import ALLOWED_CONSTANTS, validate_response
from src.nova_qwen.verify import verify_results
from src.nova_qwen.worksheet import rows_from_results, to_csv

EVIDENCE = {"Flow Duration": 101041626.0, "Total Fwd Packets": 11.0, "Flow Bytes/s": 123.7509776}
NORM = normalize_evidence(EVIDENCE)
BASE = {
    "observations": ["Flow Duration is 101.041626 seconds.", "Total Fwd Packets is 11 packets.",
                     "Flow Bytes/s is 123.7509776 bytes per second."],
    "explanation": "A long flow with few packets and a low byte rate could be consistent with a slow DoS pattern, "
                   "but these values cannot establish that an attack occurred.",
    "limitations": ["Only three flow features are available.", "No packet contents or ground truth are available."],
    "evidence_used": ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"], "priority": None, "abstain": False,
}


def errs(**changes):
    a = copy.deepcopy(BASE)
    a.update(changes)
    return sorted({e["code"] for e in validate_response(a, NORM, data_quality_notes(NORM), "DoS").errors})


def expl(sentence):
    return errs(explanation=BASE["explanation"] + " " + sentence)


# =========================================================================== validator: certainty & assertions
@pytest.mark.parametrize("sentence", [
    "There is no doubt that the flow is a DoS attack.", "There is no question that this traffic is a DoS attack.",
    "Without a doubt this is malicious.", "These features are enough to confirm a DoS attack.",
    "The values are sufficient to prove an attack.", "These values clearly indicate a DoS attack.",
    "It's a DoS attack.", "This is undeniably an attack.", "The evidence establishes a DoS attack.",
    "Payloads are not available, this confirms a DoS attack.",
])
def test_certainty_and_assertion_claims_rejected(sentence):
    e = expl(sentence)
    assert "CONFIRMATION_CLAIM" in e or "ASSERTIVE_CLASSIFICATION" in e, e


@pytest.mark.parametrize("sentence", [
    "This flow was classified as DoS by the Decision Tree.", "The traffic was flagged as a DoS attack by the model.",
    "The predicted class for this flow is DoS.", "This is the predicted DoS class.",
    "It is a possible DoS attack pattern.", "This is consistent with a DoS attack, but unverified.",
    "It was labelled as a port scan by the classifier.",
])
def test_normal_prediction_phrasing_accepted(sentence):
    assert expl(sentence) == []


@pytest.mark.parametrize("sentence", [   # real Qwen outputs (end-to-end run) that a short window wrongly rejected
    "The evidence cannot confirm the specific attack type, identify the source or destination IPs, verify protocol "
    "details, or establish attacker intent.",
    "However, it does not indicate packet contents, specific TCP flags, or attacker intent.",
    "No data quality issues were noted, but the limited feature set restricts definitive conclusions.",
    "It is not clear that this is a DoS attack.",
    "There is no evidence that the flow is malicious.",
])
def test_negation_governs_its_whole_clause(sentence):
    assert expl(sentence) == []


@pytest.mark.parametrize("sentence", [
    "The prediction is not verified and the values confirm an attack.",
    "No further checks are needed because the flow is a DoS attack.",
    "It is not surprising that this is a DoS attack.",
    "Packet contents are unavailable, so the values prove an attack.",
])
def test_new_clauses_end_the_negation(sentence):
    e = expl(sentence)
    assert "CONFIRMATION_CLAIM" in e or "ASSERTIVE_CLASSIFICATION" in e, e


def test_hedge_must_be_in_the_same_clause():
    assert expl("Further work would be needed to confirm this.") == []
    assert "CONFIRMATION_CLAIM" in expl("Analysts may review it, but this confirms the attack.")


def test_semicolon_is_a_clause_boundary():
    assert "CONFIRMATION_CLAIM" in expl("Payloads are not available; the values confirm the attack.")


def test_assertion_hedge_only_from_its_clause():
    assert "ASSERTIVE_CLASSIFICATION" in expl("The model output is unverified; this is a DoS attack.")


@pytest.mark.parametrize("sentence", [
    "No packet contents, protocol details (TCP/UDP), or confidence scores are provided.",
    "The model's confidence and the protocol (TCP or UDP) are not included.",
    "No statistics over minutes or hours are available.",
])
def test_limitations_may_name_missing_information(sentence):
    assert errs(limitations=[sentence]) == []


def test_strict_fields_reject_the_same_words():
    assert "INVENTED_NETWORK_DETAIL" in expl("The flow is consistent with a TCP SYN flood pattern.")
    assert "CONFIDENCE_CLAIM" in expl("The model's confidence is high.")
    assert "TIME_UNIT_MISMATCH" in expl("The connection stayed open for hours.")


def test_data_quality_claim_negation_scope():
    assert expl("There is no indication of corrupted data.") == []
    assert "UNSUPPORTED_DATA_QUALITY_CLAIM" in expl("Packets are not shown, so the data looks corrupted.")


def test_big_time_word_negation_scope():
    assert expl("This is not a statement about hours or days.") == []
    assert "TIME_UNIT_MISMATCH" in expl("Payloads are not shown, and it lasted for hours.")


# =========================================================================== validator: time and numbers
@pytest.mark.parametrize("text", ["Flow Duration was 3 yrs.", "Flow Duration was 3yrs.", "It lasted three yrs.",
                                  "It lasted 2 wks.", "It lasted about three seconds.", "It lasted five minutes.",
                                  "Flow Duration was 1 minute 11 seconds.", "Flow Duration was 2 min 41 s."])
def test_wrong_durations_rejected(text):
    assert "TIME_UNIT_MISMATCH" in errs(observations=[text])


@pytest.mark.parametrize("text", ["Flow Duration was one hundred and one seconds.",
                                  "Flow Duration was 101 s.", "Flow Duration was 0.02 minutes short of 1.7 minutes."
                                  if False else "Flow Duration was 101.04 s.",
                                  "Flow Duration was 101.041626 seconds, i.e. 101041626 microseconds."])
def test_correct_durations_accepted(text):
    assert errs(observations=[text]) == []


def test_spelled_seconds_checked_against_short_flows():
    norm = normalize_evidence({"Flow Duration": 5_000_000.0})
    a = copy.deepcopy(BASE)
    a.update(observations=["Flow Duration was five seconds."], evidence_used=["Flow Duration"])
    assert validate_response(a, norm, [], "DoS").passed
    a["observations"] = ["Flow Duration was six seconds."]
    assert [e["code"] for e in validate_response(a, norm, [], "DoS").errors] == ["TIME_UNIT_MISMATCH"]


def test_bare_s_with_duration_is_time_checked():
    assert errs(observations=["Flow Duration was 101 s."]) == []
    assert "TIME_UNIT_MISMATCH" in errs(observations=["Flow Duration was 5 s."])


@pytest.mark.parametrize("text, ok", [
    ("Flow Bytes/s is high for a flow of 11 packets.", True),          # informal second feature: not attached
    ("Total Fwd Packets was 11, with Flow Bytes/s of 123.75.", True),
    ("Total Fwd Packets was 123.75.", False),                        # attached to the wrong feature
    ("Flow Bytes/s of 11 bytes per second.", False),
    ("Flow Duration (in seconds) was 101041626.", False),
])
def test_attached_numbers_must_belong_to_their_feature(text, ok):
    assert (errs(observations=[text]) == []) == ok


def test_allowed_constants_are_exactly_the_dataset_year():
    assert ALLOWED_CONSTANTS == (2017.0,)
    assert "NUMBER_NOT_IN_EVIDENCE" in errs(limitations=["CICIDS 2018 is not used."])


@pytest.mark.parametrize("content", ['{"observations": [1e999]}', '{"a": -1e400}'])
def test_overflowing_numbers_in_answers_rejected(fake_ollama, tmp_path, content):
    fake_ollama.chat_actions.append(("json", 200, chat_body(content)))
    doc = run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, max_attempts=1), tmp_path / "r.json", limit=1)
    assert doc["results"][0]["status"] == "rejected" and doc["run_status"] == "complete"


def test_overflowing_number_in_ollama_metadata_is_a_protocol_error(fake_ollama):
    fake_ollama.chat_actions.append(("raw", 200, b'{"model": "%s", "message": {"content": "{}"}, "total_duration": 1e999}'
                                     % MODEL.encode(), "application/json"))
    with pytest.raises(OllamaProtocolError):
        OllamaClient(make_settings(fake_ollama.url, transport_retries=0)).chat([{"role": "user", "content": "u"}],
                                                                            schema=None)


# =========================================================================== resume
def test_fingerprint_keys_are_pinned_literally():
    assert FINGERPRINT_SETTINGS == ("model", "think", "temperature", "seed", "num_ctx", "num_predict", "max_attempts")


@pytest.mark.parametrize("key, value", [("model", "qwen3.5:4b"), ("think", True), ("temperature", 0.7), ("seed", 7),
                                        ("num_ctx", 4096), ("num_predict", 256), ("max_attempts", 1)])
def test_each_setting_blocks_resume(fake_ollama, settings, tmp_path, key, value):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=1)
    doc = load_json(out)
    doc["model"]["settings"][key] = value
    write_json(out, doc)
    with pytest.raises(ResumeMismatch):
        run(load_export(DT_FIXTURE), settings, out, limit=1, resume=True)


def test_abstained_only_file_still_checks_the_fingerprint(fake_ollama, settings, tmp_path):
    fake_ollama.default_chat = lambda b: ("json", 200, chat_body(valid_answer(evidence_from_request(b), abstain=True)))
    out = tmp_path / "r.json"
    assert run(load_export(DT_FIXTURE), settings, out, limit=1)["summary"]["abstained"] == 1
    with pytest.raises(ResumeMismatch):
        run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, seed=9), out, limit=1, resume=True)


def test_interrupted_resume_keeps_unreached_earlier_results(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    export = load_export(DT_FIXTURE)
    run(export, settings, out, limit=None)
    doc = load_json(out)
    doc["results"][0]["status"], doc["results"][0]["response"] = "failed", None  # force reprocessing of alert 1
    doc["results"][0]["error"] = {"code": "OLLAMA_TIMEOUT", "message": "x"}
    doc["summary"].update(explained=4, failed=1)
    write_json(out, doc)
    with pytest.raises(KeyboardInterrupt):
        run(export, settings, out, limit=None, resume=True,
            progress=lambda i, n, r: (_ for _ in ()).throw(KeyboardInterrupt) if i == 1 else 0)
    after = load_json(out)
    assert after["run_status"] == "interrupted" and len(after["results"]) == 5  # nothing lost
    assert verify_results(after, export, allow_incomplete=True)["passed"]


def test_resume_does_not_reuse_an_answer_that_differs_from_the_raw_output(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=1)
    doc = load_json(out)
    doc["results"][0]["response"]["explanation"] = ("These values could fit the prediction, but they cannot establish "
                                                    "it here.")
    write_json(out, doc)
    before = len(fake_ollama.chat_requests())
    new = run(load_export(DT_FIXTURE), settings, out, limit=1, resume=True)
    assert len(fake_ollama.chat_requests()) == before + 1 and new["resumed_from"]["reused_results"] == 0
    assert verify_results(new, load_export(DT_FIXTURE))["passed"]


def test_reuse_requires_unchanged_identity(fake_ollama, settings, tmp_path):
    out = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, out, limit=1)
    doc = load_json(out)
    doc["results"][0]["record_position"] = 9
    write_json(out, doc)
    new = run(load_export(DT_FIXTURE), settings, out, limit=1, resume=True)
    assert new["resumed_from"]["reused_results"] == 0 and new["results"][0]["record_position"] == 2


def test_cli_offset_counts_as_an_explicit_selection_on_resume(fake_ollama, tmp_path):
    from test_review_regressions import cli
    out = tmp_path / "r.json"
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    assert cli("explain", DT_FIXTURE, "--out", out, "--limit", "2", "--quiet", env_extra=env).returncode == 0
    r = cli("explain", DT_FIXTURE, "--out", out, "--resume", "--offset", "1", "--quiet", env_extra=env)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESUME_MISMATCH"


# =========================================================================== abort counter
def test_abort_counter_resets_after_a_success(fake_ollama, tmp_path):
    ok = lambda b: ("json", 200, chat_body(valid_answer(evidence_from_request(b))))  # noqa: E731
    bad = ("json", 500, {"error": "x"})
    fake_ollama.chat_actions += [bad, bad, ok, bad, bad]
    doc = run(load_export(DT_FIXTURE), make_settings(fake_ollama.url, transport_retries=0,
                                                     abort_after_consecutive_failures=3), tmp_path / "r.json", limit=None)
    assert doc["run_status"] == "complete" and doc["summary"]["failed"] == 4


def test_only_transport_failures_count_towards_abort(fake_ollama, tmp_path):
    s = make_settings(fake_ollama.url, num_ctx=1024, num_predict=512, abort_after_consecutive_failures=3)
    doc = run(load_export(DT_FIXTURE), s, tmp_path / "r.json", limit=None)  # every alert fails the budget check
    assert doc["run_status"] == "complete" and doc["summary"]["failed"] == 5


def test_abort_threshold_validated():
    with pytest.raises(ConfigError):
        QwenSettings(abort_after_consecutive_failures=0).validate()


# =========================================================================== verify
@pytest.fixture
def complete(fake_ollama, settings, tmp_path):
    return run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", limit=None), load_export(DT_FIXTURE)


def test_verify_checks_selected_count(complete):
    doc, export = complete
    doc = copy.deepcopy(doc)
    doc["selection"]["selected"] = 4
    assert any("selection.selected" in p for p in verify_results(doc, export)["problems"])


def test_verify_checks_done_reason_of_accepted_answers(complete):
    doc, export = complete
    doc = copy.deepcopy(doc)
    doc["results"][0]["attempts"][-1]["done_reason"] = "length"
    assert any("done_reason" in p for p in verify_results(doc, export)["problems"])


def test_verify_prefix_check_for_incomplete_runs(complete):
    doc, export = complete
    doc = copy.deepcopy(doc)
    doc["run_status"] = "interrupted"
    del doc["results"][1]
    doc["summary"].update(processed=4, explained=4)
    assert any("prefix" in p for p in verify_results(doc, export, allow_incomplete=True)["problems"])


def test_verify_accepts_long_genuine_answers(fake_ollama, settings, tmp_path):
    def long(body):
        a = valid_answer(evidence_from_request(body))
        a["limitations"] = [("Only three flow features are available, so packet contents and context are missing. "
                             * 4)[:390]] * 6
        return ("json", 200, chat_body(a))
    fake_ollama.chat_actions.append(long)
    doc = run(load_export(DT_FIXTURE), settings, tmp_path / "r.json", limit=1)
    assert doc["results"][0]["status"] == "explained" and verify_results(doc, load_export(DT_FIXTURE))["passed"]


# =========================================================================== worksheet / correlate / misc
@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t"])
def test_every_formula_prefix_is_neutralised(complete, prefix):
    doc, _ = complete
    doc = copy.deepcopy(doc)
    doc["results"][0]["response"]["explanation"] = prefix + "SUM(1)"
    rows = rows_from_results([doc])
    assert rows[0]["explanation"] == "'" + prefix + "SUM(1)"


def test_worksheet_shows_rejected_answers(complete):
    doc, _ = complete
    doc = copy.deepcopy(doc)
    doc["results"][0].update(status="rejected", response=None,
                             rejected_response={"observations": ["x"], "explanation": "REJECTED TEXT", "limitations": []})
    assert "REJECTED TEXT" in to_csv(rows_from_results([doc]))


def test_worksheet_marker_must_be_in_the_header_line(fake_ollama, settings, tmp_path):
    from test_review_regressions import cli
    res = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, res, limit=1)
    fake = tmp_path / "notes.csv"
    fake.write_text("my,own,header\nH1_faithful_to_evidence_0_2 appears only in a data row\n")
    r = cli("worksheet", res, "--out", fake, "--overwrite")
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_NOT_OURS"


def test_worksheet_refuses_a_foreign_rubric_file(fake_ollama, settings, tmp_path):
    from test_review_regressions import cli
    res = tmp_path / "r.json"
    run(load_export(DT_FIXTURE), settings, res, limit=1)
    (tmp_path / "ws.rubric.txt").write_text("my own notes")
    r = cli("worksheet", res, "--out", tmp_path / "ws.csv")
    assert r.returncode == 4 and (tmp_path / "ws.rubric.txt").read_text() == "my own notes"


def test_correlate_orders_records_numerically(tmp_path):
    dt, rf = load_json(DT_FIXTURE), load_json(RF_FIXTURE)
    for doc, pfx in ((dt, "dt"), (rf, "rf")):
        doc["alerts"].append(dict(copy.deepcopy(doc["alerts"][0]), alert_id=f"{pfx}-1000001", record_id="val-1000001"))
    report = correlate(load_export(write_json(tmp_path / "dt.json", dt)), load_export(write_json(tmp_path / "rf.json", rf)))
    ids = [r["record_id"] for r in report["records"]]
    assert ids[-1] == "val-1000001" and ids == sorted(ids, key=lambda x: int(x.split("-")[1]))


def test_negative_zero_is_shown_as_zero():
    assert normalize_value("Flow Duration", -0.0).display == "0"


def test_schema_limits_evidence_used_items():
    assert response_schema(["Flow Duration", "Flow Bytes/s"])["properties"]["evidence_used"]["maxItems"] == 2


@pytest.mark.parametrize("url", ["http://localhost.evil.com:11434", "http://localhostx:11434"])
def test_localhost_must_match_exactly(url):
    with pytest.raises(ConfigError) as e:
        QwenSettings(url=url).validate()
    assert e.value.code == "CONFIG_REMOTE_HOST"


@pytest.mark.parametrize("value, allowed", [("1", True), ("true", True), ("yes", True), ("0", False),
                                            ("false", False), ("no", False)])
def test_allow_remote_env_parsing(value, allowed):
    s = QwenSettings.from_env({"NOVA_QWEN_ALLOW_REMOTE_OLLAMA": value})
    assert s.allow_remote is allowed


def test_digest_must_be_hex():
    with pytest.raises(ConfigError) as e:
        QwenSettings(expected_digest="g" * 64).validate()
    assert e.value.code == "CONFIG_DIGEST"


def test_huge_timeout_is_a_config_error():
    with pytest.raises(ConfigError) as e:
        QwenSettings(timeout_seconds=1e12).validate()
    assert e.value.code == "CONFIG_TIMEOUT"


def test_exact_model_name_lookup(fake_ollama):
    fake_ollama.model_entry = dict(fake_ollama.model_entry, name="qwen3.5:4b", model="qwen3.5:4b")
    with pytest.raises(ModelNotAvailable):
        OllamaClient(make_settings(fake_ollama.url)).check_model()


def test_boolean_token_counts_are_protocol_errors(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, {"model": MODEL, "message": {"content": "{}"}, "eval_count": True}))
    with pytest.raises(OllamaProtocolError):
        OllamaClient(make_settings(fake_ollama.url, transport_retries=0)).chat([{"role": "user", "content": "u"}],
                                                                            schema=None)


def test_body_is_read_in_bounded_chunks(fake_ollama, monkeypatch):
    sizes = []
    real_read = http.client.HTTPResponse.read

    def spy(self, amt=None):
        sizes.append(amt)
        return real_read(self, amt)

    monkeypatch.setattr(http.client.HTTPResponse, "read", spy)
    fake_ollama.chat_actions.append(("json", 200, chat_body("x" * 200_000)))
    with pytest.raises(OllamaProtocolError):
        OllamaClient(make_settings(fake_ollama.url, max_response_bytes=1000)).chat([{"role": "user", "content": "u"}],
                                                                                   schema=None)
    assert sizes and all(a is not None and a <= 65536 for a in sizes) and sum(a for a in sizes) <= 1001


def test_connect_timeout_is_the_short_one(tmp_path):
    s = QwenSettings(url="http://10.255.255.1:11434", allow_remote=True, connect_timeout_seconds=0.5,
                     timeout_seconds=60, transport_retries=0).validate()
    start = time.monotonic()
    with pytest.raises(Exception) as e:
        OllamaClient(s).version()
    assert time.monotonic() - start < 5 and type(e.value).__name__ in ("OllamaTimeout", "OllamaUnavailable")
