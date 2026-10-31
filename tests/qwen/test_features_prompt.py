"""Deterministic unit normalisation, data-quality notes, prompt content and the context budget."""
import json

import pytest

from qwen_test_support import DT_FIXTURE
from src.nova_qwen.alerts import load_export
from src.nova_qwen.errors import PromptBudgetExceeded
from src.nova_qwen.features import data_quality_notes, normalize_evidence, normalize_value
from src.nova_qwen.prompt import (SYSTEM_PROMPT, build_messages, check_budget, feedback_message, request_bytes,
                                  response_schema)


@pytest.mark.parametrize("raw, display, value", [
    (101041626.0, "101.041626", 101.041626),   # the "3.18 years" regression case is ~101 s
    (116827162.0, "116.827162", 116.827162),   # the "3.74 years" case
    (119989925.0, "119.989925", 119.989925),   # the "33.3 hours" case
    (67.0, "0.000067", 6.7e-05),
    (21760.0, "0.02176", 0.02176),
    (5000000, "5", 5.0),
    (0.0, "0", 0.0),
    (-1.0, "-0.000001", -1e-06),
    (1, "0.000001", 1e-06),
    (120000000000.0, "120000", 120000.0),
])
def test_flow_duration_microseconds_to_seconds(raw, display, value):
    f = normalize_value("Flow Duration", raw)
    assert (f.raw_value, f.raw_unit, f.unit) == (raw, "microseconds", "seconds")
    assert f.display == display
    assert f.value == pytest.approx(value, rel=0, abs=1e-12)
    assert "e" not in f.display.lower()  # never scientific notation in front of the model


@pytest.mark.parametrize("feature, raw, display, unit", [
    ("Total Fwd Packets", 11.0, "11", "packets"),
    ("Total Fwd Packets", 4062, "4062", "packets"),
    ("Total Fwd Packets", 1.5, "1.5", "packets"),
    ("Flow Bytes/s", 123.7509776, "123.7509776", "bytes per second"),
    ("Flow Bytes/s", 0.154073759, "0.154073759", "bytes per second"),
    ("Flow Bytes/s", 1e-07, "0.0000001", "bytes per second"),
])
def test_other_features_keep_their_values(feature, raw, display, unit):
    f = normalize_value(feature, raw)
    assert (f.display, f.unit, f.raw_value) == (display, unit, raw)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "100", None])
def test_normalize_rejects_non_numbers(bad):
    with pytest.raises(ValueError):
        normalize_value("Flow Duration", bad)


def test_normalize_evidence_keeps_allowlist_order_and_raw_values():
    ev = {"Flow Bytes/s": 2.0, "Flow Duration": 3_000_000.0, "Total Fwd Packets": 4.0}
    out = normalize_evidence(ev)
    assert [f.feature for f in out] == ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]
    assert {f.feature: f.raw_value for f in out} == ev


def test_data_quality_notes_fire_only_for_implausible_values():
    assert data_quality_notes(normalize_evidence({"Flow Duration": 5.0, "Total Fwd Packets": 2.0})) == []
    neg = data_quality_notes(normalize_evidence({"Flow Duration": -1.0}))
    assert len(neg) == 1 and "negative" in neg[0] and "-0.000001 seconds" in neg[0]
    frac = data_quality_notes(normalize_evidence({"Total Fwd Packets": 2.5}))
    assert len(frac) == 1 and "whole number" in frac[0]
    neg_rate = data_quality_notes(normalize_evidence({"Flow Bytes/s": -3.0}))
    assert len(neg_rate) == 1 and "Flow Bytes/s is negative" in neg_rate[0]


def _messages_for(alert):
    n = normalize_evidence(alert.evidence)
    return n, build_messages("decision_tree", alert.predicted_class, n, data_quality_notes(n))


def test_prompt_carries_units_but_no_identifiers_or_raw_microseconds():
    export = load_export(DT_FIXTURE)
    alert = export.alerts[0]  # dt-000002, 101,041,626 us
    _, messages = _messages_for(alert)
    user = messages[1]["content"]
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["classifier"] == "Decision Tree" and payload["predicted_class"] == "DoS"
    assert payload["prediction_status"] == "unverified classifier prediction"
    assert payload["evidence"][0] == {"feature": "Flow Duration", "value": "101.041626", "unit": "seconds"}
    for forbidden in ("101041626", alert.alert_id, alert.record_id, export.run_id, "val-", "dt-", "Attack Type"):
        assert forbidden not in user
    assert "UNVERIFIED" in SYSTEM_PROMPT and "Do not convert units" in SYSTEM_PROMPT
    assert '"priority": must be null' in SYSTEM_PROMPT


def test_data_quality_note_is_passed_to_the_model():
    export = load_export(DT_FIXTURE)
    alert = next(a for a in export.alerts if a.alert_id == "dt-000005")  # Flow Duration -1 us
    _, messages = _messages_for(alert)
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert len(payload["data_quality_notes"]) == 1 and "negative" in payload["data_quality_notes"][0]


def test_response_schema_restricts_evidence_used_to_supplied_features():
    schema = response_schema(["Flow Duration"])
    assert schema["properties"]["evidence_used"]["items"]["enum"] == ["Flow Duration"]
    assert schema["properties"]["priority"] == {"type": "null"}
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"observations", "explanation", "limitations", "evidence_used",
                                       "priority", "abstain"}
    assert "alert_id" not in schema["properties"]  # identifiers are attached in code, not generated


def test_budget_accepts_a_real_alert_with_room_for_one_re_ask():
    export = load_export(DT_FIXTURE)
    n, messages = _messages_for(export.alerts[0])
    schema = response_schema([f.feature for f in n])
    size = check_budget(messages, schema, num_ctx=8192, num_predict=512)
    assert size == request_bytes(messages, schema)
    re_ask = messages + [{"role": "assistant", "content": "x" * 2000},
                         {"role": "user", "content": feedback_message([{"code": "C", "message": "m"}])}]
    check_budget(re_ask, schema, num_ctx=8192, num_predict=512)


def test_budget_refuses_requests_that_could_be_truncated():
    export = load_export(DT_FIXTURE)
    n, messages = _messages_for(export.alerts[0])
    schema = response_schema([f.feature for f in n])
    size = request_bytes(messages, schema)
    with pytest.raises(PromptBudgetExceeded) as e:
        check_budget(messages, schema, num_ctx=size + 128 + 512 - 1, num_predict=512)
    assert e.value.code == "PROMPT_BUDGET_EXCEEDED"
    assert e.value.details["request_bytes"] == size
    check_budget(messages, schema, num_ctx=size + 128 + 512, num_predict=512)  # exactly fits


def test_budget_counts_bytes_not_characters():
    schema = response_schema(["Flow Duration"])
    ascii_msgs = [{"role": "user", "content": "a" * 100}]
    multibyte = [{"role": "user", "content": "é" * 100}]  # 2 bytes each in UTF-8
    assert request_bytes(multibyte, schema) - request_bytes(ascii_msgs, schema) == 100
