"""Production consumer against the real Ollama + qwen3.5:4b-q4_K_M (opt-in, see conftest.py).

Covers: availability, exact model and digest, loading, inference, strict response validation,
identifier preservation, evidence units, empty input, timeout, truncation, repeatability, and a
bounded run over every fixture alert of both classifiers. Structural/deterministic checks only;
semantic quality needs human review (python -m src.nova_qwen worksheet ...).
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.nova_qwen.alerts import load_export
from src.nova_qwen.consumer import explain_alert, run
from src.nova_qwen.errors import ModelDigestMismatch, ModelNotAvailable
from src.nova_qwen.ollama import OllamaClient
from src.nova_qwen.verify import verify_results

FIXTURES = Path(__file__).resolve().parents[1] / "qwen" / "fixtures"
DT = FIXTURES / "decision_tree_alerts.json"
RF = FIXTURES / "random_forest_alerts.json"


def test_ollama_and_pinned_model(live_client, live_settings):
    info = live_client.check_model()
    assert info.name == live_settings.model == "qwen3.5:4b-q4_K_M"
    assert info.digest_verified and info.digest == live_settings.expected_digest
    assert info.quantization == "Q4_K_M" and "thinking" in info.capabilities


def test_missing_model_is_not_replaced(live_settings):
    client = OllamaClient(replace(live_settings, model="qwen3.5:does-not-exist"))
    with pytest.raises(ModelNotAvailable):
        client.check_model()


def test_wrong_digest_is_refused(live_settings):
    client = OllamaClient(replace(live_settings, expected_digest="0" * 64))
    with pytest.raises(ModelDigestMismatch):
        client.check_model()


@pytest.mark.parametrize("fixture", [DT, RF], ids=["decision_tree", "random_forest"])
def test_bounded_real_inference_for_each_classifier(live_settings, tmp_path, fixture):
    export = load_export(fixture)
    doc = run(export, live_settings, tmp_path / "results.json", limit=None)
    (tmp_path / "results.json").replace(tmp_path / f"{export.detector}_live_results.json")
    assert doc["model"]["contacted"] and doc["model"]["digest_verified"]
    assert doc["summary"]["failed"] == 0 and doc["summary"]["processed"] == 5
    assert verify_results(doc, export)["passed"]  # ids, classes, raw evidence, and re-validated answers
    accepted = [r for r in doc["results"] if r["status"] in ("explained", "abstained")]
    assert len(accepted) >= 4, [(r["alert_id"], r["validation"]) for r in doc["results"]]
    for r in accepted:
        assert r["attempts"][-1]["done_reason"] == "stop"
        assert r["attempts"][-1]["prompt_eval_count"] <= live_settings.num_ctx
        text = json.dumps(r["response"])
        assert "year" not in text and "hour" not in text  # the old unit failure


def test_regression_101_second_flow(live_client, live_settings):
    export = load_export(DT)
    alert = next(a for a in export.alerts if a.alert_id == "dt-000002")
    r = explain_alert(live_client, export, alert, live_settings)
    assert r["status"] in ("explained", "abstained"), r["validation"]
    assert r["evidence"]["Flow Duration"] == 101041626.0
    assert r["evidence_normalized"][0]["display"] == "101.041626"


def test_empty_export_does_not_contact_ollama(live_settings, tmp_path):
    doc = run(load_export(FIXTURES / "decision_tree_alerts_empty.json"), live_settings, tmp_path / "r.json")
    assert doc["model"]["contacted"] is False and doc["results"] == []


def test_repeatability_under_fixed_settings(live_client, live_settings):
    export = load_export(RF)
    alert = export.alerts[0]
    a = explain_alert(live_client, export, alert, live_settings)
    b = explain_alert(live_client, export, alert, live_settings)
    assert a["attempts"][0]["raw_response"] == b["attempts"][0]["raw_response"]


def test_truncated_generation_is_detected_and_rejected(live_settings):
    s = replace(live_settings, num_predict=64, max_attempts=1)
    export = load_export(DT)
    r = explain_alert(OllamaClient(s), export, export.alerts[0], s)
    assert r["status"] == "rejected" and r["attempts"][0]["done_reason"] == "length"
    assert r["response"] is None


def test_timeout_is_reported_not_hung(live_settings):
    s = replace(live_settings, timeout_seconds=0.5, transport_retries=0)
    export = load_export(DT)
    r = explain_alert(OllamaClient(s), export, export.alerts[1], s)
    assert r["status"] == "failed" and r["error"]["code"] == "OLLAMA_TIMEOUT"
