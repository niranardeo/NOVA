"""Schema 1.1 reader. Every negative case starts from a VALID document, changes exactly one thing,
and asserts the exact error class and code, so a test cannot pass because an earlier check fired."""
import copy
import hashlib
import json
import os

import pytest

from qwen_test_support import DT_EMPTY, DT_FIXTURE, RF_FIXTURE, load_json, write_json
from src.nova_qwen.alerts import load_export, parse_export
from src.nova_qwen.errors import (AlertContractError, AlertFileInvalidJSON, AlertFileMissing, AlertFileUnreadable,
                                  UnsupportedSchemaVersion)

VALID_DT = load_json(DT_FIXTURE)


def test_valid_decision_tree_export_is_read_completely():
    exp = load_export(DT_FIXTURE)
    raw = DT_FIXTURE.read_bytes()
    assert exp.detector == "decision_tree" and exp.schema_version == "1.1"
    assert exp.run_id == "validation-20261030T000000000000Z" and exp.dataset == "CICIDS2017"
    assert exp.sha256 == hashlib.sha256(raw).hexdigest() and exp.size_bytes == len(raw)
    assert [a.alert_id for a in exp.alerts] == ["dt-000002", "dt-000004", "dt-000005", "dt-000006", "dt-000008"]
    assert [a.record_id for a in exp.alerts] == ["val-000002", "val-000004", "val-000005", "val-000006", "val-000008"]
    assert [a.record_position for a in exp.alerts] == [2, 4, 5, 6, 8]
    for alert, src in zip(exp.alerts, VALID_DT["alerts"]):
        assert alert.predicted_class == src["predicted_class"]
        assert alert.evidence == src["evidence"]


def test_valid_random_forest_export_and_source_record_has_no_paths():
    exp = load_export(RF_FIXTURE, expected_detector="random_forest")
    assert [a.alert_id for a in exp.alerts] == ["rf-000002", "rf-000003", "rf-000004", "rf-000006", "rf-000008"]
    rec = exp.source_record()
    assert rec["file_name"] == "random_forest_alerts.json" and "/" not in rec["file_name"]
    assert rec["alert_count"] == 5


def test_empty_export_is_valid():
    exp = load_export(DT_EMPTY)
    assert exp.alerts == ()


# --------------------------------------------------------------------------- file level
def test_missing_file(tmp_path):
    with pytest.raises(AlertFileMissing) as e:
        load_export(tmp_path / "nope.json")
    assert e.value.code == "ALERT_FILE_MISSING"


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(AlertFileUnreadable) as e:
        load_export(tmp_path)
    assert e.value.code == "ALERT_FILE_UNREADABLE"


@pytest.mark.skipif(os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="permission bits are not enforced for root or on this OS")
def test_permission_denied(tmp_path):
    p = write_json(tmp_path / "a.json", VALID_DT)
    p.chmod(0)
    try:
        with pytest.raises(AlertFileUnreadable) as e:
            load_export(p)
        assert e.value.code == "ALERT_FILE_UNREADABLE"
    finally:
        p.chmod(0o644)


@pytest.mark.parametrize("content", [
    b'{"schema_version": "1.1", ', b"\xff\xfe not utf8", b"",
    json.dumps(VALID_DT).replace('"Flow Bytes/s": 123.7509776', '"Flow Bytes/s": NaN').encode(),
    json.dumps(VALID_DT).replace('"Flow Bytes/s": 123.7509776', '"Flow Bytes/s": Infinity').encode(),
    json.dumps(VALID_DT)[:-1].encode() + b', "dataset": "other"}',
], ids=["truncated", "not_utf8", "empty", "nan_literal", "infinity_literal", "duplicate_key"])
def test_invalid_json(tmp_path, content):
    p = tmp_path / "a.json"
    p.write_bytes(content)
    with pytest.raises(AlertFileInvalidJSON) as e:
        load_export(p)
    assert e.value.code == "ALERT_FILE_INVALID_JSON"


# --------------------------------------------------------------------------- document level
def mutated(fn):
    doc = copy.deepcopy(VALID_DT)
    fn(doc)
    return doc


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version="1.0"), lambda d: d.pop("schema_version"),
    lambda d: d.update(schema_version=1.1), lambda d: d.update(schema_version="2.0"),
], ids=["v1.0", "missing", "number", "v2.0"])
def test_unsupported_schema_version(change):
    with pytest.raises(UnsupportedSchemaVersion) as e:
        parse_export(mutated(change))
    assert e.value.code == "UNSUPPORTED_SCHEMA_VERSION"


CONTRACT_CASES = {
    "top_level_list": (lambda d: None, "CONTRACT_TOP_LEVEL"),
    "extra_top_key": (lambda d: d.update(ground_truth=["DoS"]), "CONTRACT_TOP_LEVEL_KEYS"),
    "missing_run_id": (lambda d: d.pop("run_id"), "CONTRACT_TOP_LEVEL_KEYS"),
    "missing_dataset": (lambda d: d.pop("dataset"), "CONTRACT_TOP_LEVEL_KEYS"),
    "blank_run_id": (lambda d: d.update(run_id="   "), "CONTRACT_RUN_ID"),
    "non_string_run_id": (lambda d: d.update(run_id=123), "CONTRACT_RUN_ID"),
    "padded_run_id": (lambda d: d.update(run_id=" validation-1 "), "CONTRACT_RUN_ID"),
    "control_char_run_id": (lambda d: d.update(run_id="validation\x00-1"), "CONTRACT_RUN_ID"),
    "invisible_char_run_id": (lambda d: d.update(run_id="validation​-1"), "CONTRACT_RUN_ID"),
    "blank_dataset": (lambda d: d.update(dataset=""), "CONTRACT_DATASET"),
    "non_string_dataset": (lambda d: d.update(dataset=None), "CONTRACT_DATASET"),
    "unknown_detector": (lambda d: d.update(detector="naive_bayes"), "CONTRACT_DETECTOR"),
    "detector_not_string": (lambda d: d.update(detector=["decision_tree"]), "CONTRACT_DETECTOR"),
    "alerts_not_list": (lambda d: d.update(alerts={}), "CONTRACT_ALERTS_TYPE"),
    "alert_not_object": (lambda d: d["alerts"].__setitem__(0, "dt-000002"), "CONTRACT_ALERT_TYPE"),
    "alert_missing_record_id": (lambda d: d["alerts"][0].pop("record_id"), "CONTRACT_ALERT_KEYS"),
    "alert_extra_ground_truth": (lambda d: d["alerts"][0].update(ground_truth="DoS"), "CONTRACT_ALERT_KEYS"),
    "alert_extra_confidence": (lambda d: d["alerts"][0].update(confidence=0.9), "CONTRACT_ALERT_KEYS"),
    "alert_id_wrong_prefix": (lambda d: d["alerts"][0].update(alert_id="rf-000002"), "CONTRACT_ALERT_ID"),
    "alert_id_malformed": (lambda d: d["alerts"][0].update(alert_id="dt-2"), "CONTRACT_ALERT_ID"),
    "alert_id_not_string": (lambda d: d["alerts"][0].update(alert_id=2), "CONTRACT_ALERT_ID"),
    "record_id_malformed": (lambda d: d["alerts"][0].update(record_id="row-000002"), "CONTRACT_RECORD_ID"),
    "record_id_other_row": (lambda d: d["alerts"][0].update(record_id="val-000003"), "CONTRACT_ID_POSITION_MISMATCH"),
    "record_position_zero": (lambda d: d["alerts"][0].update(alert_id="dt-000000", record_id="val-000000"),
                             "CONTRACT_ID_POSITION_MISMATCH"),
    "duplicate_alert": (lambda d: d["alerts"].append(copy.deepcopy(d["alerts"][0])), "CONTRACT_DUPLICATE_ALERT_ID"),
    "normal_traffic_alert": (lambda d: d["alerts"][0].update(predicted_class="Normal Traffic"),
                             "CONTRACT_PREDICTED_CLASS"),
    "class_not_string": (lambda d: d["alerts"][0].update(predicted_class=4), "CONTRACT_PREDICTED_CLASS"),
    "label_encoded_class": (lambda d: d["alerts"][0].update(predicted_class="4"), "CONTRACT_UNKNOWN_CLASS"),
    "unknown_class": (lambda d: d["alerts"][0].update(predicted_class="Benign"), "CONTRACT_UNKNOWN_CLASS"),
    "bytes_repr_class": (lambda d: d["alerts"][0].update(predicted_class="b'DoS'"), "CONTRACT_UNKNOWN_CLASS"),
    "evidence_empty": (lambda d: d["alerts"][0].update(evidence={}), "CONTRACT_EVIDENCE_EMPTY"),
    "evidence_not_object": (lambda d: d["alerts"][0].update(evidence=[1, 2]), "CONTRACT_EVIDENCE_EMPTY"),
    "evidence_extra_feature": (lambda d: d["alerts"][0]["evidence"].update({"Destination Port": 80}),
                               "CONTRACT_EVIDENCE_FEATURE"),
    "evidence_ground_truth": (lambda d: d["alerts"][0]["evidence"].update({"Attack Type": "DoS"}),
                              "CONTRACT_EVIDENCE_FEATURE"),
    "evidence_string": (lambda d: d["alerts"][0]["evidence"].update({"Flow Duration": "101"}),
                        "CONTRACT_EVIDENCE_TYPE"),
    "evidence_null": (lambda d: d["alerts"][0]["evidence"].update({"Flow Duration": None}), "CONTRACT_EVIDENCE_TYPE"),
    "evidence_bool": (lambda d: d["alerts"][0]["evidence"].update({"Total Fwd Packets": True}),
                      "CONTRACT_EVIDENCE_TYPE"),
}


@pytest.mark.parametrize("name", list(CONTRACT_CASES))
def test_contract_violation_reaches_its_own_check(name):
    change, code = CONTRACT_CASES[name]
    doc = ["not", "an", "object"] if name == "top_level_list" else mutated(change)
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert type(e.value) is AlertContractError
    assert e.value.code == code


def test_same_record_under_a_second_alert_id_is_rejected_as_position_mismatch():
    # A duplicate record_id can only appear next to a different alert_id, which names another row.
    doc = mutated(lambda d: d["alerts"].append(dict(copy.deepcopy(d["alerts"][0]), alert_id="dt-000009")))
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert e.value.code == "CONTRACT_ID_POSITION_MISMATCH"


def test_non_finite_evidence_after_parse():
    doc = mutated(lambda d: d["alerts"][0]["evidence"].update({"Flow Duration": float("inf")}))
    with pytest.raises(AlertContractError) as e:
        parse_export(doc)
    assert e.value.code == "CONTRACT_EVIDENCE_FINITE"


def test_expected_detector_mismatch():
    with pytest.raises(AlertContractError) as e:
        load_export(DT_FIXTURE, expected_detector="random_forest")
    assert e.value.code == "CONTRACT_DETECTOR_MISMATCH"


def test_integer_evidence_and_partial_allowlist_accepted():
    doc = mutated(lambda d: d["alerts"][0].update(evidence={"Flow Duration": 5}))
    exp = parse_export(doc)
    assert exp.alerts[0].evidence == {"Flow Duration": 5}
