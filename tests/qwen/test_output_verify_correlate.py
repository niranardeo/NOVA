"""Result-file safety, independent verification of results, and DT/RF correlation."""
import copy
import os
import stat

import pytest

from qwen_test_support import DT_FIXTURE, RF_FIXTURE, load_json, write_json
from src.nova_qwen.alerts import load_export
from src.nova_qwen.consumer import run
from src.nova_qwen.correlate import check_compatible, correlate
from src.nova_qwen.errors import IncompatibleExports, OutputError
from src.nova_qwen.output import check_output_path, write_json_atomic
from src.nova_qwen.verify import verify_results


# --------------------------------------------------------------------------- writing
def test_atomic_write_produces_complete_json_with_normal_permissions(tmp_path):
    old = os.umask(0o022)
    try:
        write_json_atomic(tmp_path / "sub" / "r.json", {"a": 1})
    finally:
        os.umask(old)
    p = tmp_path / "sub" / "r.json"
    assert load_json(p) == {"a": 1}
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o644  # readable by another consumer, unlike 0600
    assert [x.name for x in p.parent.iterdir()] == ["r.json"]


def test_failed_replace_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    write_json_atomic(p, {"version": 1})
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    with pytest.raises(OutputError) as e:
        write_json_atomic(p, {"version": 2})
    assert e.value.code == "OUTPUT_WRITE_FAILED"
    assert load_json(p) == {"version": 1}
    assert [x.name for x in tmp_path.iterdir()] == ["r.json"]


def test_unserialisable_document_writes_nothing(tmp_path):
    p = tmp_path / "r.json"
    with pytest.raises(ValueError):
        write_json_atomic(p, {"x": float("nan")})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs non-root POSIX")
def test_permission_denied_directory_is_a_typed_error(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    try:
        with pytest.raises(OutputError) as e:
            write_json_atomic(d / "r.json", {"a": 1})
        assert e.value.code == "OUTPUT_WRITE_FAILED"
    finally:
        d.chmod(0o755)


def test_output_path_guards(tmp_path):
    export = tmp_path / "a.json"
    export.write_text("{}")
    with pytest.raises(OutputError) as e:
        check_output_path(export, [export], overwrite=True, resume=False)
    assert e.value.code == "OUTPUT_IS_INPUT"
    other = tmp_path / "notes.json"
    other.write_text('{"mine": true}')
    with pytest.raises(OutputError) as e:
        check_output_path(other, [export], overwrite=True, resume=False)
    assert e.value.code == "OUTPUT_NOT_OURS"
    with pytest.raises(OutputError) as e:
        check_output_path(tmp_path, [export], overwrite=True, resume=False)
    assert e.value.code == "OUTPUT_IS_DIRECTORY"
    ours = tmp_path / "r.json"
    write_json_atomic(ours, {"output_schema": "nova-qwen-explanations"})
    with pytest.raises(OutputError) as e:
        check_output_path(ours, [export], overwrite=False, resume=False)
    assert e.value.code == "OUTPUT_EXISTS"
    assert check_output_path(ours, [export], overwrite=True, resume=False) == ours
    assert check_output_path(tmp_path / "new.json", [export], overwrite=False, resume=False)


# --------------------------------------------------------------------------- verification
@pytest.fixture
def good_results(fake_ollama, settings, tmp_path):
    export = load_export(DT_FIXTURE)
    return run(export, settings, tmp_path / "r.json", limit=None), export


def test_genuine_results_verify(good_results):
    doc, export = good_results
    report = verify_results(doc, export)
    assert report == {"passed": True, "complete": True, "problems": [], "checked": 5}


TAMPER = {
    "predicted_class": (lambda d: d["results"][0].update(predicted_class="DDoS"), "predicted_class"),
    "record_id": (lambda d: d["results"][0].update(record_id="val-000003"), "record_id"),
    "run_id": (lambda d: d["results"][1].update(run_id="validation-other"), "run_id"),
    "evidence": (lambda d: d["results"][0]["evidence"].update({"Flow Duration": 101.041626}), "evidence differs"),
    "normalized": (lambda d: d["results"][0]["evidence_normalized"][0].update(value=101041626.0),
                   "evidence_normalized"),
    "fabricated": (lambda d: d["results"].append(dict(copy.deepcopy(d["results"][0]), alert_id="dt-000003")),
                   "not present in the source export"),
    "duplicate": (lambda d: d["results"].append(copy.deepcopy(d["results"][0])), "duplicate result"),
    "source_hash": (lambda d: d["source"].update(sha256="0" * 64), "source.sha256"),
    "summary": (lambda d: d["summary"].update(explained=4), "summary.explained"),
    "status": (lambda d: d["results"][0].update(status="maybe"), "invalid status"),
    "accepted_bad_answer": (lambda d: d["results"][0]["response"].update(
        explanation="This confirms the attack lasted 3.18 years."), "fails validation on re-check"),
    "rejected_with_response": (lambda d: d["results"][0].update(status="rejected"), "must not carry"),
    "abstain_mismatch": (lambda d: d["results"][0].update(status="abstained"), "disagrees with response.abstain"),
}


@pytest.mark.parametrize("name", list(TAMPER))
def test_tampered_results_are_detected(good_results, name):
    doc, export = good_results
    doc = copy.deepcopy(doc)
    change, fragment = TAMPER[name]
    change(doc)
    report = verify_results(doc, export)
    assert not report["passed"]
    assert any(fragment in p for p in report["problems"]), report["problems"]


# --------------------------------------------------------------------------- correlation
def test_correlation_of_the_fixture_run():
    report = correlate(load_export(DT_FIXTURE), load_export(RF_FIXTURE))
    assert report["counts"] == {"both_same_class": 3, "both_different_class": 1, "decision_tree_only": 1,
                                "random_forest_only": 1}
    d = report["disagreements"][0]
    assert d["record_id"] == "val-000004"
    assert (d["decision_tree"]["predicted_class"], d["random_forest"]["predicted_class"]) == ("DDoS", "DoS")
    assert report["compatibility"]["shared_records_checked"] == 4


def test_correlation_includes_explanation_status_without_merging(fake_ollama, settings, tmp_path):
    dt, rf = load_export(DT_FIXTURE), load_export(RF_FIXTURE)
    dtr = run(dt, settings, tmp_path / "dt.json", alert_ids=["dt-000004"])
    rfr = run(rf, settings, tmp_path / "rf.json", alert_ids=["rf-000004"])
    report = correlate(dt, rf, dtr, rfr)
    row = next(x for x in report["records"] if x["record_id"] == "val-000004")
    assert row["decision_tree"]["explanation_status"] == "explained"
    assert row["random_forest"]["explanation_status"] == "explained"
    assert "explanation" not in row["decision_tree"]  # explanations are not merged into the view
    with pytest.raises(IncompatibleExports) as e:
        correlate(dt, rf, rfr, dtr)
    assert e.value.code == "RESULTS_EXPORT_MISMATCH"


def _export_with(path, tmp_path, name, change):
    doc = load_json(path)
    change(doc)
    return load_export(write_json(tmp_path / name, doc))


@pytest.mark.parametrize("change, fragment", [
    (lambda d: d.update(run_id="validation-20261031T000000000000Z"), "run_id differs"),
    (lambda d: d.update(dataset="CICIDS2017-v3"), "dataset differs"),
    (lambda d: d["alerts"][0]["evidence"].update({"Flow Duration": 1.0}), "carry different evidence"),
])
def test_incompatible_exports_are_not_correlated(tmp_path, change, fragment):
    rf = _export_with(RF_FIXTURE, tmp_path, "rf.json", change)
    with pytest.raises(IncompatibleExports) as e:
        check_compatible(load_export(DT_FIXTURE), rf)
    assert e.value.code == "INCOMPATIBLE_EXPORTS" and fragment in str(e.value)


def test_detectors_must_be_one_of_each():
    with pytest.raises(IncompatibleExports) as e:
        check_compatible(load_export(DT_FIXTURE), load_export(DT_FIXTURE))
    assert "expected a decision_tree and a random_forest export" in str(e.value)


def _results_doc(*records):
    return {"results": [{"record_id": rec, "detector": det, "predicted_class": cls, "status": "explained",
                         "evidence": {}, "evidence_normalized": []} for rec, det, cls in records]}


def test_worksheet_uses_the_team_criteria_with_prioritization_not_applicable():
    from src.nova_qwen.worksheet import HUMAN_COLUMNS, NO_PRIORITY_POLICY, PRIORITIZATION_COLUMN, rows_from_results
    assert HUMAN_COLUMNS[:5] == ["explanation_accuracy_0_2", "evidence_grounding_0_2", "uncertainty_handling_0_2",
                                 "consistency_0_2", PRIORITIZATION_COLUMN]
    row = rows_from_results([_results_doc(("val-000002", "decision_tree", "DoS"))])[0]
    assert row[PRIORITIZATION_COLUMN] == NO_PRIORITY_POLICY
    assert row["explanation_accuracy_0_2"] == "" and row["reviewer"] == ""


def test_worksheet_groups_the_same_row_across_classifiers_for_consistency_review():
    from src.nova_qwen.worksheet import rows_from_results
    rows = rows_from_results([_results_doc(("val-000002", "decision_tree", "DoS"), ("val-000003", "decision_tree", "DoS")),
                              _results_doc(("val-000002", "random_forest", "DDoS"))])
    groups = {(r["record_id"], r["detector"]): r["consistency_group"] for r in rows}
    assert groups[("val-000002", "decision_tree")] == groups[("val-000002", "random_forest")] == "val-000002"
    assert groups[("val-000003", "decision_tree")] == ""


def test_worksheet_flags_rare_classes_and_states_the_false_negative_limit():
    from src.nova_qwen.worksheet import RUBRIC, rows_from_results
    rows = rows_from_results([_results_doc(("val-1", "decision_tree", "Bots"), ("val-2", "decision_tree", "Web Attacks"),
                                           ("val-3", "decision_tree", "DoS"))])
    assert "never what the classifier missed" in rows[0]["class_review_note"] and rows[1]["class_review_note"]
    assert rows[2]["class_review_note"] == ""
    assert "No priority policy is approved" in RUBRIC and "Report totals out of 8" in RUBRIC


def test_worksheet_header_marker_accepts_current_and_earlier_sheets_only():
    from src.nova_qwen.worksheet import is_worksheet_header
    assert is_worksheet_header("alert_id,explanation_accuracy_0_2,reviewer\nrow")
    assert is_worksheet_header("alert_id,H1_faithful_to_evidence_0_2,reviewer\nrow")
    assert not is_worksheet_header("mine\nexplanation_accuracy_0_2 in a data row")
    assert not is_worksheet_header("")


def test_consistency_group_needs_the_same_run_dataset_and_evidence():
    from src.nova_qwen.worksheet import rows_from_results

    def doc(det, run_id, duration):
        return {"results": [{"record_id": "val-000002", "detector": det, "predicted_class": "DoS", "status": "explained",
                             "run_id": run_id, "dataset": "d", "evidence": {"Flow Duration": duration},
                             "evidence_normalized": []}]}

    assert all(r["consistency_group"] == "" for r in rows_from_results([doc("decision_tree", "a", 1), doc("random_forest", "b", 2)]))
    assert all(r["consistency_group"] == "" for r in rows_from_results([doc("decision_tree", "a", 1), doc("random_forest", "a", 2)]))
    same = rows_from_results([doc("decision_tree", "a", 1), doc("random_forest", "a", 1)])
    assert [r["consistency_group"] for r in same] == ["val-000002", "val-000002"]


def test_text_that_cannot_be_encoded_is_a_write_failure_and_leaves_no_file(tmp_path):
    from src.nova_qwen.errors import OutputError
    from src.nova_qwen.output import write_text_atomic
    with pytest.raises(OutputError) as info:
        write_text_atomic(tmp_path / "r.json", "cut off \ud83d")
    assert info.value.code == "OUTPUT_WRITE_FAILED" and list(tmp_path.iterdir()) == []
