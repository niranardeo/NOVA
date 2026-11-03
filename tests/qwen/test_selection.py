"""Representative alert selection: deterministic, bounded, and honest about what it did not cover."""
import pytest

from qwen_test_support import DT_FIXTURE, RF_FIXTURE
from src.nova_qwen.alerts import AlertExport, Alert, load_export
from src.nova_qwen import selection as S


def alert(prefix, position, cls, duration=1000):
    return Alert(alert_id=f"{prefix}-{position:06d}", record_id=f"val-{position:06d}", record_position=position,
                 predicted_class=cls, evidence={"Flow Duration": float(duration), "Total Fwd Packets": 2.0,
                                                "Flow Bytes/s": 1.5})


def export(det, alerts):
    return AlertExport(file_name=f"{det}.json", sha256="0" * 64, size_bytes=1, schema_version="1.1", run_id="r",
                       dataset="d", detector=det, alerts=tuple(alerts))


def pair():
    dt = [alert("dt", 1, "DoS"), alert("dt", 2, "DoS"), alert("dt", 3, "DDoS"), alert("dt", 4, "Bots"),
          alert("dt", 5, "Web Attacks"), alert("dt", 6, "DoS"), alert("dt", 7, "Port Scanning", 110_000_000),
          alert("dt", 8, "Brute Force"), alert("dt", 9, "DoS", -3), alert("dt", 10, "DDoS")]
    rf = [alert("rf", 2, "DoS"), alert("rf", 3, "DDoS"), alert("rf", 4, "Bots"), alert("rf", 5, "Web Attacks"),
          alert("rf", 6, "DDoS"), alert("rf", 7, "Port Scanning", 110_000_000), alert("rf", 8, "Brute Force"),
          alert("rf", 11, "DoS")]
    return export("decision_tree", dt), export("random_forest", rf)


def test_covers_each_requested_case_within_the_limit():
    dt, rf = pair()
    plan = S.plan_representative(dt, rf, 6)
    entry = plan["decision_tree"]
    assert len(entry["alert_ids"]) <= 6
    for target in ("class:DoS", "class:DDoS", "class:Bots", "class:Web Attacks", S.DISAGREEMENT, S.LONGEST):
        assert target in entry["covered"], entry
    classes = {a["predicted_class"] for a in entry["alerts"]}
    assert {"DoS", "DDoS", "Bots", "Web Attacks"} <= classes


def test_prefers_a_real_class_disagreement_and_the_same_row_for_both_classifiers():
    dt, rf = pair()
    plan = S.plan_representative(dt, rf, 6)
    dt_records = {a["record_id"] for a in plan["decision_tree"]["alerts"] if S.DISAGREEMENT in a["reasons"]}
    assert "val-000006" in dt_records  # DT says DoS, RF says DDoS
    shared = {a["record_id"] for a in plan["decision_tree"]["alerts"]} & {a["record_id"] for a in plan["random_forest"]["alerts"]}
    assert shared == {f"val-00000{i}" for i in range(2, 8)}  # both classifiers explain the same six rows


def test_is_deterministic():
    dt, rf = pair()
    assert S.plan_representative(dt, rf, 6) == S.plan_representative(dt, rf, 6)


@pytest.mark.parametrize("limit", [1, 2, 3, 6, 25])
def test_never_exceeds_the_limit(limit):
    dt, rf = pair()
    plan = S.plan_representative(dt, rf, limit)
    assert len(plan["decision_tree"]["alert_ids"]) <= limit and len(plan["random_forest"]["alert_ids"]) <= limit
    assert len(set(plan["decision_tree"]["alert_ids"])) == len(plan["decision_tree"]["alert_ids"])


def test_reports_what_the_limit_left_out():
    dt, rf = pair()
    entry = S.plan_representative(dt, rf, 2)["decision_tree"]
    assert entry["not_covered_because_of_limit"] and len(entry["alert_ids"]) == 2


def test_reports_classes_absent_from_the_export():
    dt = export("decision_tree", [alert("dt", 1, "DoS"), alert("dt", 2, "DDoS")])
    rf = export("random_forest", [alert("rf", 1, "DoS")])
    entry = S.plan_representative(dt, rf, 6)["decision_tree"]
    assert "class:Bots" in entry["not_available_in_export"] and "class:Web Attacks" in entry["not_available_in_export"]
    assert entry["alert_ids"] == ["dt-000001", "dt-000002"]


def test_fills_up_to_the_limit_in_row_order_when_few_targets_exist():
    dt = export("decision_tree", [alert("dt", i, "DoS") for i in range(1, 9)])
    rf = export("random_forest", [alert("rf", i, "DoS") for i in range(1, 9)])
    entry = S.plan_representative(dt, rf, 3)["decision_tree"]
    assert entry["alert_ids"] == ["dt-000001", "dt-000002", "dt-000003"]
    reasons = {a["alert_id"]: a["reasons"] for a in entry["alerts"]}
    assert S.FILL not in reasons["dt-000001"] and reasons["dt-000002"] == [S.FILL] and reasons["dt-000003"] == [S.FILL]


def test_negative_duration_is_chosen_when_room_allows():
    dt, rf = pair()
    entry = S.plan_representative(dt, rf, 7)["decision_tree"]
    assert S.NEGATIVE in entry["covered"] and "dt-000009" in entry["alert_ids"]


def test_empty_exports_select_nothing():
    dt, rf = export("decision_tree", []), export("random_forest", [])
    plan = S.plan_representative(dt, rf, 6)
    assert plan["decision_tree"]["alert_ids"] == [] and plan["random_forest"]["alert_ids"] == []
    assert plan["decision_tree"]["covered"] == []


def test_one_empty_export_still_works():
    dt, rf = export("decision_tree", [alert("dt", 1, "DoS")]), export("random_forest", [])
    plan = S.plan_representative(dt, rf, 6)
    assert plan["decision_tree"]["alert_ids"] == ["dt-000001"]
    assert S.ONE_CLASSIFIER_ONLY in plan["decision_tree"]["covered"]
    assert S.DISAGREEMENT in plan["decision_tree"]["not_available_in_export"]


def test_first_method_is_row_order():
    dt, rf = load_export(DT_FIXTURE), load_export(RF_FIXTURE)
    plan = S.plan_first(dt, rf, 2)
    assert plan["decision_tree"]["alert_ids"] == [a.alert_id for a in dt.alerts[:2]]


def test_requested_ids_are_added_once_with_their_reason():
    dt, rf = pair()
    plan = S.plan_representative(dt, rf, 1)
    S.add_requested(plan, dt, rf, ["dt-000010", "dt-000010"])
    entry = plan["decision_tree"]
    assert entry["alert_ids"].count("dt-000010") == 1
    assert [a for a in entry["alerts"] if a["alert_id"] == "dt-000010"][0]["reasons"] == [S.REQUESTED]


def test_runs_on_real_adapter_fixtures():
    plan = S.plan_representative(load_export(DT_FIXTURE), load_export(RF_FIXTURE), 6)
    assert plan["decision_tree"]["alert_ids"] and plan["random_forest"]["alert_ids"]
    assert S.NEGATIVE in plan["decision_tree"]["covered"]  # fixture row 5 has a negative duration


def test_a_one_sided_row_does_not_count_as_a_class_disagreement():
    dt = export("decision_tree", [alert("dt", 1, "Bots"), alert("dt", 2, "DoS")])
    rf = export("random_forest", [alert("rf", 2, "DDoS")])
    entry = S.plan_representative(dt, rf, 6)["decision_tree"]
    assert "dt-000002" in entry["alert_ids"]
    reasons = {a["alert_id"]: a["reasons"] for a in entry["alerts"]}
    assert S.DISAGREEMENT in reasons["dt-000002"] and S.DISAGREEMENT not in reasons["dt-000001"]
    assert S.ONE_CLASSIFIER_ONLY in reasons["dt-000001"]


def test_no_class_disagreement_is_reported_as_unavailable():
    dt = export("decision_tree", [alert("dt", 1, "DoS"), alert("dt", 2, "DoS")])
    rf = export("random_forest", [alert("rf", 1, "DoS")])
    entry = S.plan_representative(dt, rf, 6)["decision_tree"]
    assert S.DISAGREEMENT in entry["not_available_in_export"] and S.DISAGREEMENT not in entry["covered"]


def test_a_requested_row_moves_its_target_out_of_not_covered():
    dt, rf = pair()
    plan = S.plan_representative(dt, rf, 1)
    assert S.NEGATIVE in plan["decision_tree"]["not_covered_because_of_limit"]
    S.add_requested(plan, dt, rf, ["dt-000009"])
    entry = plan["decision_tree"]
    assert S.NEGATIVE in entry["covered"] and S.NEGATIVE not in entry["not_covered_because_of_limit"]


def test_selection_stays_fast_on_a_full_size_export():
    import time
    dt = export("decision_tree", [alert("dt", i, "DoS" if i % 2 else "DDoS") for i in range(1, 64_001)])
    rf = export("random_forest", [alert("rf", i, "DoS") for i in range(1, 64_001)])
    start = time.perf_counter()
    S.plan_representative(dt, rf, 6)
    assert time.perf_counter() - start < 5
