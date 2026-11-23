"""A run is reported as PASS only when it really is one."""
import pytest

from src.nova_qwen import outcome as O


def doc(explained=0, abstained=0, rejected=0, failed=0, pending=0, run_status="complete"):
    selected = explained + abstained + rejected + failed + pending
    return {"run_status": run_status,
            "summary": {"explained": explained, "abstained": abstained, "rejected": rejected, "failed": failed,
                        "selected": selected, "processed": selected - pending, "pending": pending}}


OK = {"passed": True, "problems": []}
BAD = {"passed": False, "problems": ["evidence differs"]}


@pytest.mark.parametrize("d, verify, alerts, verdict", [
    (doc(explained=5), OK, 100, O.PASS),
    (doc(explained=3, abstained=2), OK, 100, O.PASS),
    (doc(abstained=5), OK, 100, O.NO_EXPLANATIONS),
    (doc(explained=4, rejected=1), OK, 100, O.REJECTED_ANSWERS),
    (doc(rejected=5), OK, 100, O.REJECTED_ANSWERS),
    (doc(explained=4, failed=1), OK, 100, O.FAILED),
    (doc(explained=3, rejected=1, failed=1), OK, 100, O.FAILED),
    (doc(explained=5), BAD, 100, O.FAILED),
    (doc(explained=5), None, 100, O.FAILED),
    (doc(explained=3, pending=2, run_status="interrupted"), OK, 100, O.INCOMPLETE),
    (doc(explained=3, pending=2, run_status="interrupted"), BAD, 100, O.INCOMPLETE),
    (doc(explained=3, pending=2, failed=3, run_status="interrupted"), OK, 100, O.FAILED),
    (doc(explained=5, run_status="in_progress"), OK, 100, O.INCOMPLETE),
    (doc(), OK, 0, O.SKIPPED_NO_ALERTS),
    (None, None, 100, O.INCOMPLETE),
])
def test_verdicts(d, verify, alerts, verdict):
    assert O.classifier_outcome(d, verify, alerts)["verdict"] == verdict


def test_abstentions_and_skips_are_counted_separately():
    o = O.classifier_outcome(doc(explained=3, abstained=2), OK, 100)
    assert o["counts"] == {"explained": 3, "abstained": 2, "rejected": 0, "failed": 0, "incomplete": 0, "skipped": 95,
                           "selected": 5, "alerts_in_export": 100}
    assert "explained=3 abstained=2 rejected=0 failed=0 incomplete=0 skipped=95" == O.counts_line(o["counts"])


def test_reasons_name_each_problem():
    o = O.classifier_outcome(doc(explained=1, rejected=2, failed=1), BAD, 10)
    text = " ".join(o["reasons"])
    assert "failed" in text and "rejected" in text and "evidence differs" in text


@pytest.mark.parametrize("a, b, overall", [
    (O.PASS, O.PASS, O.PASS),
    (O.PASS, O.REJECTED_ANSWERS, O.REJECTED_ANSWERS),
    (O.NO_EXPLANATIONS, O.PASS, O.NO_EXPLANATIONS),
    (O.FAILED, O.INCOMPLETE, O.FAILED),
    (O.SKIPPED_NO_ALERTS, O.PASS, O.SKIPPED_NO_ALERTS),
    (O.SKIPPED_NO_ALERTS, O.SKIPPED_NO_ALERTS, O.SKIPPED_NO_ALERTS),
])
def test_overall_is_the_worst_classifier(a, b, overall):
    result = O.overall_outcome({"decision_tree": {"verdict": a, "counts": None},
                                "random_forest": {"verdict": b, "counts": None}})
    assert result["verdict"] == overall and result["success"] is (overall == O.PASS)


def test_totals_add_up():
    r = O.overall_outcome({"decision_tree": O.classifier_outcome(doc(explained=2), OK, 5),
                           "random_forest": O.classifier_outcome(doc(abstained=1, explained=1), OK, 9)})
    assert r["totals"]["explained"] == 3 and r["totals"]["abstained"] == 1 and r["totals"]["skipped"] == 10


def test_counts_line_for_an_unprocessed_classifier():
    assert O.counts_line(None) == "not processed"


def test_an_unreached_classifier_still_counts_in_the_totals():
    unreached = O.classifier_outcome(None, None, 50, selected=6)
    assert unreached["verdict"] == O.INCOMPLETE and unreached["counts"]["incomplete"] == 6 and unreached["counts"]["skipped"] == 44
    r = O.overall_outcome({"decision_tree": O.classifier_outcome(doc(explained=2), OK, 5), "random_forest": unreached})
    assert r["totals"]["incomplete"] == 6 and r["totals"]["alerts_in_export"] == 55 and r["verdict"] == O.INCOMPLETE
