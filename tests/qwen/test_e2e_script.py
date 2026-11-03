"""scripts/qwen_e2e_from_notebook.py, run as teammates run it, against a fake Ollama. Each refusal is asserted
to happen before any model request. Live and real-data runs are separate (tests/qwen_live, docs)."""
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from qwen_test_support import DT_FIXTURE, DT_EMPTY, RF_EMPTY, RF_FIXTURE, ROOT, chat_body, evidence_from_request, load_json, valid_answer
from src.nova_qwen import datasets as D

SCRIPT = ROOT / "scripts" / "qwen_e2e_from_notebook.py"
HEADER = list(D.V3_COLUMNS)
REJECTED = ("json", 200, chat_body({"observations": ["x"], "explanation": "This confirms a DoS attack beyond doubt.",
                                    "limitations": [], "evidence_used": [], "priority": None, "abstain": False}))


def abstaining(body):
    return ("json", 200, chat_body(valid_answer(evidence_from_request(body), abstain=True)))


def script(*args, fake=None, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if fake is not None:
        env["NOVA_OLLAMA_URL"] = fake.url
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.fixture
def exports(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    shutil.copy(DT_FIXTURE, folder / "decision_tree_alerts.json")
    shutil.copy(RF_FIXTURE, folder / "random_forest_alerts.json")
    return folder


def run_args(tmp_path, exports, *extra):
    return ("--skip-notebook", "--no-dataset-check", "--exports-dir", exports, "--out-dir", tmp_path / "out", *extra)


def last_line(r):
    return r.stdout.strip().splitlines()[-1]


def summary(tmp_path):
    return load_json(tmp_path / "out" / "summary.json")


def chats(fake):
    return len(fake.chat_requests())


@pytest.mark.parametrize("limit", ["-1", "-100", "0", "abc", "2.5", ""])
def test_invalid_limit_is_refused_before_any_request(fake_ollama, tmp_path, exports, limit):
    r = script(*run_args(tmp_path, exports, "--limit", limit), fake=fake_ollama)
    assert r.returncode == 2 and "--limit" in r.stderr
    assert fake_ollama.requests == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("limit", ["26", "100000"])
def test_large_limit_needs_explicit_authorization(fake_ollama, tmp_path, exports, limit):
    r = script(*run_args(tmp_path, exports, "--limit", limit), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "LARGE_RUN_NOT_AUTHORIZED"
    assert fake_ollama.requests == [] and "NOT RUN" in last_line(r)


def test_limit_at_the_boundary_is_allowed(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports, "--limit", "25"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert chats(fake_ollama) == 10  # the fixtures hold 5 alerts each


def test_alert_ids_count_toward_the_limit(fake_ollama, tmp_path, exports):
    ids = [x for i in range(3) for x in ("--alert-id", f"dt-{i:06d}")]
    r = script(*run_args(tmp_path, exports, "--limit", "24", *ids), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "LARGE_RUN_NOT_AUTHORIZED"
    assert fake_ollama.requests == []


def test_large_run_flag_authorizes_but_stays_within_the_export(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports, "--limit", "100000", "--allow-large-run", "--select", "first"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert chats(fake_ollama) == 10 and summary(tmp_path)["limits"]["large_run_allowed"] is True


def test_default_selection_is_bounded_on_a_large_export(fake_ollama, tmp_path, exports):
    """A negative or missing limit once meant nearly every alert; a big export must stay at the default sample."""
    doc = load_json(DT_FIXTURE)
    template = doc["alerts"][0]
    doc["alerts"] = [{**template, "alert_id": f"dt-{i:06d}", "record_id": f"val-{i:06d}"} for i in range(1, 501)]
    (exports / "decision_tree_alerts.json").write_text(json.dumps(doc))
    rf = load_json(RF_FIXTURE)
    rf["alerts"] = [{**rf["alerts"][0], "alert_id": f"rf-{i:06d}", "record_id": f"val-{i:06d}"} for i in range(1, 501)]
    (exports / "random_forest_alerts.json").write_text(json.dumps(rf))
    r = script(*run_args(tmp_path, exports), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert chats(fake_ollama) == 12
    s = summary(tmp_path)["classifiers"]["decision_tree"]["outcome"]["counts"]
    assert s["selected"] == 6 and s["skipped"] == 494 and s["alerts_in_export"] == 500


def test_clean_run_is_pass_and_records_everything(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert last_line(r) == "E2E PASS (NOT a verified V3 run)"
    s = summary(tmp_path)
    assert s["report"] == "nova-qwen-e2e-summary" and s["outcome"]["verdict"] == "PASS"
    assert s["dataset"]["declared_version"] == "V3" and s["dataset"]["real_v3_verified"] is False
    assert s["dataset"]["v3_profile_matched"] is False
    assert s["dataset"]["alignment"] == "NOT CHECKED" and s["dataset"]["export_dataset_field"]["decision_tree"] == "CICIDS2017"
    assert "CICIDS2017-V3" in s["dataset"]["field_note"] and "pending team approval" in s["dataset"]["field_note"]
    assert "No dataset file was read" in s["dataset"]["field_note"] and "from the files" not in s["dataset"]["field_note"]
    assert s["paths"]["exports_dir"] == str(exports) and s["paths"]["validation"] is None and s["paths"]["cwd"]
    assert "microseconds" in s["unit_assumption"] and "false negative" in s["scope_note"]
    assert s["correlation_counts"] and s["review_worksheet"] == "review_worksheet.csv"
    for det in ("decision_tree", "random_forest"):
        assert load_json(tmp_path / "out" / f"{det}_explanations.json")["provenance"]["verified"] is False
    assert (tmp_path / "out" / "review_worksheet.rubric.txt").exists()


def test_rejected_answers_are_not_reported_as_success(fake_ollama, tmp_path, exports):
    fake_ollama.default_chat = lambda body: REJECTED
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 1 and last_line(r).startswith("E2E REJECTED_ANSWERS")
    assert "PASS" not in r.stdout.splitlines()[-1]
    counts = summary(tmp_path)["classifiers"]["decision_tree"]["outcome"]["counts"]
    assert counts["rejected"] == 2 and counts["explained"] == 0
    assert "rejected and remain unresolved" in r.stdout


def test_one_rejected_answer_among_good_ones_is_not_success(fake_ollama, tmp_path, exports):
    fake_ollama.chat_actions = [REJECTED, REJECTED]  # first alert: answer and corrective re-ask both rejected
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 1 and last_line(r).startswith("E2E REJECTED_ANSWERS")
    s = summary(tmp_path)
    assert s["classifiers"]["decision_tree"]["outcome"]["verdict"] == "REJECTED_ANSWERS"
    assert s["classifiers"]["random_forest"]["outcome"]["verdict"] == "PASS"


def test_technical_failure_is_failed_not_rejected(fake_ollama, tmp_path, exports):
    fake_ollama.chat_actions = [("json", 400, {"error": "bad request"})]
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 1 and last_line(r).startswith("E2E FAILED")
    assert summary(tmp_path)["classifiers"]["decision_tree"]["outcome"]["counts"]["failed"] == 1


def test_abstentions_are_reported_separately_and_do_not_fail_a_run(fake_ollama, tmp_path, exports):
    answers = []

    def mixed(body):
        answers.append(1)
        return abstaining(body) if len(answers) % 2 else ("json", 200, chat_body(valid_answer(evidence_from_request(body))))

    fake_ollama.default_chat = mixed
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 0, r.stdout
    counts = summary(tmp_path)["outcome"]["totals"]
    assert counts["explained"] == 2 and counts["abstained"] == 2
    assert "abstained=1" in r.stdout


def test_a_run_that_explained_nothing_is_not_a_pass(fake_ollama, tmp_path, exports):
    fake_ollama.default_chat = abstaining
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 1 and last_line(r).startswith("E2E NO_EXPLANATIONS")


def test_empty_exports_skip_inference_and_are_not_a_pass(fake_ollama, tmp_path, exports):
    shutil.copy(DT_EMPTY, exports / "decision_tree_alerts.json")
    shutil.copy(RF_EMPTY, exports / "random_forest_alerts.json")
    r = script(*run_args(tmp_path, exports), fake=fake_ollama)
    assert r.returncode == 1 and last_line(r).startswith("E2E SKIPPED_NO_ALERTS")
    assert fake_ollama.requests == []


def test_ollama_unreachable_from_the_start_is_failed_with_exit_3_and_changes_nothing(fake_ollama, tmp_path, exports, unused_port):
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    before = {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()}

    class Down:
        url = f"http://127.0.0.1:{unused_port}"
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=Down)
    assert r.returncode == 3 and last_line(r) == "E2E FAILED: Ollama or model problem"
    assert {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()} == before  # earlier results and verdict stay together


def test_model_lost_between_classifiers_replaces_an_earlier_pass(fake_ollama, tmp_path, exports):
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    assert summary(tmp_path)["outcome"]["verdict"] == "PASS"
    answers = []

    def lose_it_after_decision_tree(body):
        answers.append(1)
        if len(answers) == 2:
            fake_ollama.version_action = ("drop",)
        return ("json", 200, chat_body(valid_answer(evidence_from_request(body))))

    fake_ollama.default_chat = lose_it_after_decision_tree
    r = script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama)
    assert r.returncode == 3, r.stdout + r.stderr
    assert last_line(r).startswith("E2E INCOMPLETE")
    s = summary(tmp_path)
    assert s["outcome"]["success"] is False and s["outcome"]["verdict"] == "INCOMPLETE"
    rf = s["classifiers"]["random_forest"]
    assert rf["outcome"]["verdict"] == "INCOMPLETE" and rf["results_file"] is None
    assert s["classifiers"]["decision_tree"]["outcome"]["verdict"] == "PASS"
    assert s["outcome"]["totals"]["incomplete"] == 2 and "earlier run" in s["note"]
    assert not (tmp_path / "out" / "correlation.json").exists()


def test_interrupt_leaves_no_earlier_pass_behind(fake_ollama, tmp_path, exports, sigint_deliverable):
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    fake_ollama.requests.clear()
    fake_ollama.default_chat = lambda body: ("sleep", 30, ("json", 200, chat_body(valid_answer(evidence_from_request(body)))))
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([sys.executable, str(SCRIPT), *map(str, run_args(tmp_path, exports, "--limit", "2"))], cwd=ROOT,
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while not fake_ollama.chat_requests() and time.time() < deadline:
            time.sleep(0.05)
        assert fake_ollama.chat_requests(), "the run never reached the model"
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=60)
    finally:
        p.kill()
    assert p.returncode == 130 and out.strip().splitlines()[-1] == "E2E INCOMPLETE (NOT a verified V3 run)"
    s = summary(tmp_path)
    assert s["outcome"]["verdict"] == "INCOMPLETE" and s["outcome"]["success"] is False


def test_an_interrupted_run_ends_with_the_verdict_its_summary_holds(fake_ollama, tmp_path, exports, sigint_deliverable):
    """One alert has already failed when Ctrl-C arrives, and a failure outranks incomplete."""
    fake_ollama.chat_actions = [("json", 400, {"error": "bad"})]
    fake_ollama.default_chat = lambda body: ("sleep", 30, ("json", 200, chat_body(valid_answer(evidence_from_request(body)))))
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([sys.executable, str(SCRIPT), *map(str, run_args(tmp_path, exports, "--limit", "3"))], cwd=ROOT,
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while len(fake_ollama.chat_requests()) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert len(fake_ollama.chat_requests()) >= 2, "the run never reached the second alert"
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=60)
    finally:
        p.kill()
    assert p.returncode == 130 and out.strip().splitlines()[-1] == "E2E FAILED (NOT a verified V3 run)"
    assert summary(tmp_path)["outcome"]["verdict"] == "FAILED"


def test_an_interrupt_during_the_final_verify_is_never_a_pass(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    """Every result can be good by then, but the correlation and the worksheet were skipped."""
    e2e = load_script()
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    real = e2e.verify_results

    def interrupted(doc, export, **kw):
        if export.detector == "random_forest" and not kw:
            raise KeyboardInterrupt
        return real(doc, export, **kw)

    monkeypatch.setattr(e2e, "verify_results", interrupted)
    assert e2e.entry([str(a) for a in run_args(tmp_path, exports, "--limit", "1")]) == 130
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E INCOMPLETE (NOT a verified V3 run)"
    s = summary(tmp_path)
    assert s["outcome"]["verdict"] == "INCOMPLETE" and s["outcome"]["success"] is False
    assert not (tmp_path / "out" / "review_worksheet.csv").exists()


def test_a_results_file_this_attempt_did_not_change_is_not_credited_to_it(fake_ollama, tmp_path, exports):
    """A clock that moved must not attribute an earlier file to this run, so contents decide, not updated_utc."""
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    rf = tmp_path / "out" / "random_forest_explanations.json"
    doc = load_json(rf)
    doc["updated_utc"] = "2099-01-01T00:00:00.000000Z"
    rf.write_text(json.dumps(doc))
    answers = []

    def lose_it_after_decision_tree(body):
        answers.append(1)
        if len(answers) == 2:
            fake_ollama.version_action = ("drop",)
        return ("json", 200, chat_body(valid_answer(evidence_from_request(body))))

    fake_ollama.default_chat = lose_it_after_decision_tree
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 3
    s = summary(tmp_path)
    assert s["classifiers"]["decision_tree"]["results_file"] is not None
    assert s["classifiers"]["random_forest"]["results_file"] is None
    assert s["classifiers"]["random_forest"]["outcome"]["reasons"] == ["this classifier was not processed"]


def test_ctrl_c_before_anything_changed_says_so_and_changes_nothing(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    e2e = load_script()
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    args = [str(a) for a in run_args(tmp_path, exports, "--limit", "1")]
    assert e2e.entry(args) == 0
    before = {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()}
    capsys.readouterr()
    e2e.STARTED = False

    def pressed(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(e2e, "check_alignment", pressed)
    monkeypatch.setattr(e2e, "load_export", pressed)
    assert e2e.entry(args) == 130
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E NOT RUN: interrupted before anything changed"
    assert {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()} == before


def test_an_unexpected_error_still_ends_with_a_verdict_line(monkeypatch, capsys):
    e2e = load_script()

    def broken(argv=None):
        raise AttributeError("'str' object has no attribute 'get'")

    monkeypatch.setattr(e2e, "main", broken)
    assert e2e.entry([]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines()[-1] == "E2E FAILED: unexpected error"
    assert json.loads(captured.err)["error"]["code"] == "UNEXPECTED_ERROR"


def test_a_killed_run_leaves_no_earlier_pass_behind(fake_ollama, tmp_path, exports):
    """Nothing can run after SIGKILL (for example the out-of-memory killer), so the summary is replaced at the start."""
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    fake_ollama.requests.clear()
    fake_ollama.default_chat = lambda body: ("sleep", 30, ("json", 200, chat_body(valid_answer(evidence_from_request(body)))))
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([sys.executable, str(SCRIPT), *map(str, run_args(tmp_path, exports, "--limit", "2"))], cwd=ROOT,
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while not fake_ollama.chat_requests() and time.time() < deadline:
        time.sleep(0.05)
    assert fake_ollama.chat_requests(), "the run never reached the model"
    p.kill()
    p.wait(timeout=30)
    s = summary(tmp_path)
    assert s["outcome"]["verdict"] == "INCOMPLETE" and s["outcome"]["success"] is False and "has not reached" in s["note"]


def test_a_refused_run_changes_nothing(fake_ollama, tmp_path, exports):
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    before = {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()}
    r = script(*run_args(tmp_path, exports, "--alert-id", "rf-999999"), fake=fake_ollama)
    assert r.returncode == 2 and {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()} == before


def test_transport_failures_on_a_short_run_still_exit_3(fake_ollama, tmp_path, exports):
    fake_ollama.default_chat = lambda body: ("drop",)
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 3 and last_line(r).startswith("E2E FAILED"), r.stdout + r.stderr
    assert summary(tmp_path)["outcome"]["totals"]["failed"] == 2


def test_resume_reuses_good_answers(fake_ollama, tmp_path, exports):
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    fake_ollama.requests.clear()
    r = script(*run_args(tmp_path, exports, "--limit", "2", "--resume"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert chats(fake_ollama) == 0
    assert load_json(tmp_path / "out" / "decision_tree_explanations.json")["resumed_from"]["reused_results"] == 2


def test_resume_finishes_an_interrupted_run_and_records_what_it_established(fake_ollama, tmp_path, exports):
    common = ("--skip-notebook", "--exports-dir", exports, "--out-dir", tmp_path / "out", "--limit", "5")
    checked = ("--dataset-version", "SYNTHETIC-DEMO", "--validation", csv_for_fixtures(tmp_path / "v.csv"))
    results = tmp_path / "out" / "decision_tree_explanations.json"
    fake_ollama.default_chat = lambda body: ("json", 503, {"error": "model runner has unexpectedly stopped"})
    assert script(*common, *checked, fake=fake_ollama).returncode == 3
    first = load_json(results)
    assert first["provenance"]["validation_sha256"]
    fake_ollama.default_chat = lambda body: ("json", 200, chat_body(valid_answer(evidence_from_request(body))))
    r = script(*common, "--dataset-version", "OTHER", "--no-dataset-check", "--resume", fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESUME_MISMATCH", r.stdout + r.stderr
    # The same version with no check: the label says what this run established, and the earlier identity stays beside it.
    r = script(*common, "--dataset-version", "SYNTHETIC-DEMO", "--no-dataset-check", "--resume", fake=fake_ollama)
    assert r.returncode == 0, r.stdout + r.stderr
    assert summary(tmp_path)["outcome"]["verdict"] == "PASS"
    doc = load_json(results)
    assert doc["provenance"]["validation_sha256"] is None
    assert doc["provenance"]["previous_provenance"]["validation_sha256"] == first["provenance"]["validation_sha256"]
    assert doc["resumed_from"]["previous_updated_utc"] == first["updated_utc"]


@pytest.mark.parametrize("change", ["limit", "export", "second_export", "malformed", "second_malformed", "seed"])
def test_a_refused_resume_changes_nothing(fake_ollama, tmp_path, exports, change):
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    out = tmp_path / "out"
    fake_ollama.requests.clear()
    limit, env, code = "2", {}, "RESUME_MISMATCH"
    if change == "limit":
        limit = "3"
    elif change in ("export", "second_export"):
        path = exports / ("decision_tree_alerts.json" if change == "export" else "random_forest_alerts.json")
        path.write_text(json.dumps(load_json(path), indent=3))
    elif change in ("malformed", "second_malformed"):
        path = out / ("decision_tree_explanations.json" if change == "malformed" else "random_forest_explanations.json")
        doc = load_json(path)
        doc["provenance"] = "x"
        path.write_text(json.dumps(doc))
        code = "RESULTS_MALFORMED"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    if change == "seed":
        e = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
        e.update(NOVA_OLLAMA_URL=fake_ollama.url, NOVA_QWEN_SEED="7", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, str(SCRIPT), *map(str, run_args(tmp_path, exports, "--limit", "2", "--resume"))],
                           cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    else:
        r = script(*run_args(tmp_path, exports, "--limit", limit, "--resume"), fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E NOT RUN: invalid input", r.stdout + r.stderr
    assert json.loads(r.stderr)["error"]["code"] == code
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before and chats(fake_ollama) == 0


def test_a_stale_second_results_file_is_named_and_can_be_moved_aside(fake_ollama, tmp_path, exports):
    """A real run stops during the first classifier, so the second file still holds an older trial's selection."""
    assert script(*run_args(tmp_path, exports, "--limit", "2"), fake=fake_ollama).returncode == 0
    answer = fake_ollama.default_chat
    fake_ollama.default_chat = lambda body: ("json", 503, {"error": "model runner has unexpectedly stopped"})
    assert script(*run_args(tmp_path, exports, "--limit", "5"), fake=fake_ollama).returncode == 3
    fake_ollama.default_chat = answer
    fake_ollama.requests.clear()
    r = script(*run_args(tmp_path, exports, "--limit", "5", "--resume"), fake=fake_ollama)
    assert r.returncode == 2 and chats(fake_ollama) == 0, r.stdout + r.stderr
    message = json.loads(r.stderr)["error"]["message"]
    assert "random_forest" in message and "move it out of --out-dir" in message and "without selection flags" not in message
    (tmp_path / "out" / "random_forest_explanations.json").rename(tmp_path / "older_trial.json")
    r = script(*run_args(tmp_path, exports, "--limit", "5", "--resume"), fake=fake_ollama)
    assert r.returncode == 0, r.stdout + r.stderr
    assert summary(tmp_path)["outcome"]["verdict"] == "PASS"


def test_run_aborted_keeps_partial_results_and_reports_them(fake_ollama, tmp_path, exports):
    fake_ollama.default_chat = lambda body: ("json", 503, {"error": "model runner has unexpectedly stopped"})
    r = script(*run_args(tmp_path, exports, "--limit", "5"), fake=fake_ollama)
    assert r.returncode == 3, r.stdout + r.stderr
    s = summary(tmp_path)
    assert s["outcome"]["success"] is False and s["outcome"]["verdict"] == "FAILED"
    assert s["classifiers"]["random_forest"]["outcome"]["verdict"] == "INCOMPLETE"
    assert load_json(tmp_path / "out" / "decision_tree_explanations.json")["run_status"] == "interrupted"
    assert not (tmp_path / "out" / "review_worksheet.csv").exists()


@pytest.mark.parametrize("bad", ["xx-000001", "000001"])
def test_alert_id_with_an_unknown_prefix_is_refused(fake_ollama, tmp_path, exports, bad):
    r = script(*run_args(tmp_path, exports, "--alert-id", bad), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "BAD_SELECTION"
    assert fake_ollama.requests == []


def test_alert_id_missing_from_the_export_is_refused_before_inference(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports, "--alert-id", "rf-999999"), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "ALERT_ID_NOT_FOUND"
    assert fake_ollama.requests == []


def test_requested_alert_is_added_and_verified(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports, "--limit", "1", "--select", "first", "--alert-id", "dt-000008"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    ids = [x["alert_id"] for x in load_json(tmp_path / "out" / "decision_tree_explanations.json")["results"]]
    assert ids == ["dt-000002", "dt-000008"]


def test_existing_worksheet_is_never_overwritten(fake_ollama, tmp_path, exports):
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    sheet = tmp_path / "out" / "review_worksheet.csv"
    sheet.write_text(sheet.read_text() + "human scores\n")
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    assert sheet.read_text().endswith("human scores\n")
    assert summary(tmp_path)["review_worksheet"].startswith("review_worksheet_")


def test_missing_v3_is_blocked_and_never_falls_back_to_old_files(fake_ollama, tmp_path, exports):
    data = tmp_path / "data"
    data.mkdir()
    for old in ("training_dataset_CIC_70.csv", "validation_dataset_CIC_15.csv"):
        (data / old).write_text("old data")
    r = script("--skip-notebook", "--data-dir", data, "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E BLOCKED: missing input"
    err = json.loads(r.stderr)["error"]
    assert err["code"] == "DATASET_MISSING" and D.V3_VALIDATION_FILE in err["message"]
    assert "No other dataset is substituted" in err["message"]
    assert fake_ollama.requests == []


def test_missing_export_is_blocked(fake_ollama, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    r = script("--skip-notebook", "--no-dataset-check", "--exports-dir", empty, "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E BLOCKED: missing input"
    assert json.loads(r.stderr)["error"]["code"] == "EXPORT_MISSING" and fake_ollama.requests == []


@pytest.mark.parametrize("flag", ["--validation", "--train"])
def test_reserved_split_is_refused_on_either_argument(fake_ollama, tmp_path, exports, flag):
    reserved = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    reserved.write_text("must never be read")
    r = script("--skip-notebook", flag, reserved, "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESERVED_SPLIT_REFUSED"
    assert fake_ollama.requests == []


def test_named_training_file_that_does_not_exist_is_blocked(fake_ollama, tmp_path, exports):
    csv_path = csv_for_fixtures(tmp_path / "v.csv")
    r = script("--skip-notebook", "--dataset-version", "X", "--validation", csv_path, "--train", tmp_path / "typo.csv",
               "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E BLOCKED: missing input" and fake_ollama.requests == []


def test_expected_hash_with_no_file_to_compare_is_refused(fake_ollama, tmp_path, exports):
    csv_path = csv_for_fixtures(tmp_path / "v.csv")
    r = script("--skip-notebook", "--dataset-version", "X", "--validation", csv_path, "--expected-train-sha256", "0" * 64,
               "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and "no train file" in r.stderr and fake_ollama.requests == []


def test_wrong_published_hash_stops_the_run(fake_ollama, tmp_path, exports):
    csv_path = csv_for_fixtures(tmp_path / "v.csv")
    r = script("--skip-notebook", "--dataset-version", "X", "--validation", csv_path, "--expected-validation-sha256", "0" * 64,
               "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and "sha256" in r.stderr and fake_ollama.requests == []


def test_out_dir_that_cannot_be_used_is_refused_first(fake_ollama, tmp_path, exports):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    r = script("--skip-notebook", "--no-dataset-check", "--exports-dir", exports, "--out-dir", blocker, fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E NOT RUN: invalid input" and fake_ollama.requests == []


@pytest.mark.parametrize("name", ["summary.json", "correlation.json", "decision_tree_explanations.json"])
def test_foreign_files_in_the_out_dir_are_not_replaced(fake_ollama, tmp_path, exports, name):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / name).write_text("my own notes")
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_NOT_OURS"
    assert (tmp_path / "out" / name).read_text() == "my own notes" and chats(fake_ollama) == 0


def test_a_foreign_rubric_is_kept_and_the_run_writes_its_own_pair_beside_it(fake_ollama, tmp_path, exports):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "review_worksheet.rubric.txt").write_text("my own notes")
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    assert (tmp_path / "out" / "review_worksheet.rubric.txt").read_text() == "my own notes"
    sheet = summary(tmp_path)["review_worksheet"]
    assert sheet != "review_worksheet.csv" and (tmp_path / "out" / sheet).with_suffix(".rubric.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="needs symlinks")
def test_a_rubric_symlink_is_replaced_not_written_through(fake_ollama, tmp_path, exports):
    (tmp_path / "out").mkdir()
    elsewhere = tmp_path / "elsewhere.txt"
    (tmp_path / "out" / "review_worksheet.csv").symlink_to(tmp_path / "absent_sheet.csv")
    (tmp_path / "out" / "review_worksheet.rubric.txt").symlink_to(elsewhere)  # both dangling, so neither "exists"
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    assert not elsewhere.exists() and not (tmp_path / "absent_sheet.csv").exists()


def test_exports_from_different_runs_are_refused_before_inference(fake_ollama, tmp_path, exports):
    doc = load_json(exports / "random_forest_alerts.json")
    doc["run_id"] = "validation-20261031T000000000000Z"
    (exports / "random_forest_alerts.json").write_text(json.dumps(doc))
    r = script(*run_args(tmp_path, exports), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "INCOMPATIBLE_EXPORTS"
    assert fake_ollama.requests == [] and last_line(r) == "E2E NOT RUN: invalid input"


def test_swapped_exports_are_refused_before_inference(fake_ollama, tmp_path, exports):
    shutil.copy(RF_FIXTURE, exports / "decision_tree_alerts.json")
    shutil.copy(DT_FIXTURE, exports / "random_forest_alerts.json")
    r = script(*run_args(tmp_path, exports), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "CONTRACT_DETECTOR_MISMATCH"
    assert fake_ollama.requests == []


@pytest.mark.parametrize("select, expected", [("first", ["dt-000002", "dt-000004", "dt-000005"]),
                                              ("representative", ["dt-000002", "dt-000004", "dt-000008"])])
def test_select_decides_which_alerts_are_sent(fake_ollama, tmp_path, exports, select, expected):
    r = script(*run_args(tmp_path, exports, "--limit", "3", "--select", select), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert [x["alert_id"] for x in load_json(tmp_path / "out" / "decision_tree_explanations.json")["results"]] == expected


def csv_for_fixtures(path, *, shift=0):
    """A CSV whose rows match the fixtures' evidence (both classifiers come from the same frame)."""
    evidence = {}
    for fixture in (DT_FIXTURE, RF_FIXTURE):
        for a in load_json(fixture)["alerts"]:
            evidence[a["record_id"]] = a["evidence"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for position in range(1, 9):
            values = dict.fromkeys(HEADER, "0")
            values[D.LABEL_COLUMN] = "Normal Traffic"
            ev = evidence.get(f"val-{position - shift:06d}", {"Flow Duration": 11, "Total Fwd Packets": 2, "Flow Bytes/s": 3.5})
            values.update({k: str(v) for k, v in ev.items()})
            w.writerow([values[c] for c in HEADER])
    return path


def test_synthetic_dataset_runs_and_is_recorded_as_not_v3(fake_ollama, tmp_path, exports):
    csv_path = csv_for_fixtures(tmp_path / "synthetic_validation.csv")
    r = script("--skip-notebook", "--dataset-version", "SYNTHETIC-DEMO", "--validation", csv_path, "--exports-dir", exports,
               "--out-dir", tmp_path / "out", "--limit", "2", fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert "This run is NOT a V3 result" in r.stdout and last_line(r) == "E2E PASS (NOT a verified V3 run)"
    s = summary(tmp_path)
    assert s["dataset"]["declared_version"] == "SYNTHETIC-DEMO" and s["dataset"]["real_v3_verified"] is False
    assert s["dataset"]["alignment"]["decision_tree"]["passed"] and s["dataset"]["file_check"]["validation"]["path"] == str(csv_path)
    doc = load_json(tmp_path / "out" / "decision_tree_explanations.json")
    assert doc["provenance"]["dataset_version"] == "SYNTHETIC-DEMO" and doc["provenance"]["validation_sha256"]
    with open(tmp_path / "out" / "review_worksheet.csv") as fh:
        row = next(csv.DictReader(fh))
    assert row["dataset_version"] == "SYNTHETIC-DEMO" and row["dataset_verified"] == "False"


def test_a_file_named_like_v3_that_is_not_v3_is_refused(fake_ollama, tmp_path, exports):
    data = tmp_path / "data"
    data.mkdir()
    csv_for_fixtures(data / D.V3_VALIDATION_FILE)
    r = script("--skip-notebook", "--data-dir", data, "--exports-dir", exports, "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and "dataset check failed" in json.loads(r.stderr)["error"]["message"]
    assert fake_ollama.requests == [] and last_line(r) == "E2E NOT RUN: invalid input"


def test_exports_not_matching_the_validation_rows_are_refused_before_inference(fake_ollama, tmp_path, exports):
    csv_path = csv_for_fixtures(tmp_path / "shifted.csv", shift=1)  # as if the export came from a filtered frame
    r = script("--skip-notebook", "--dataset-version", "X", "--validation", csv_path, "--exports-dir", exports,
               "--out-dir", tmp_path / "out", fake=fake_ollama)
    assert r.returncode == 2 and "does not match" in json.loads(r.stderr)["error"]["message"]
    assert fake_ollama.requests == []


def test_exports_that_say_v3_cannot_be_declared_something_else(fake_ollama, tmp_path, exports):
    for name in ("decision_tree_alerts.json", "random_forest_alerts.json"):
        doc = load_json(exports / name)
        doc["dataset"] = "CICIDS2017-V3"
        (exports / name).write_text(json.dumps(doc))
    csv_path = csv_for_fixtures(tmp_path / "v.csv")
    r = script("--skip-notebook", "--dataset-version", "OTHER", "--validation", csv_path, "--exports-dir", exports,
               "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and "Refusing to mix" in json.loads(r.stderr)["error"]["message"]


def test_exports_naming_another_version_cannot_be_declared_v3(fake_ollama, tmp_path, exports):
    for name in ("decision_tree_alerts.json", "random_forest_alerts.json"):
        doc = load_json(exports / name)
        doc["dataset"] = "CICIDS2017-V2"
        (exports / name).write_text(json.dumps(doc))
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 2 and "Refusing to mix" in json.loads(r.stderr)["error"]["message"]
    assert fake_ollama.requests == []


def test_the_recommended_dataset_value_needs_no_change_to_the_consumer(fake_ollama, tmp_path, exports):
    for name in ("decision_tree_alerts.json", "random_forest_alerts.json"):
        doc = load_json(exports / name)
        doc["dataset"] = "CICIDS2017-V3"
        (exports / name).write_text(json.dumps(doc))
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    assert summary(tmp_path)["classifiers"]["decision_tree"]["export"]["dataset"] == "CICIDS2017-V3"


@pytest.mark.parametrize("args, code", [
    (("--skip-notebook", "--notebook-only"), "BAD_ARGUMENTS"),
    (("--no-dataset-check",), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--no-dataset-check", "--validation", "v.csv"), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--data-dir", "d", "--validation", "v.csv"), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--dataset-version", "v3"), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--dataset-version", " "), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--dataset-version", "V3 "), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--dataset-version", "CICIDS2017-V3"), "BAD_ARGUMENTS"),
    (("--resume",), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--no-dataset-check", "--expected-validation-sha256", "abc"), "BAD_ARGUMENTS"),
    (("--skip-notebook", "--dataset-version", "OTHER", "--data-dir", "d"), "BAD_ARGUMENTS"),
    (("--exports-dir", "somewhere"), "BAD_ARGUMENTS"),
])
def test_argument_conflicts_are_refused(fake_ollama, tmp_path, args, code):
    r = script(*args, "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == code
    assert fake_ollama.requests == []


def test_no_dependencies_needed_to_consume_exports(fake_ollama, tmp_path, exports):
    """-S and removing PYTHONPATH exclude site-packages: consuming exports must work with the standard library alone."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_") and k != "PYTHONPATH"}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-S", str(SCRIPT), *map(str, run_args(tmp_path, exports, "--limit", "1"))],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_verified_v3_branch_with_a_narrowed_profile(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    """A rehearsal, not real V3 verification: the V3 profile is narrowed to this small file, so the reporting branch
    that real V3 files will reach is exercised. With a matching published hash it is verified; without, a match only."""
    e2e = load_script()
    for name, value in (("V3_VALIDATION_ROWS", 8), ("V3_TRAIN_ROWS", 8)):
        monkeypatch.setattr(D, name, value)
    for name in ("V3_VALIDATION_CLASSES", "V3_TRAIN_CLASSES"):
        monkeypatch.setattr(D, name, {"Normal Traffic": 8})
    monkeypatch.setattr(e2e, "V3_VALIDATION_ROWS", 8)
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    data = tmp_path / "data"
    data.mkdir()
    validation = csv_for_fixtures(data / D.V3_VALIDATION_FILE)
    train = csv_for_fixtures(data / D.V3_TRAIN_FILE)
    base = ["--skip-notebook", "--data-dir", str(data), "--exports-dir", str(exports), "--limit", "2"]

    assert e2e.main([*base, "--out-dir", str(tmp_path / "unpinned")]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E PASS (V3 profile matched; file hash not pinned)"
    s = load_json(tmp_path / "unpinned" / "summary.json")
    assert s["dataset"]["v3_profile_matched"] is True and s["dataset"]["real_v3_verified"] is False
    assert s["dataset"]["verification_basis"]["training_file_checked"] is True
    assert s["paths"]["train"] == str(train) and "from the files" in s["dataset"]["field_note"]

    out = tmp_path / "pinned"
    assert e2e.main([*base, "--out-dir", str(out), "--expected-validation-sha256", D.sha256_file(validation)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E PASS"
    s = load_json(out / "summary.json")
    assert s["dataset"]["real_v3_verified"] is True and s["dataset"]["verification_basis"]["validation_hash_pinned"] is True
    assert load_json(out / "decision_tree_explanations.json")["provenance"]["verified"] is True
    with open(out / "review_worksheet.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["dataset_version"] == "V3" and r["dataset_verified"] == "True" for r in rows)


def narrowed_v3(monkeypatch, e2e, rows=8):
    """A rehearsal: V3's row and class counts are narrowed so a small CSV can stand in for the real file."""
    for name, value in (("V3_VALIDATION_ROWS", rows), ("V3_TRAIN_ROWS", rows)):
        monkeypatch.setattr(D, name, value)
    monkeypatch.setattr(D, "V3_VALIDATION_CLASSES", {"Normal Traffic": rows})
    monkeypatch.setattr(D, "V3_TRAIN_CLASSES", {"Normal Traffic": rows})
    monkeypatch.setattr(e2e, "V3_VALIDATION_ROWS", rows)


def test_results_files_never_claim_more_than_the_run_that_wrote_them(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    e2e = load_script()
    narrowed_v3(monkeypatch, e2e)
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    data = tmp_path / "data"
    data.mkdir()
    validation = csv_for_fixtures(data / D.V3_VALIDATION_FILE)
    pin = ("--expected-validation-sha256", D.sha256_file(validation))
    out = tmp_path / "o"
    results = out / "decision_tree_explanations.json"
    base = ["--skip-notebook", "--exports-dir", str(exports), "--out-dir", str(out), "--limit", "2"]
    checked = ["--data-dir", str(data)]

    def sheet():
        with open(out / load_json(out / "summary.json")["review_worksheet"]) as fh:
            return {r["dataset_verified"] for r in csv.DictReader(fh)}

    assert e2e.main([*base, *checked, *pin]) == 0
    assert load_json(results)["provenance"]["verified"] is True and sheet() == {"True"}

    # A resume with no dataset check says so, in the file and the worksheet, and keeps the earlier identity beside it.
    assert e2e.main([*base, "--no-dataset-check", "--resume"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E PASS (NOT a verified V3 run)"
    doc = load_json(results)["provenance"]
    assert doc["verified"] is False and doc["previous_provenance"]["verified"] is True and sheet() == {"False"}
    assert "previous_provenance" not in doc["previous_provenance"]

    # A profile match is recorded as one, and still points back at the stronger identity.
    assert e2e.main([*base, *checked, "--resume"]) == 0
    doc = load_json(results)["provenance"]
    assert doc["v3_profile_matched"] is True and doc["verified"] is False and doc["previous_provenance"]["verified"] is True

    # The hash again: verified, with nothing stronger left to keep.
    assert e2e.main([*base, *checked, *pin, "--resume"]) == 0
    doc = load_json(results)["provenance"]
    assert doc["verified"] is True and "previous_provenance" not in doc and sheet() == {"True"}


def test_validation_alone_does_not_read_a_training_file_nobody_named(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    e2e = load_script()
    narrowed_v3(monkeypatch, e2e)
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    default_folder = tmp_path / "processed"
    default_folder.mkdir()
    (default_folder / D.V3_TRAIN_FILE).write_text("an unrelated file that nobody asked about")
    monkeypatch.setattr(e2e, "DEFAULT_DATA_DIR", default_folder)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    validation = csv_for_fixtures(elsewhere / D.V3_VALIDATION_FILE)
    assert e2e.main(["--skip-notebook", "--validation", str(validation), "--exports-dir", str(exports),
                     "--out-dir", str(tmp_path / "o"), "--limit", "1"]) == 0
    s = load_json(tmp_path / "o" / "summary.json")
    assert s["paths"]["train"] is None and s["dataset"]["verification_basis"]["training_file_checked"] is False


def test_v3_export_with_an_alert_free_tail_is_refused_before_inference(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    """The row count is right, so only the V3 tail check can catch this."""
    e2e = load_script()
    narrowed_v3(monkeypatch, e2e, rows=400)
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    data = tmp_path / "data"
    data.mkdir()
    validation = csv_for_fixtures(data / D.V3_VALIDATION_FILE)
    lines = validation.read_text().splitlines()
    validation.write_text("\n".join(lines + lines[-1:] * 392) + "\n")  # 400 rows; the fixtures' alerts end by row 8
    code = e2e.entry(["--skip-notebook", "--validation", str(validation), "--exports-dir", str(exports),
                      "--out-dir", str(tmp_path / "o"), "--limit", "1"])
    captured = capsys.readouterr()
    assert code == 2 and captured.out.strip().splitlines()[-1] == "E2E NOT RUN: invalid input"
    assert "truncated" in captured.err and fake_ollama.requests == []


def test_v3_alignment_needs_alerts_to_compare(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    """Empty exports compare nothing, so even a matching file cannot give a verified result."""
    e2e = load_script()
    for name, value in (("V3_VALIDATION_ROWS", 8), ("V3_TRAIN_ROWS", 8)):
        monkeypatch.setattr(D, name, value)
    monkeypatch.setattr(D, "V3_VALIDATION_CLASSES", {"Normal Traffic": 8})
    monkeypatch.setattr(e2e, "V3_VALIDATION_ROWS", 8)
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    shutil.copy(DT_EMPTY, exports / "decision_tree_alerts.json")
    data = tmp_path / "data"
    data.mkdir()
    csv_for_fixtures(data / D.V3_VALIDATION_FILE)
    assert e2e.main(["--skip-notebook", "--data-dir", str(data), "--exports-dir", str(exports), "--out-dir", str(tmp_path / "o"),
                     "--limit", "1", "--expected-validation-sha256", D.sha256_file(data / D.V3_VALIDATION_FILE)]) == 1
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert "PASS" not in out and "(NOT a verified V3 run)" in out
    assert load_json(tmp_path / "o" / "summary.json")["dataset"]["real_v3_verified"] is False


# ---------------------------------------------------------------- notebook mode (no notebook is executed here)
def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_e2e_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen_e2e_script"] = module
    spec.loader.exec_module(module)
    return module


def notebook(path, *sources):
    cells = [{"cell_type": "code", "source": [s]} for s in sources]
    path.write_text(json.dumps({"cells": cells}))
    return path


def test_reads_the_paths_from_jenns_committed_notebook():
    e2e = load_script()
    train, validation = e2e.notebook_data_files(e2e.NOTEBOOK)
    assert train.name == D.V3_TRAIN_FILE and validation.name == D.V3_VALIDATION_FILE
    assert str(train.parent) == "/media/sf_NOVA_Data"


def test_notebook_with_a_relative_path_resolves_from_the_notebook(tmp_path):
    e2e = load_script()
    nb = notebook(tmp_path / "n.ipynb", 'DATA_DIR = Path("../data")\n"a_train.csv"\n"a_validation.csv"')
    train, validation = e2e.notebook_data_files(nb)
    assert train == (tmp_path / "../data").resolve() / "a_train.csv" and validation.name == "a_validation.csv"


@pytest.mark.parametrize("source, exc", [
    ('DATA_DIR = Path("d")\n"cicids2017_V3_test_RESERVED_15.csv"\n"a_train.csv"\n"a_validation.csv"', "RESERVED_SPLIT_REFUSED"),
    ('DATA_DIR = Path("d")\nDATA_DIR = Path("e")\n"a_train.csv"\n"a_validation.csv"', "DATASET_ERROR"),
    ('DATA_DIR = Path("d")\n"a_train.csv"\n"a_validation.csv"\n"other.csv"', "DATASET_ERROR"),
    ('"a_train.csv"\n"a_validation.csv"', "DATASET_ERROR"),
])
def test_notebook_data_files_stops_when_it_cannot_attribute_files(tmp_path, source, exc):
    e2e = load_script()
    with pytest.raises(e2e.NovaQwenError) as info:
        e2e.notebook_data_files(notebook(tmp_path / "n.ipynb", source))
    assert info.value.code == exc


def test_notebook_mode_blocks_when_its_data_path_does_not_exist(fake_ollama, tmp_path):
    nb = notebook(tmp_path / "n.ipynb", f'DATA_DIR = Path("{tmp_path}/absent")\n"{D.V3_TRAIN_FILE}"\n"{D.V3_VALIDATION_FILE}"')
    r = script("--notebook", nb, "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E BLOCKED: missing input"
    assert fake_ollama.requests == []


def test_notebook_mode_refuses_when_the_arguments_disagree_with_the_notebook(fake_ollama, tmp_path):
    nb = notebook(tmp_path / "n.ipynb", f'DATA_DIR = Path("{tmp_path}")\n"{D.V3_TRAIN_FILE}"\n"{D.V3_VALIDATION_FILE}"')
    r = script("--notebook", nb, "--validation", tmp_path / "elsewhere.csv", "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and "must match" in json.loads(r.stderr)["error"]["message"]


def test_v3_notebook_mode_refuses_a_notebook_reading_other_files(fake_ollama, tmp_path):
    nb = notebook(tmp_path / "n.ipynb", f'DATA_DIR = Path("{tmp_path}")\n"training_dataset_CIC_70.csv"\n"validation_dataset_CIC_15.csv"')
    r = script("--notebook", nb, "--out-dir", tmp_path / "o", fake=fake_ollama)
    assert r.returncode == 2 and "not the V3 files" in json.loads(r.stderr)["error"]["message"]


def test_stale_notebook_record_is_not_attributed_to_other_exports(fake_ollama, tmp_path, exports):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notebook_run.json").write_text(json.dumps({"exports_sha256": {"decision_tree": "0", "random_forest": "0"},
                                                       "data_files": {"old.csv": {"sha256": "STALE"}}}))
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 0, r.stderr
    s = summary(tmp_path)
    assert s["notebook"] is None and "different exports" in s["notebook_note"]


def test_matching_notebook_record_is_kept(fake_ollama, tmp_path, exports):
    e2e = load_script()
    out = tmp_path / "out"
    out.mkdir()
    recorded = {d: e2e.sha256(exports / f"{d}_alerts.json") for d in e2e.DETECTORS}
    (out / "notebook_run.json").write_text(json.dumps({"exports_sha256": recorded, "note": "kept"}))
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    s = summary(tmp_path)
    assert s["notebook"]["note"] == "kept" and "produced these exports" in s["notebook_note"]


def test_unreadable_notebook_record_is_ignored(fake_ollama, tmp_path, exports):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notebook_run.json").write_text("{not json")
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    assert summary(tmp_path)["notebook"] is None


class NotebookArgs:
    def __init__(self, nb, version="SYNTHETIC-DEMO"):
        self.notebook, self.dataset_version = nb, version


def notebook_files(tmp_path):
    nb = notebook(tmp_path / "n.ipynb", 'DATA_DIR = Path("d")\n"a_train.csv"\n"a_validation.csv"')
    train, validation = tmp_path / "a_train.csv", tmp_path / "a_validation.csv"
    train.write_text("x")
    validation.write_text("y")
    return nb, train, validation


def test_notebook_that_fails_is_a_failed_run_not_invalid_input(tmp_path, monkeypatch, capsys):
    e2e = load_script()
    nb, train, validation = notebook_files(tmp_path)
    monkeypatch.setattr(e2e, "DEFAULT_EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(e2e.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "kernel died"))
    with pytest.raises(e2e.NovaQwenError) as info:
        e2e.run_notebook(NotebookArgs(nb), train, validation, tmp_path / "o")
    assert info.value.code == "NOTEBOOK_FAILED" and "kernel died" in str(info.value)
    monkeypatch.setattr(e2e, "main", lambda argv=None: (_ for _ in ()).throw(info.value))
    assert e2e.entry() == 1 and capsys.readouterr().out.strip().splitlines()[-1] == "E2E FAILED: NOTEBOOK_FAILED"


def test_notebook_that_writes_no_new_exports_is_not_credited_with_old_ones(tmp_path, monkeypatch):
    e2e = load_script()
    nb, train, validation = notebook_files(tmp_path)
    exports = tmp_path / "exports"
    exports.mkdir()
    for det in e2e.DETECTORS:
        (exports / f"{det}_alerts.json").write_text("old")
    monkeypatch.setattr(e2e, "DEFAULT_EXPORTS", exports)
    monkeypatch.setattr(e2e.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    with pytest.raises(e2e.NovaQwenError) as info:
        e2e.run_notebook(NotebookArgs(nb), train, validation, tmp_path / "o")
    assert info.value.code == "NOTEBOOK_FAILED" and "did not write" in str(info.value)
    assert not (tmp_path / "o" / "notebook_run.json").exists()


def test_notebook_run_records_what_it_read_and_wrote(tmp_path, monkeypatch):
    e2e = load_script()
    nb, train, validation = notebook_files(tmp_path)
    exports = tmp_path / "exports"
    exports.mkdir()

    def fake_nbconvert(*a, **k):
        for det in e2e.DETECTORS:
            (exports / f"{det}_alerts.json").write_text(det)
        return subprocess.CompletedProcess(a, 0, "", "")

    monkeypatch.setattr(e2e, "DEFAULT_EXPORTS", exports)
    monkeypatch.setattr(e2e.subprocess, "run", fake_nbconvert)
    (tmp_path / "o").mkdir()
    info = e2e.run_notebook(NotebookArgs(nb), train, validation, tmp_path / "o")
    assert info["data_files"]["a_train.csv"]["sha256"] == e2e.sha256(train)
    assert info["exports_sha256"]["decision_tree"] == e2e.sha256(exports / "decision_tree_alerts.json")
    assert info["note"] == "Notebook executed as given."  # the caveat describes Jenn's notebook only
    assert load_json(tmp_path / "o" / "notebook_run.json")["notebook"]["sha256"] == e2e.sha256(nb)


def test_a_missing_notebook_record_does_not_claim_where_the_exports_came_from(fake_ollama, tmp_path, exports):
    assert script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama).returncode == 0
    s = summary(tmp_path)
    assert s["notebook"] is None and "not recorded" in s["notebook_note"] and "came from" not in s["notebook_note"]


@pytest.mark.skipif(os.name != "posix", reason="needs symlinks")
def test_a_notebook_record_pointing_at_a_reserved_file_is_refused(fake_ollama, tmp_path, exports):
    (tmp_path / "out").mkdir()
    reserved = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    reserved.write_text("must never be read")
    (tmp_path / "out" / "notebook_run.json").symlink_to(reserved)
    r = script(*run_args(tmp_path, exports, "--limit", "1"), fake=fake_ollama)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESERVED_SPLIT_REFUSED"
    assert chats(fake_ollama) == 0


@pytest.mark.parametrize("path", ["cicids2017_V3_test_RESERVED_15.csv", "/dev/stdin"])
def test_a_reserved_or_device_notebook_path_is_refused(tmp_path, path):
    e2e = load_script()
    with pytest.raises(e2e.ReservedSplitRefused):
        e2e.notebook_data_files(tmp_path / path if not path.startswith("/") else e2e.Path(path))


@pytest.fixture
def stub_notebook(tmp_path, monkeypatch):
    """A notebook with real paths, and an nbconvert stand-in that records each call and writes the exports."""
    e2e = load_script()
    nb, _, _ = notebook_files(tmp_path)
    (tmp_path / "d").mkdir()
    csv_for_fixtures(tmp_path / "d" / "a_train.csv")
    csv_for_fixtures(tmp_path / "d" / "a_validation.csv")
    exports = tmp_path / "exports"
    exports.mkdir()
    calls = []

    def fake_nbconvert(*a, **k):
        calls.append(a)
        shutil.copy(DT_FIXTURE, exports / "decision_tree_alerts.json")
        shutil.copy(RF_FIXTURE, exports / "random_forest_alerts.json")
        return subprocess.CompletedProcess(a, 0, "", "")

    monkeypatch.setattr(e2e, "DEFAULT_EXPORTS", exports)
    monkeypatch.setattr(e2e.subprocess, "run", fake_nbconvert)
    monkeypatch.setattr(e2e, "notebook_tools_installed", lambda: True)
    return e2e, nb, calls, exports, tmp_path / "o"


def test_a_notebook_that_cannot_start_keeps_the_earlier_verdict(stub_notebook, fake_ollama, monkeypatch, capsys):
    e2e, nb, calls, _, out = stub_notebook
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    args = ["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(out), "--limit", "1"]
    assert e2e.entry(args) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    calls.clear()
    capsys.readouterr()
    monkeypatch.setattr(e2e, "notebook_tools_installed", lambda: False)
    assert e2e.entry(args) == 2
    captured = capsys.readouterr()
    assert "NOTEBOOK_TOOLS_MISSING" in captured.err and captured.out.strip().splitlines()[-1] == "E2E NOT RUN: invalid input"
    assert calls == [] and {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("mode", [[], ["--notebook-only"]])
@pytest.mark.parametrize("name", ["summary.json", "correlation.json", "notebook_run.json", "03_executed_by_e2e.ipynb",
                                  "decision_tree_explanations.json"])
def test_foreign_files_are_refused_before_the_notebook_runs(stub_notebook, fake_ollama, monkeypatch, capsys, mode, name):
    e2e, nb, calls, _, out = stub_notebook
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    out.mkdir()
    (out / name).write_text("my own notes")
    code = e2e.entry(["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(out), *mode])
    assert code == 2 and calls == [] and "OUTPUT_NOT_OURS" in capsys.readouterr().err
    assert (out / name).read_text() == "my own notes" and sorted(p.name for p in out.iterdir()) == [name]


def test_notebook_mode_with_the_model_down_never_starts_the_notebook(stub_notebook, monkeypatch, unused_port, capsys):
    e2e, nb, calls, _, out = stub_notebook
    monkeypatch.setenv("NOVA_OLLAMA_URL", f"http://127.0.0.1:{unused_port}")
    code = e2e.entry(["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(out)])
    assert code == 3 and calls == []
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E FAILED: Ollama or model problem"


@pytest.mark.parametrize("mode", [[], ["--notebook-only"]])
def test_a_notebook_run_retires_an_earlier_verdict_even_when_it_fails(stub_notebook, fake_ollama, monkeypatch, capsys, mode):
    e2e, nb, calls, _, out = stub_notebook
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    args = ["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(out), "--limit", "1"]
    assert e2e.entry(args) == 0
    assert load_json(out / "summary.json")["outcome"]["verdict"] == "PASS" and (out / "correlation.json").exists()
    capsys.readouterr()
    monkeypatch.setattr(e2e.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "kernel died"))
    assert e2e.entry([*args, *mode]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E FAILED: NOTEBOOK_FAILED"
    s = load_json(out / "summary.json")
    assert s["outcome"]["verdict"] == "INCOMPLETE" and "notebook was started" in s["note"]
    assert not (out / "correlation.json").exists()


def test_a_refusal_before_the_notebook_keeps_the_earlier_verdict(stub_notebook, fake_ollama, monkeypatch, capsys):
    e2e, nb, calls, _, out = stub_notebook
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    args = ["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(out), "--limit", "1"]
    assert e2e.entry(args) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    calls.clear()
    assert e2e.entry([*args, "--expected-validation-sha256", "0" * 64]) == 2
    assert calls == [] and {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_notebook_only_prints_a_next_command_that_keeps_the_pins(stub_notebook, monkeypatch, capsys):
    e2e, nb, calls, _, out = stub_notebook
    pin = D.sha256_file(nb.parent / "d" / "a_validation.csv")
    train_pin = D.sha256_file(nb.parent / "d" / "a_train.csv")
    monkeypatch.chdir(nb.parent)
    code = e2e.entry(["--notebook", str(nb), "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", "o", "--notebook-only",
                      "--expected-validation-sha256", pin, "--expected-train-sha256", train_pin])
    printed = capsys.readouterr().out
    assert code == 0 and printed.strip().splitlines()[-1] == "NOTEBOOK DONE: no Qwen inference was run."
    line = next(ln for ln in printed.splitlines() if ln.startswith("Next, from the repository root: "))
    assert f"--expected-validation-sha256 {pin}" in line and f"--expected-train-sha256 {train_pin}" in line
    assert f"--out-dir {out.resolve()}" in line


@pytest.mark.skipif(os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs non-root POSIX")
def test_out_dir_that_exists_but_is_read_only_is_refused_first(fake_ollama, tmp_path, exports):
    out = tmp_path / "read_only"
    out.mkdir()
    out.chmod(0o555)
    try:
        r = script("--skip-notebook", "--no-dataset-check", "--exports-dir", exports, "--out-dir", out, fake=fake_ollama)
    finally:
        out.chmod(0o755)
    assert r.returncode == 2 and last_line(r) == "E2E NOT RUN: invalid input" and fake_ollama.requests == []
    assert "cannot be written" in json.loads(r.stderr)["error"]["message"]


def test_invalid_argument_ends_with_a_verdict_line(fake_ollama, tmp_path, exports):
    r = script(*run_args(tmp_path, exports, "--limit", "-1"), fake=fake_ollama)
    assert r.returncode == 2 and last_line(r) == "E2E NOT RUN: invalid input"


def test_ctrl_c_after_the_run_started_is_incomplete_not_a_refusal(fake_ollama, tmp_path, exports, monkeypatch, capsys):
    e2e = load_script()
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)

    def pressed(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(e2e, "correlate", pressed)
    assert e2e.entry([str(a) for a in run_args(tmp_path, exports, "--limit", "1")]) == 130
    assert capsys.readouterr().out.strip().splitlines()[-1] == "E2E INCOMPLETE"
    assert summary(tmp_path)["outcome"]["verdict"] == "INCOMPLETE"
