"""scripts/qwen_smoke.py must not say ALL PASS when answers were rejected or the run was not clean."""
import os
import subprocess
import sys
import tempfile

import pytest

from qwen_test_support import ROOT, chat_body, evidence_from_request, valid_answer

SCRIPT = ROOT / "scripts" / "qwen_smoke.py"
REJECTED = ("json", 200, chat_body({"observations": ["x"], "explanation": "This confirms a DoS attack beyond doubt.",
                                    "limitations": [], "evidence_used": [], "priority": None, "abstain": False}))


def answering(actions=()):
    queue = list(actions)

    def respond(body):
        if "ready" in body["messages"][-1]["content"]:
            return ("json", 200, chat_body({"ready": True}))
        return queue.pop(0) if queue else ("json", 200, chat_body(valid_answer(evidence_from_request(body))))

    return respond


def smoke(fake, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake.url, PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory() as out_dir:  # never the repository's outputs/: these answers come from a fake
        return subprocess.run([sys.executable, str(SCRIPT), "--live-only", "--out-dir", out_dir, *map(str, args)], cwd=ROOT,
                              env=env, capture_output=True, text=True, timeout=180)


def test_clean_live_smoke_passes_and_shows_every_count(fake_ollama):
    fake_ollama.default_chat = answering()
    r = smoke(fake_ollama)
    assert r.returncode == 0, r.stdout
    assert "ALL PASS" in r.stdout and "explained=2 abstained=0 rejected=0 failed=0 incomplete=0 skipped=3" in r.stdout


def test_one_rejected_answer_is_not_all_pass(fake_ollama):
    fake_ollama.default_chat = answering([REJECTED, REJECTED])
    r = smoke(fake_ollama)
    assert r.returncode == 1 and "ALL PASS" not in r.stdout
    assert "FAIL  decision_tree: no unresolved rejected or failed answers" in r.stdout
    assert "REJECTED_ANSWERS" in r.stdout


def test_technical_failure_is_not_all_pass(fake_ollama):
    fake_ollama.default_chat = answering([("json", 400, {"error": "bad"})])
    r = smoke(fake_ollama)
    assert r.returncode == 1 and "ALL PASS" not in r.stdout
    assert "decision_tree: FAILED (explained=1 abstained=0 rejected=0 failed=1" in r.stdout


def test_one_abstention_is_shown_separately_and_is_not_a_failure(fake_ollama):
    seen = []

    def abstain_first(body):
        if "ready" in body["messages"][-1]["content"]:
            return ("json", 200, chat_body({"ready": True}))
        seen.append(1)
        return ("json", 200, chat_body(valid_answer(evidence_from_request(body), abstain=len(seen) == 1)))

    fake_ollama.default_chat = abstain_first
    r = smoke(fake_ollama)
    assert r.returncode == 0, r.stdout
    assert "explained=1 abstained=1 rejected=0" in r.stdout and "ALL PASS" in r.stdout


def test_a_classifier_that_only_abstained_is_not_all_pass(fake_ollama):
    def abstain_all(body):
        if "ready" in body["messages"][-1]["content"]:
            return ("json", 200, chat_body({"ready": True}))
        return ("json", 200, chat_body(valid_answer(evidence_from_request(body), abstain=True)))

    fake_ollama.default_chat = abstain_all
    r = smoke(fake_ollama)
    assert r.returncode == 1 and "abstained=2" in r.stdout and "NO_EXPLANATIONS" in r.stdout and "ALL PASS" not in r.stdout


@pytest.mark.parametrize("limit", ["0", "-1", "26", "x"])
def test_invalid_limit_is_refused(fake_ollama, limit):
    r = smoke(fake_ollama, "--limit", limit)
    assert r.returncode == 2 and "--limit" in r.stderr and fake_ollama.requests == []


def test_digest_pinning_off_is_not_reported_as_verified(fake_ollama):
    fake_ollama.default_chat = answering()
    fake_ollama.model_entry["digest"] = "a" * 64
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1", NOVA_QWEN_EXPECTED_DIGEST="")
    with tempfile.TemporaryDirectory() as out_dir:
        r = subprocess.run([sys.executable, str(SCRIPT), "--live-only", "--out-dir", out_dir], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=180)
    assert r.returncode == 1 and "FAIL  Ollama reachable, pinned model present, digest verified" in r.stdout


def test_smoke_output_folder_is_respected(fake_ollama, tmp_path):
    fake_ollama.default_chat = answering()
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env.update(NOVA_OLLAMA_URL=fake_ollama.url, PYTHONDONTWRITEBYTECODE="1")
    before = {p: p.stat().st_mtime_ns for p in (ROOT / "outputs" / "qwen_smoke").glob("*")} \
        if (ROOT / "outputs" / "qwen_smoke").is_dir() else {}
    r = subprocess.run([sys.executable, str(SCRIPT), "--live-only", "--out-dir", tmp_path / "here"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout
    assert (tmp_path / "here" / "decision_tree_live_results.json").exists()
    after = {p: p.stat().st_mtime_ns for p in (ROOT / "outputs" / "qwen_smoke").glob("*")} \
        if (ROOT / "outputs" / "qwen_smoke").is_dir() else {}
    assert after == before
