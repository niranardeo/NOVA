"""The command-line interface, run exactly as teammates run it: python -m src.nova_qwen ... from the
repository root, in a subprocess."""
import json
import os
import subprocess
import sys

import pytest

from qwen_test_support import DT_FIXTURE, MODEL, RF_FIXTURE, ROOT, DT_EMPTY, load_json, write_json


def cli(*args, env_extra=None, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOVA_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "src.nova_qwen", *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


def test_help_and_version():
    r = cli("--help")
    assert r.returncode == 0
    for sub in ("check-ollama", "validate-export", "explain", "verify", "correlate", "worksheet"):
        assert sub in r.stdout
    assert cli("--version").stdout.startswith("nova_qwen ")


def test_validate_export_ok():
    r = cli("validate-export", DT_FIXTURE, "--detector", "decision_tree")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["valid"] and out["alert_count"] == 5 and out["run_id"] == "validation-20261030T000000000000Z"


def test_validate_export_invalid_reports_code_and_exit_2(tmp_path):
    doc = load_json(DT_FIXTURE)
    doc["schema_version"] = "1.0"
    r = cli("validate-export", write_json(tmp_path / "old.json", doc))
    assert r.returncode == 2
    assert json.loads(r.stderr)["error"]["code"] == "UNSUPPORTED_SCHEMA_VERSION"


def test_explain_verify_worksheet_correlate_against_fake_ollama(fake_ollama, tmp_path):
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    dt_out, rf_out = tmp_path / "dt.json", tmp_path / "rf.json"
    r = cli("explain", DT_FIXTURE, "--out", dt_out, "--limit", "3", "--detector", "decision_tree", env_extra=env)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)["summary"]
    assert summary["explained"] == 3 and "[3/3] dt-000005 (Web Attacks): explained" in r.stderr
    r = cli("explain", RF_FIXTURE, "--out", rf_out, "--alert-id", "rf-000004", "--quiet", env_extra=env)
    assert r.returncode == 0 and r.stderr == ""
    r = cli("verify", dt_out, "--export", DT_FIXTURE)
    assert r.returncode == 0 and json.loads(r.stdout)["passed"]
    r = cli("verify", dt_out, "--export", RF_FIXTURE)
    assert r.returncode == 1 and not json.loads(r.stdout)["passed"]
    ws = tmp_path / "review" / "worksheet.csv"
    r = cli("worksheet", dt_out, rf_out, "--out", ws)
    assert r.returncode == 0 and json.loads(r.stdout)["rows"] == 4
    header = ws.read_text().splitlines()[0]
    assert "H1_faithful_to_evidence_0_2" in header and "automated_errors" in header
    assert ws.with_suffix(".rubric.txt").exists()
    rep = tmp_path / "corr.json"
    r = cli("correlate", "--dt-export", DT_FIXTURE, "--rf-export", RF_FIXTURE, "--dt-results", dt_out,
            "--rf-results", rf_out, "--out", rep)
    assert r.returncode == 0 and load_json(rep)["counts"]["both_different_class"] == 1


def test_explain_refuses_to_overwrite_without_flag(fake_ollama, tmp_path):
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    out = tmp_path / "r.json"
    assert cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", env_extra=env).returncode == 0
    r = cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", env_extra=env)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_EXISTS"
    assert cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", "--overwrite", env_extra=env).returncode == 0
    r = cli("explain", DT_FIXTURE, "--out", DT_FIXTURE, "--limit", "1", "--overwrite", env_extra=env)
    assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_IS_INPUT"


def test_explain_empty_export_needs_no_ollama(tmp_path, unused_port):
    out = tmp_path / "r.json"
    r = cli("explain", DT_EMPTY, "--out", out, env_extra={"NOVA_OLLAMA_URL": f"http://127.0.0.1:{unused_port}"})
    assert r.returncode == 0 and load_json(out)["results"] == []


def test_unavailable_ollama_exit_3(tmp_path, unused_port):
    env = {"NOVA_OLLAMA_URL": f"http://127.0.0.1:{unused_port}"}
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "r.json", "--limit", "1", "--retries", "0", env_extra=env)
    assert r.returncode == 3 and json.loads(r.stderr)["error"]["code"] == "OLLAMA_UNAVAILABLE"
    assert not (tmp_path / "r.json").exists()
    assert cli("check-ollama", env_extra=env).returncode == 3


def test_remote_ollama_refused_without_opt_in(tmp_path):
    r = cli("check-ollama", "--url", "http://203.0.113.10:11434")
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "CONFIG_REMOTE_HOST"


def test_check_ollama_with_inference(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, {"model": MODEL, "message": {"content": '{"ready": true}'},
                                                   "done": True, "done_reason": "stop"}))
    r = cli("check-ollama", "--inference", env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["model"]["digest_verified"] and out["inference_test"]["ok"]


def test_strict_mode_fails_on_rejections(fake_ollama, tmp_path):
    fake_ollama.default_chat = lambda b: ("json", 200, {"model": MODEL, "message": {"content": "not json"},
                                                        "done": True, "done_reason": "stop"})
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "a.json", "--limit", "1", env_extra=env)
    assert r.returncode == 0 and json.loads(r.stdout)["summary"]["rejected"] == 1
    r = cli("explain", DT_FIXTURE, "--out", tmp_path / "b.json", "--limit", "1", "--strict", env_extra=env)
    assert r.returncode == 1


def test_cli_works_from_another_working_directory(tmp_path):
    env = {"PYTHONPATH": str(ROOT)}
    r = cli("validate-export", DT_FIXTURE, env_extra=env, cwd=tmp_path)
    assert r.returncode == 0


@pytest.mark.parametrize("args", [["explain", str(DT_FIXTURE), "--out", "x.json", "--limit", "1", "--all"],
                                  ["explain", str(DT_FIXTURE)], ["unknown-command"]])
def test_argument_errors(args):
    assert cli(*args).returncode == 2
