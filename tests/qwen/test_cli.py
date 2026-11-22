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
    assert "explanation_accuracy_0_2" in header and "automated_errors" in header
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


def many_alerts(tmp_path, n=26):
    doc = load_json(DT_FIXTURE)
    doc["alerts"] = [{**doc["alerts"][0], "alert_id": f"dt-{i:06d}", "record_id": f"val-{i:06d}"} for i in range(1, n + 1)]
    return write_json(tmp_path / "many.json", doc)


@pytest.mark.parametrize("flags", [("--all",), ("--limit", "26")])
def test_explain_refuses_a_large_selection_before_contacting_the_model(fake_ollama, tmp_path, flags):
    r = cli("explain", many_alerts(tmp_path), "--out", tmp_path / "r.json", *flags, env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "LARGE_RUN_NOT_AUTHORIZED"
    assert fake_ollama.requests == [] and not (tmp_path / "r.json").exists()


def test_explain_allows_a_large_selection_when_authorized(fake_ollama, tmp_path):
    r = cli("explain", many_alerts(tmp_path), "--out", tmp_path / "r.json", "--all", "--allow-large-run", "--quiet",
            env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["summary"]["processed"] == 26


def test_explain_at_the_limit_needs_no_flag(fake_ollama, tmp_path):
    r = cli("explain", many_alerts(tmp_path, 25), "--out", tmp_path / "r.json", "--all", "--quiet",
            env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr


def test_resume_of_a_large_selection_needs_the_flag_too(fake_ollama, tmp_path):
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    export, out = many_alerts(tmp_path), tmp_path / "r.json"
    assert cli("explain", export, "--out", out, "--all", "--allow-large-run", "--quiet", env_extra=env).returncode == 0
    fake_ollama.requests.clear()
    r = cli("explain", export, "--out", out, "--resume", "--quiet", env_extra=env)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "LARGE_RUN_NOT_AUTHORIZED"
    assert fake_ollama.requests == []


def test_explain_records_a_declared_dataset_version_as_unverified(fake_ollama, tmp_path):
    out = tmp_path / "r.json"
    r = cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", "--dataset-version", "V3", "--quiet",
            env_extra={"NOVA_OLLAMA_URL": fake_ollama.url})
    assert r.returncode == 0, r.stderr
    assert load_json(out)["provenance"] == {"dataset_version": "V3", "verified": False}
    r = cli("worksheet", out, "--out", tmp_path / "w.csv")
    assert r.returncode == 0
    header, row = (tmp_path / "w.csv").read_text().splitlines()[:2]
    assert header.split(",")[3:5] == ["dataset_version", "dataset_verified"] and ",V3,False," in row


def test_results_without_provenance_stay_readable(fake_ollama, tmp_path):
    out = tmp_path / "r.json"
    assert cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", "--quiet",
               env_extra={"NOVA_OLLAMA_URL": fake_ollama.url}).returncode == 0
    doc = load_json(out)
    assert doc["provenance"] is None
    del doc["provenance"]  # a results file from version 0.1
    write_json(out, doc)
    assert cli("verify", out, "--export", DT_FIXTURE).returncode == 0
    assert cli("worksheet", out, "--out", tmp_path / "w.csv").returncode == 0
    assert ",not recorded,False," in (tmp_path / "w.csv").read_text()


def _validation_csv(path):
    import csv as _csv
    from src.nova_qwen import datasets as D
    by_row = {a["record_id"]: a["evidence"] for a in load_json(DT_FIXTURE)["alerts"]}
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(D.V3_COLUMNS)
        for position in range(1, 9):
            values = dict.fromkeys(D.V3_COLUMNS, "0")
            values[D.LABEL_COLUMN] = "Normal Traffic"
            values.update({k: str(v) for k, v in by_row.get(f"val-{position:06d}", {}).items()})
            w.writerow([values[c] for c in D.V3_COLUMNS])
    return path


def test_check_dataset_and_check_alignment_commands(tmp_path):
    path = _validation_csv(tmp_path / "v.csv")
    r = cli("check-dataset", "--validation", path, "--dataset-version", "X", "--out", tmp_path / "d.json")
    assert r.returncode == 0 and json.loads(r.stdout)["v3_validation_verified"] is False
    assert load_json(tmp_path / "d.json")["report"] == "nova-qwen-dataset-check"
    r = cli("check-dataset", "--validation", path)  # V3 is the default and this file is not V3
    assert r.returncode == 1 and json.loads(r.stdout)["problems"]
    r = cli("check-alignment", DT_FIXTURE, "--validation", path, "--dataset-version", "X", "--out", tmp_path / "a.json")
    assert r.returncode == 0 and json.loads(r.stdout)["alerts_checked"] == 5
    assert load_json(tmp_path / "a.json")["dataset_version"] == "X"
    r = cli("check-alignment", DT_FIXTURE, "--validation", path)
    assert r.returncode == 1 and "expected 377880" in r.stdout
    r = cli("check-dataset", "--validation", tmp_path / "absent.csv")
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "DATASET_MISSING"
    reserved = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    reserved.write_text("x")
    r = cli("check-alignment", DT_FIXTURE, "--validation", reserved)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESERVED_SPLIT_REFUSED"


RESERVED_NAME = "cicids2017_V3_test_RESERVED_15.csv"
DT, RF = str(DT_FIXTURE), str(RF_FIXTURE)


def _open_recorder(tmp_path):
    """chmod 000 proves nothing: root ignores it and the --out guards swallow the OSError. This works for any user."""
    folder, log = tmp_path / "hook", tmp_path / "opened.log"
    folder.mkdir()
    (folder / "sitecustomize.py").write_text(
        "import sys\n"
        f"def _h(event, a, name={RESERVED_NAME!r}, log={str(log)!r}):\n"
        "    if event == 'open' and str(a[0]).endswith(name):\n"
        "        with open(log, 'a') as f:\n"
        "            f.write(str(a[0]) + '\\n')\n"
        "sys.addaudithook(_h)\n")
    return str(folder), log


@pytest.mark.parametrize("args", [
    ("validate-export", "{r}"), ("check-alignment", "{r}", "--validation", "{v}", "--dataset-version", "X"),
    ("verify", "{r}", "--export", DT), ("worksheet", "{r}", "--out", "{o}"), ("explain", "{r}", "--out", "{o}"),
    ("check-dataset", "--validation", "{v}", "--dataset-version", "X", "--out", "{r}"), ("worksheet", DT, "--out", "{r}"),
    ("correlate", "--dt-export", "{r}", "--rf-export", RF), ("correlate", "--dt-export", DT, "--rf-export", "{r}"),
    ("correlate", "--dt-export", DT, "--rf-export", RF, "--dt-results", "{r}"),
    ("correlate", "--dt-export", DT, "--rf-export", RF, "--rf-results", "{r}"),
    ("correlate", "--dt-export", DT, "--rf-export", RF, "--out", "{r}"), ("verify", "{res}", "--export", "{r}"),
    ("check-dataset", "--validation", "{r}"), ("check-dataset", "--validation", "{v}", "--train", "{r}", "--dataset-version", "X"),
    ("check-alignment", DT, "--validation", "{r}", "--dataset-version", "X"),
    ("check-alignment", DT, "--validation", "{v}", "--dataset-version", "X", "--out", "{r}"), ("explain", DT, "--out", "{r}")])
def test_reserved_file_is_refused_on_every_path_argument(tmp_path, args, fake_ollama):
    reserved = tmp_path / RESERVED_NAME
    reserved.write_text("must never be read")
    v = _validation_csv(tmp_path / "v.csv")
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    res = tmp_path / "res.json"
    if "{res}" in args:
        assert cli("explain", DT, "--out", res, "--limit", "1", "--quiet", env_extra=env).returncode == 0
    hook, log = _open_recorder(tmp_path)
    filled = [a.format(r=reserved, v=v, o=tmp_path / "o.json", res=res) for a in args]
    r = cli(*filled, env_extra={**env, "PYTHONPATH": hook})
    assert r.returncode != 0 and "RESERVED_SPLIT_REFUSED" in r.stderr, r.stdout + r.stderr
    assert not log.exists(), log.read_text()


def test_open_recorder_sees_a_real_open(tmp_path):
    hook, log = _open_recorder(tmp_path)
    target = tmp_path / RESERVED_NAME
    target.write_text("x")
    subprocess.run([sys.executable, "-c", f"open({str(target)!r}).read()"], check=True,
                   env={**os.environ, "PYTHONPATH": hook})
    assert log.exists()


def test_check_alignment_fails_an_empty_export_for_v3_only(tmp_path):
    path = _validation_csv(tmp_path / "v.csv")
    r = cli("check-alignment", DT_EMPTY, "--validation", path)
    assert r.returncode == 1 and any("nothing was compared" in p for p in json.loads(r.stdout)["problems"])
    assert cli("check-alignment", DT_EMPTY, "--validation", path, "--dataset-version", "X").returncode == 0


@pytest.mark.parametrize("damage", [{"validation": "x"}, {"validation": {"errors": 5}}, {"detector": ["x"]}])
def test_a_malformed_results_field_is_an_input_error_not_a_traceback(fake_ollama, tmp_path, damage):
    out = tmp_path / "res.json"
    assert cli("explain", DT, "--out", out, "--limit", "1", "--quiet",
               env_extra={"NOVA_OLLAMA_URL": fake_ollama.url}).returncode == 0
    doc = load_json(out)
    doc["results"][0].update(damage)
    write_json(out, doc)
    r = cli("worksheet", out, "--out", tmp_path / "sheet.csv")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert json.loads(r.stderr)["error"]["code"] == "RESULTS_MALFORMED"


@pytest.mark.parametrize("key,value", [("provenance", "x"), ("generator", "x"), ("provenance", {"previous_provenance": "x"})])
def test_a_malformed_stored_block_is_an_input_error(fake_ollama, tmp_path, key, value):
    out = tmp_path / "res.json"
    assert cli("explain", DT, "--out", out, "--limit", "1", "--quiet",
               env_extra={"NOVA_OLLAMA_URL": fake_ollama.url}).returncode == 0
    doc = load_json(out)
    doc[key] = value
    write_json(out, doc)
    r = cli("verify", out, "--export", DT)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESULTS_MALFORMED"


def test_report_file_of_another_kind_is_not_overwritten(tmp_path):
    path = _validation_csv(tmp_path / "v.csv")
    for base in (("check-alignment", DT_FIXTURE, "--validation", path, "--dataset-version", "X"),
                 ("check-dataset", "--validation", path, "--dataset-version", "X")):
        notes = tmp_path / "notes.json"
        notes.write_text("my notes")
        r = cli(*base, "--out", notes)
        assert r.returncode == 4 and json.loads(r.stderr)["error"]["code"] == "OUTPUT_NOT_OURS"
        assert notes.read_text() == "my notes"
        assert json.loads(r.stdout)["report"].startswith("nova-qwen-")  # the finished result is still printed
        report = tmp_path / f"{base[0]}.json"
        assert cli(*base, "--out", report).returncode == 0
        assert cli(*base, "--out", report).returncode == 0  # its own earlier report may be replaced


@pytest.mark.parametrize("version", ["v3", "V3 ", "V-3", "", "CICIDS2017-V3"])
def test_near_spellings_of_v3_do_not_skip_the_v3_checks(tmp_path, version):
    path = _validation_csv(tmp_path / "v.csv")
    for command in (("check-dataset", "--validation", path), ("check-alignment", DT_FIXTURE, "--validation", path)):
        r = cli(*command, "--dataset-version", version)
        assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "BAD_ARGUMENTS"


def test_unreadable_csv_is_invalid_input_not_a_crash(tmp_path):
    bad = tmp_path / "v.csv"
    bad.write_bytes(b"\xff\xfe\x00bad")
    for command in (("check-dataset", "--validation", bad), ("check-alignment", DT_FIXTURE, "--validation", bad)):
        r = cli(*command)
        assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "DATASET_UNREADABLE"


def test_resume_keeps_or_refuses_the_recorded_dataset_identity(fake_ollama, tmp_path):
    env = {"NOVA_OLLAMA_URL": fake_ollama.url}
    out = tmp_path / "r.json"
    assert cli("explain", DT_FIXTURE, "--out", out, "--limit", "1", "--dataset-version", "V3", "--quiet", env_extra=env).returncode == 0
    assert cli("explain", DT_FIXTURE, "--out", out, "--resume", "--quiet", env_extra=env).returncode == 0
    assert load_json(out)["provenance"]["dataset_version"] == "V3"
    r = cli("explain", DT_FIXTURE, "--out", out, "--resume", "--dataset-version", "OTHER", "--quiet", env_extra=env)
    assert r.returncode == 2 and json.loads(r.stderr)["error"]["code"] == "RESUME_MISMATCH"
