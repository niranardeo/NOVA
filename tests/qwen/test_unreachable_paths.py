"""A path whose parent cannot be searched, or an over-long name, is a reported problem, never a traceback."""
import json
import os

import pytest

from src.nova_qwen.cli import main
from qwen_test_support import DT_FIXTURE

needs_permissions = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                       reason="needs POSIX non-root")


@pytest.fixture
def blocked(tmp_path):
    d = tmp_path / "noaccess"
    d.mkdir()
    d.chmod(0)
    yield d
    d.chmod(0o755)


@needs_permissions
@pytest.mark.parametrize("argv,code", [
    (lambda b: ["verify", str(b / "r.json"), "--export", str(DT_FIXTURE)], "RESULTS_UNREADABLE"),
    (lambda b: ["worksheet", str(b / "r.json"), "--out", str(b.parent / "w.csv")], "RESULTS_UNREADABLE"),
    (lambda b: ["validate-export", str(b / "a.json")], "FILE_ERROR"),
    (lambda b: ["check-dataset", "--validation", str(b / "v.csv")], "FILE_ERROR"),
    (lambda b: ["check-alignment", str(DT_FIXTURE), "--validation", str(b / "v.csv")], "FILE_ERROR"),
    (lambda b: ["correlate", "--dt-export", str(b / "a.json"), "--rf-export", str(b / "a.json")], "FILE_ERROR"),
])
def test_unreachable_input_is_exit_2(blocked, capsys, argv, code):
    assert main(argv(blocked)) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == code


def test_over_long_results_name_is_exit_2(tmp_path, capsys):
    assert main(["verify", str(tmp_path / ("x" * 300)), "--export", str(DT_FIXTURE)]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "RESULTS_UNREADABLE"


@needs_permissions
def test_unreachable_out_is_exit_4_before_any_model_call(blocked, capsys, fake_ollama, monkeypatch):
    monkeypatch.setenv("NOVA_OLLAMA_URL", fake_ollama.url)
    assert main(["explain", str(DT_FIXTURE), "--out", str(blocked / "o.json"), "--limit", "1"]) == 4
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "OUTPUT_WRITE_FAILED"
    assert fake_ollama.requests == []
