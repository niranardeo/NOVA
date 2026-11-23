"""Integration with the team's real, unmodified src/alert_adapter.py: exports it writes are consumed
directly (no hand-written JSON). Requires pandas/numpy (the classifier stack); skipped otherwise."""
import pytest

pd = pytest.importorskip("pandas")

from qwen_test_support import ROOT  # noqa: E402
from src.alert_adapter import export_alerts  # noqa: E402  (team code)
from src.nova_qwen.alerts import load_export  # noqa: E402
from src.nova_qwen.consumer import run  # noqa: E402
from src.nova_qwen.correlate import correlate  # noqa: E402
from src.nova_qwen.verify import verify_results  # noqa: E402

RUN_ID = "validation-20261030T120000000000Z"


def frame(n=12):
    return pd.DataFrame({
        "Flow Duration": [1_000_000 * (i + 1) for i in range(n)],
        "Total Fwd Packets": [i + 2 for i in range(n)],
        "Flow Bytes/s": [100.5 + i for i in range(n)],
        "Bwd Packet Length Max": [9] * n,  # not allowlisted
    })


DT = ["Normal Traffic", "DoS", "DDoS", "Normal Traffic", "Port Scanning", "Bots",
      "Normal Traffic", "Brute Force", "Web Attacks", "DoS", "Normal Traffic", "DDoS"]
RF = ["Normal Traffic", "DoS", "DoS", "Port Scanning", "Normal Traffic", "Bots",
      "Normal Traffic", "Brute Force", "Web Attacks", "DDoS", "Normal Traffic", "DDoS"]


@pytest.fixture
def exports(tmp_path):
    f = frame()
    export_alerts(f, DT, "decision_tree", tmp_path / "outputs" / "decision_tree_alerts.json",
                  run_id=RUN_ID, dataset="CICIDS2017")
    export_alerts(f, RF, "random_forest", tmp_path / "outputs" / "random_forest_alerts.json",
                  run_id=RUN_ID, dataset="CICIDS2017")
    return tmp_path / "outputs" / "decision_tree_alerts.json", tmp_path / "outputs" / "random_forest_alerts.json"


def test_adapter_exports_are_accepted_and_traceable(exports, fake_ollama, settings, tmp_path):
    dt_path, rf_path = exports
    dt, rf = load_export(dt_path), load_export(rf_path)
    assert len(dt.alerts) == 8 and len(rf.alerts) == 8
    for export, out in ((dt, tmp_path / "dt_r.json"), (rf, tmp_path / "rf_r.json")):
        doc = run(export, settings, out, limit=None)
        assert doc["summary"]["explained"] == 8
        assert verify_results(doc, export)["passed"]
        f = frame()
        for r in doc["results"]:
            row = f.iloc[r["record_position"] - 1]
            assert r["evidence"] == {k: float(row[k]) for k in ("Flow Duration", "Total Fwd Packets", "Flow Bytes/s")}
            assert r["evidence_normalized"][0]["value"] == r["record_position"] * 1.0  # 1e6 us per row = seconds
    report = correlate(dt, rf)
    assert report["counts"] == {"both_same_class": 5, "both_different_class": 2, "decision_tree_only": 1,
                                "random_forest_only": 1}


def test_adapter_export_files_are_0600_but_still_readable_by_the_same_user(exports):
    # Documented adapter behaviour (audit NOVA-28): mkstemp files are owner-only. The consumer runs as
    # the same user in the documented workflow, so it can read them.
    import os
    import stat
    if os.name == "posix":
        assert stat.S_IMODE(exports[0].stat().st_mode) == 0o600
    assert len(load_export(exports[0]).alerts) == 8


def test_two_runs_of_the_adapter_cannot_be_correlated(tmp_path):
    f = frame()
    export_alerts(f, DT, "decision_tree", tmp_path / "dt.json", run_id=RUN_ID, dataset="CICIDS2017")
    export_alerts(f, RF, "random_forest", tmp_path / "rf.json", run_id=RUN_ID.replace("12", "13"), dataset="CICIDS2017")
    from src.nova_qwen.errors import IncompatibleExports
    with pytest.raises(IncompatibleExports):
        correlate(load_export(tmp_path / "dt.json"), load_export(tmp_path / "rf.json"))


def test_reordered_frame_is_detected_as_incompatible(tmp_path):
    """record_id is a row position: an RF export from a reordered frame must not be joined."""
    f = frame()
    export_alerts(f, DT, "decision_tree", tmp_path / "dt.json", run_id=RUN_ID, dataset="CICIDS2017")
    shuffled = f.iloc[::-1].reset_index(drop=True)
    export_alerts(shuffled, RF[::-1], "random_forest", tmp_path / "rf.json", run_id=RUN_ID, dataset="CICIDS2017")
    from src.nova_qwen.errors import IncompatibleExports
    with pytest.raises(IncompatibleExports) as e:
        correlate(load_export(tmp_path / "dt.json"), load_export(tmp_path / "rf.json"))
    assert "carry different evidence" in str(e.value)


def test_adapter_source_is_unmodified():
    """Guard against accidental edits to the team's adapter in this branch."""
    import hashlib
    digest = hashlib.sha256((ROOT / "src" / "alert_adapter.py").read_bytes()).hexdigest()
    assert digest == "1430c3b5a40ed86acfe47eeb2bf5004664ee96311b0d5adbd036ef654acbff9c"
