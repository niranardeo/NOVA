"""scripts/qwen_export_v3_alerts.py and the synthetic demo generator, with the team's unmodified adapter."""
import json
import os
import pickle
import subprocess
import sys

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from qwen_test_support import (ROOT, DecisionTreeStandIn, PickledModel, PipelineStandIn, RaisingModel,  # noqa: E402
                               RandomForestStandIn, RunsOnLoad, SearchStandIn, load_json)
from src.nova_qwen import datasets as D  # noqa: E402
from src.nova_qwen.alerts import load_export  # noqa: E402

BRIDGE = ROOT / "scripts" / "qwen_export_v3_alerts.py"
GENERATOR = ROOT / "scripts" / "qwen_make_synthetic_demo.py"
VERSION = ("--dataset-version", "SYNTHETIC-DEMO", "--dataset", "SYNTHETIC-NOT-CICIDS2017")


def run(script, *args, cwd=ROOT):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "tests" / "qwen")}
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=180)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    r = run(GENERATOR, "--out-dir", folder, "--train-rows", 200, "--validation-rows", 300)
    assert r.returncode == 0, r.stderr
    return folder


def bridge(demo, out, *extra):
    return run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--out-dir", out, *VERSION, *extra)


def from_files(demo, out, *extra):
    return bridge(demo, out, "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions", demo / "rf_predictions.txt", *extra)


def error_code(r):
    return json.loads(r.stderr)["error"]["code"]


def test_generator_labels_itself_synthetic_and_cannot_pass_as_v3(demo):
    assert "not CICIDS2017" in (demo / "SYNTHETIC_DATA_NOT_CICIDS2017.txt").read_text()
    frame = pd.read_csv(demo / "synthetic_validation.csv")
    assert list(frame.columns) == list(D.V3_COLUMNS) and set(frame[D.LABEL_COLUMN]) == set(D.ALL_CLASSES)
    report = D.check_dataset(validation=demo / "synthetic_validation.csv")
    assert not report["passed"] and not report["v3_validation_verified"]


def test_generator_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(GENERATOR, "--out-dir", a, "--train-rows", 100, "--validation-rows", 100).returncode == 0
    assert run(GENERATOR, "--out-dir", b, "--train-rows", 100, "--validation-rows", 100).returncode == 0
    assert (a / "synthetic_validation.csv").read_bytes() == (b / "synthetic_validation.csv").read_bytes()


def test_generator_refuses_tiny_files(tmp_path):
    assert run(GENERATOR, "--out-dir", tmp_path, "--train-rows", 5).returncode == 2


def test_exports_come_from_the_complete_frame_in_order(demo, tmp_path):
    r = from_files(demo, tmp_path)
    assert r.returncode == 0, r.stderr
    frame = pd.read_csv(demo / "synthetic_validation.csv")
    preds = (demo / "dt_predictions.txt").read_text().split("\n")[:-1]
    export = load_export(tmp_path / "decision_tree_alerts.json", expected_detector="decision_tree")
    assert len(export.alerts) == sum(p != "Normal Traffic" for p in preds)
    for alert in export.alerts:
        row = frame.iloc[alert.record_position - 1]
        assert alert.evidence["Flow Duration"] == row["Flow Duration"] and alert.predicted_class == preds[alert.record_position - 1]
    assert D.check_alignment(export, demo / "synthetic_validation.csv")["passed"]


def test_both_exports_share_a_run_and_correlate(demo, tmp_path):
    from src.nova_qwen.correlate import check_compatible
    assert from_files(demo, tmp_path).returncode == 0
    dt, rf = load_export(tmp_path / "decision_tree_alerts.json"), load_export(tmp_path / "random_forest_alerts.json")
    assert dt.run_id == rf.run_id and dt.dataset == "SYNTHETIC-NOT-CICIDS2017"
    assert check_compatible(dt, rf)["shared_records_checked"] > 0


def test_exports_hold_no_labels_or_unapproved_features(demo, tmp_path):
    assert from_files(demo, tmp_path).returncode == 0
    text = (tmp_path / "decision_tree_alerts.json").read_text()
    assert "Attack Type" not in text and "Destination Port" not in text
    assert set(load_export(tmp_path / "decision_tree_alerts.json").alerts[0].evidence) == {"Flow Duration", "Total Fwd Packets", "Flow Bytes/s"}


def test_manifest_records_dataset_hash_and_metrics(demo, tmp_path):
    assert from_files(demo, tmp_path).returncode == 0
    m = load_json(tmp_path / "export_manifest.json")
    assert m["v3_validation_verified"] is False and m["dataset_field"] == "SYNTHETIC-NOT-CICIDS2017"
    assert m["dataset_check"]["validation"]["sha256"] == D.sha256_file(demo / "synthetic_validation.csv")
    dt = m["classifiers"]["decision_tree"]
    labels = pd.read_csv(demo / "synthetic_validation.csv")[D.LABEL_COLUMN].to_numpy(dtype=str)
    preds = np.array((demo / "dt_predictions.txt").read_text().split("\n")[:-1])
    assert dt["metrics_on_validation"]["accuracy"] == pytest.approx(float((labels == preds).mean()))
    assert dt["metrics_on_validation"]["binary"]["TP"] + dt["metrics_on_validation"]["binary"]["FP"] == dt["export"]["alert_count"]
    assert dt["source"]["kind"] == "predictions" and dt["alignment"]["passed"]


def test_metrics_match_a_hand_worked_example():
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    labels = np.array(["Normal Traffic", "Normal Traffic", "DoS", "DoS", "Bots"])
    preds = np.array(["Normal Traffic", "DoS", "DoS", "Normal Traffic", "Bots"])
    m = bridge_module.metrics(labels, preds)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["per_class"]["DoS"]["recall"] == pytest.approx(0.5) and m["per_class"]["Bots"]["f1"] == pytest.approx(1.0)
    assert m["binary"] == {"TP": 2, "FP": 1, "FN": 1, "TN": 1}
    assert m["macro_f1"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    only_predicted = bridge_module.metrics(np.array(["DoS", "DoS"]), np.array(["DoS", "Bots"]))
    assert set(only_predicted["per_class"]) == {"DoS", "Bots"}  # as in sklearn's default


def test_model_settings_become_plain_values():
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    assert bridge_module.recorded_param(np.int64(15)) == 15 and isinstance(bridge_module.recorded_param(np.int64(15)), int)
    assert bridge_module.recorded_param(None) is None and bridge_module.recorded_param(object()) == "<object>"


def test_identity_note_says_what_was_compared():
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    note = bridge_module.identity_note
    assert "SHA-256 matches" in note("V3", {"v3_validation_verified": True, "v3_profile_matched": True})
    assert "SHA-256 not compared" in note("V3", {"v3_validation_verified": False, "v3_profile_matched": True})
    assert "NOT V3" in note("X", {"v3_validation_verified": False, "v3_profile_matched": False})


def test_npy_predictions_work_and_object_arrays_are_refused(demo, tmp_path):
    preds = np.array((demo / "dt_predictions.txt").read_text().split("\n")[:-1])
    np.save(tmp_path / "dt.npy", preds)
    np.save(tmp_path / "rf.npy", np.array((demo / "rf_predictions.txt").read_text().split("\n")[:-1]))
    r = bridge(demo, tmp_path / "o", "--dt-predictions", tmp_path / "dt.npy", "--rf-predictions", tmp_path / "rf.npy")
    assert r.returncode == 0, r.stderr
    np.save(tmp_path / "obj.npy", preds.astype(object), allow_pickle=True)
    r = bridge(demo, tmp_path / "o2", "--dt-predictions", tmp_path / "obj.npy", "--rf-predictions", tmp_path / "rf.npy")
    assert r.returncode == 2 and error_code(r) == "DATASET_ERROR" and not (tmp_path / "o2" / "decision_tree_alerts.json").exists()
    assert "astype(str)" in json.loads(r.stderr)["error"]["message"]


def test_a_sample_or_smoke_subset_is_refused(demo, tmp_path):
    subset = tmp_path / "subset.txt"
    subset.write_text("\n".join((demo / "dt_predictions.txt").read_text().split("\n")[:100]))
    r = bridge(demo, tmp_path / "o", "--dt-predictions", subset, "--rf-predictions", demo / "rf_predictions.txt")
    assert r.returncode == 2 and error_code(r) == "PREDICTIONS_NOT_COMPLETE"
    assert not (tmp_path / "o" / "decision_tree_alerts.json").exists()


def test_numeric_or_unknown_labels_are_refused(demo, tmp_path):
    numeric = tmp_path / "numeric.txt"
    numeric.write_text("\n".join(["1"] * 300))
    r = bridge(demo, tmp_path / "o", "--dt-predictions", numeric, "--rf-predictions", demo / "rf_predictions.txt")
    assert r.returncode == 2 and error_code(r) == "PREDICTIONS_UNKNOWN_CLASS"
    assert "np.str_" not in r.stderr and "['1']" in json.loads(r.stderr)["error"]["message"]


def test_reserved_validation_and_predictions_are_refused(demo, tmp_path):
    reserved = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    reserved.write_text("must never be read")
    r = run(BRIDGE, "--validation", reserved, "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions",
            demo / "rf_predictions.txt", "--out-dir", tmp_path / "o", *VERSION)
    assert r.returncode == 2 and error_code(r) == "RESERVED_SPLIT_REFUSED"
    preds = tmp_path / "test_predictions.txt"
    preds.write_text((demo / "dt_predictions.txt").read_text())
    r = bridge(demo, tmp_path / "o", "--dt-predictions", preds, "--rf-predictions", demo / "rf_predictions.txt")
    assert r.returncode == 2 and error_code(r) == "RESERVED_SPLIT_REFUSED"


def test_v3_is_the_default_and_a_non_v3_file_is_refused(demo, tmp_path):
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--dt-predictions", demo / "dt_predictions.txt",
            "--rf-predictions", demo / "rf_predictions.txt", "--dataset", "CICIDS2017-V3", "--out-dir", tmp_path / "o")
    assert r.returncode == 2 and "dataset check failed" in json.loads(r.stderr)["error"]["message"]
    assert not (tmp_path / "o").exists()


def test_dataset_field_naming_v3_needs_the_v3_version(demo, tmp_path):
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--dt-predictions", demo / "dt_predictions.txt",
            "--rf-predictions", demo / "rf_predictions.txt", "--dataset", "CICIDS2017-V3", "--dataset-version", "SYNTHETIC-DEMO",
            "--out-dir", tmp_path / "o")
    assert r.returncode == 2 and error_code(r) == "BAD_ARGUMENTS"


def test_dataset_field_is_required(demo, tmp_path):
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--out-dir", tmp_path)
    assert r.returncode == 2 and "--dataset" in r.stderr


@pytest.mark.parametrize("extra", [(), ("--rf-predictions", "x.txt"),
                                   ("--dt-predictions", "d.txt", "--dt-model", "d.pkl", "--rf-predictions", "x.txt",
                                    "--trust-model-files")])
def test_each_classifier_needs_exactly_one_source(demo, tmp_path, extra):
    r = bridge(demo, tmp_path, *extra)
    assert r.returncode == 2 and error_code(r) == "BAD_ARGUMENTS" and "exactly one of" in r.stderr


def model_file(path, **kw):
    with open(path, "wb") as fh:
        pickle.dump(PickledModel(**kw), fh)
    return path


def test_model_files_need_explicit_trust(demo, tmp_path):
    marker = tmp_path / "model_code_ran"
    dt = tmp_path / "dt.pkl"
    dt.write_bytes(pickle.dumps(RunsOnLoad(marker)))
    r = bridge(demo, tmp_path / "o", "--dt-model", dt, "--rf-predictions", demo / "rf_predictions.txt")
    assert r.returncode == 2 and "--trust-model-files" in json.loads(r.stderr)["error"]["message"]
    assert not marker.exists()


def test_model_named_like_a_reserved_split_is_refused_before_loading(demo, tmp_path):
    marker = tmp_path / "model_was_loaded"
    model = tmp_path / "test_model.pkl"
    model.write_bytes(pickle.dumps(RunsOnLoad(marker)))
    r = bridge(demo, tmp_path / "o", "--dt-model", model, "--rf-predictions", demo / "rf_predictions.txt", "--trust-model-files")
    assert r.returncode == 2 and error_code(r) == "RESERVED_SPLIT_REFUSED"
    assert not marker.exists()


def test_model_predictions_are_made_in_chunks_and_recorded(demo, tmp_path):
    params = {"max_depth": 15, "min_samples_leaf": 10, "n_estimators": 50, "random_state": 42, "n_jobs": 1, "other": 9}
    dt = model_file(tmp_path / "dt.pkl", every=2, params=params)
    r = bridge(demo, tmp_path / "o", "--dt-model", dt, "--rf-predictions", demo / "rf_predictions.txt",
               "--trust-model-files", "--chunksize", "71")
    assert r.returncode == 0, r.stderr
    entry = load_json(tmp_path / "o" / "export_manifest.json")["classifiers"]["decision_tree"]
    assert entry["source"]["kind"] == "model" and entry["source"]["params"] == {k: params[k] for k in params if k != "other"}
    assert entry["source"]["sha256"] == D.sha256_file(dt)
    # the stand-in's counter restarts per chunk, so an odd chunk size gives a count that one piece would not
    expected = sum(1 for start in range(0, 300, 71) for i in range(min(71, 300 - start)) if i % 2 == 0)
    assert expected != 150 and entry["export"]["alert_count"] == expected
    assert entry["source"]["feature_order_verified"] is False and "WARNING: decision_tree" in r.stdout


def test_model_with_different_feature_columns_is_refused(demo, tmp_path):
    model = PickledModel()
    model.feature_names_in_ = ["only", "these"]
    (tmp_path / "dt.pkl").write_bytes(pickle.dumps(model))
    r = bridge(demo, tmp_path / "o", "--dt-model", tmp_path / "dt.pkl", "--rf-predictions", demo / "rf_predictions.txt", "--trust-model-files")
    assert r.returncode == 2 and error_code(r) == "MODEL_FEATURES_DIFFER"


def test_bad_chunksize_is_refused(demo, tmp_path):
    r = from_files(demo, tmp_path, "--chunksize", "0")
    assert r.returncode == 2 and error_code(r) == "BAD_ARGUMENTS"


def snapshot(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()} if folder.exists() else {}


def test_a_failure_on_the_second_classifier_leaves_a_fresh_folder_empty(demo, tmp_path):
    short = tmp_path / "short.txt"
    short.write_text("Normal Traffic\n")
    r = bridge(demo, tmp_path / "o", "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions", short)
    assert r.returncode == 2 and error_code(r) == "PREDICTIONS_NOT_COMPLETE"
    assert snapshot(tmp_path / "o") == {}


def test_a_failure_on_the_second_classifier_leaves_an_earlier_run_untouched(demo, tmp_path):
    out = tmp_path / "o"
    assert from_files(demo, out).returncode == 0
    before = snapshot(out)
    with open(tmp_path / "raises.pkl", "wb") as fh:
        pickle.dump(RaisingModel(), fh)
    r = bridge(demo, out, "--dt-predictions", demo / "rf_predictions.txt", "--rf-model", tmp_path / "raises.pkl", "--trust-model-files")
    assert r.returncode == 1 and r.stdout.strip().splitlines()[-1] == "EXPORT NOT DONE"
    assert json.loads(r.stderr)["error"]["code"] == "UNEXPECTED_ERROR" and snapshot(out) == before


def test_no_staging_folder_is_left_behind(demo, tmp_path):
    assert from_files(demo, tmp_path / "o").returncode == 0
    assert sorted(snapshot(tmp_path / "o")) == ["decision_tree_alerts.json", "export_manifest.json", "random_forest_alerts.json"]


def test_swapped_models_are_noticed(demo, tmp_path):
    with open(tmp_path / "forest.pkl", "wb") as fh:
        pickle.dump(RandomForestStandIn(), fh)
    r = bridge(demo, tmp_path / "o", "--dt-model", tmp_path / "forest.pkl", "--rf-predictions", demo / "rf_predictions.txt", "--trust-model-files")
    assert r.returncode == 2 and error_code(r) == "MODEL_DETECTOR_MISMATCH" and snapshot(tmp_path / "o") == {}


@pytest.mark.parametrize("wrap", [PipelineStandIn, SearchStandIn])
def test_a_swap_is_noticed_inside_a_wrapped_model(demo, tmp_path, wrap):
    with open(tmp_path / "wrapped.pkl", "wb") as fh:
        pickle.dump(wrap(RandomForestStandIn()), fh)
    r = bridge(demo, tmp_path / "o", "--dt-model", tmp_path / "wrapped.pkl", "--rf-predictions", demo / "rf_predictions.txt", "--trust-model-files")
    assert r.returncode == 2 and error_code(r) == "MODEL_DETECTOR_MISMATCH" and snapshot(tmp_path / "o") == {}


@pytest.mark.parametrize("wrap", [PipelineStandIn, SearchStandIn])
def test_a_wrapped_model_records_the_real_classifier_and_its_settings(demo, tmp_path, wrap):
    with open(tmp_path / "wrapped.pkl", "wb") as fh:
        pickle.dump(wrap(DecisionTreeStandIn()), fh)
    r = bridge(demo, tmp_path / "o", "--dt-model", tmp_path / "wrapped.pkl", "--rf-predictions", demo / "rf_predictions.txt", "--trust-model-files")
    assert r.returncode == 0, r.stderr
    source = load_json(tmp_path / "o" / "export_manifest.json")["classifiers"]["decision_tree"]["source"]
    assert source["class"] == wrap.__name__ and source["final_estimator_class"] == "DecisionTreeStandIn"
    assert source["params"] == {"max_depth": 7, "random_state": 42}


def test_one_source_for_both_classifiers_is_refused(demo, tmp_path):
    r = bridge(demo, tmp_path / "o", "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions", demo / "dt_predictions.txt")
    assert r.returncode == 2 and error_code(r) == "SAME_CLASSIFIER_SOURCE"


def test_one_model_file_copied_for_both_classifiers_is_refused(demo, tmp_path):
    for name in ("a.pkl", "b.pkl"):
        with open(tmp_path / name, "wb") as fh:
            pickle.dump(PickledModel(), fh)
    r = bridge(demo, tmp_path / "o", "--dt-model", tmp_path / "a.pkl", "--rf-model", tmp_path / "b.pkl", "--trust-model-files")
    assert r.returncode == 2 and error_code(r) == "SAME_CLASSIFIER_SOURCE"


def test_identical_predictions_in_another_format_are_reported_outside_v3(demo, tmp_path):
    copy = tmp_path / "same.txt"
    copy.write_text("prediction\n" + (demo / "dt_predictions.txt").read_text())
    r = bridge(demo, tmp_path / "o", "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions", copy)
    assert r.returncode == 0 and "identical on every row" in r.stdout
    assert load_json(tmp_path / "o" / "export_manifest.json")["predictions_identical"] is True


def test_different_predictions_give_no_warning(demo, tmp_path):
    r = from_files(demo, tmp_path / "o")
    assert r.returncode == 0 and "WARNING" not in r.stdout
    assert load_json(tmp_path / "o" / "export_manifest.json")["predictions_identical"] is False


def narrowed_v3(demo, tmp_path, monkeypatch):
    """A rehearsal: the demo file, under the V3 name, with V3's row and class counts narrowed to match it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    data = tmp_path / "data"
    data.mkdir()
    frame = pd.read_csv(demo / "synthetic_validation.csv")
    validation = data / D.V3_VALIDATION_FILE
    frame.to_csv(validation, index=False)
    monkeypatch.setattr(D, "V3_VALIDATION_ROWS", len(frame))
    monkeypatch.setattr(D, "V3_VALIDATION_CLASSES", {k: int(v) for k, v in frame[D.LABEL_COLUMN].value_counts().items()})
    monkeypatch.setattr(bridge_module, "V3_VALIDATION_ROWS", len(frame))
    return bridge_module, frame, validation


def test_identical_predictions_are_refused_for_v3(demo, tmp_path, monkeypatch, capsys):
    bridge_module, _, validation = narrowed_v3(demo, tmp_path, monkeypatch)
    copy = tmp_path / "same.txt"
    copy.write_text((demo / "dt_predictions.txt").read_text())
    out = tmp_path / "o"
    code = bridge_module.entry(["--validation", str(validation), "--dt-predictions", str(demo / "dt_predictions.txt"),
                                "--rf-predictions", str(copy), "--dataset", "CICIDS2017-V3", "--out-dir", str(out)])
    assert code == 2 and "SAME_CLASSIFIER_SOURCE" in capsys.readouterr().err and snapshot(out) == {}


def test_a_v3_export_with_no_alerts_is_refused(demo, tmp_path, monkeypatch, capsys):
    bridge_module, frame, validation = narrowed_v3(demo, tmp_path, monkeypatch)
    empty = tmp_path / "all_normal.txt"
    empty.write_text("\n".join(["Normal Traffic"] * len(frame)) + "\n")
    out = tmp_path / "o"
    code = bridge_module.entry(["--validation", str(validation), "--dt-predictions", str(empty), "--rf-predictions",
                                str(demo / "rf_predictions.txt"), "--dataset", "CICIDS2017-V3", "--out-dir", str(out)])
    captured = capsys.readouterr()
    assert code == 2 and "nothing was compared" in captured.err and "EXPORT NOT DONE" in captured.out
    assert snapshot(out) == {}


def test_v3_export_with_an_alert_free_tail_is_refused(demo, tmp_path, monkeypatch, capsys):
    """The row count is right, so only the V3 tail check can catch this."""
    bridge_module, frame, validation = narrowed_v3(demo, tmp_path, monkeypatch)
    sources = []
    for name, attacks in (("dt.txt", 3), ("rf.txt", 4)):  # every alert is in the first rows, as after head(n)
        preds = ["DoS"] * attacks + ["Normal Traffic"] * (len(frame) - attacks)
        (tmp_path / name).write_text("\n".join(preds) + "\n")
        sources.append(tmp_path / name)
    out = tmp_path / "o"
    code = bridge_module.entry(["--validation", str(validation), "--dt-predictions", str(sources[0]), "--rf-predictions",
                                str(sources[1]), "--dataset", "CICIDS2017-V3", "--out-dir", str(out)])
    captured = capsys.readouterr()
    assert code == 2 and "truncated" in captured.err and "EXPORT NOT DONE" in captured.out
    assert snapshot(out) == {}


def test_the_adapter_staging_path_is_not_printed_and_the_final_paths_are(demo, tmp_path):
    r = from_files(demo, tmp_path / "o")
    assert r.returncode == 0 and ".export-staging" not in r.stdout
    assert str(tmp_path / "o" / "decision_tree_alerts.json") in r.stdout
    assert str(tmp_path / "o" / "random_forest_alerts.json") in r.stdout


def test_a_directory_at_an_output_name_is_refused_and_changes_nothing(demo, tmp_path):
    out = tmp_path / "o"
    assert from_files(demo, out).returncode == 0
    (out / "random_forest_alerts.json").unlink()
    (out / "random_forest_alerts.json").mkdir()
    before = {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()}
    r = from_files(demo, out, "--run-id", "run-two")
    assert r.returncode == 2 and error_code(r) == "BAD_ARGUMENTS"
    assert {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()} == before
    assert (out / "random_forest_alerts.json").is_dir()


def test_a_failed_final_move_says_the_folder_may_be_mixed(demo, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    real = os.replace

    def fails_on_the_second_move(src, dst):
        if str(dst) == str(tmp_path / "o" / "random_forest_alerts.json"):
            raise OSError("read-only target")
        return real(src, dst)

    monkeypatch.setattr(bridge_module.os, "replace", fails_on_the_second_move)
    code = bridge_module.entry(["--validation", str(demo / "synthetic_validation.csv"), "--dt-predictions", str(demo / "dt_predictions.txt"),
                                "--rf-predictions", str(demo / "rf_predictions.txt"), "--dataset", "SYNTHETIC-NOT-CICIDS2017",
                                "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(tmp_path / "o")])
    err = json.loads(capsys.readouterr().err)["error"]
    assert code == 2 and err["code"] == "EXPORT_MOVE_INCOMPLETE" and "decision_tree_alerts.json" in err["message"]


def test_sigterm_during_the_export_leaves_no_staging_folder(demo, tmp_path):
    """kill and docker stop send SIGTERM. The adapter is slowed so the signal lands while staging holds an export."""
    out = tmp_path / "o"
    argv = ["x", "--validation", str(demo / "synthetic_validation.csv"), "--dt-predictions", str(demo / "dt_predictions.txt"),
            "--rf-predictions", str(demo / "rf_predictions.txt"), *VERSION, "--out-dir", str(out)]
    driver = ("import runpy, sys, time\n"
              "import src.alert_adapter as A\n"
              "real = A.export_alerts\n"
              "def slow(**kw):\n"
              "    real(**kw)\n"
              "    print('STAGED', file=sys.__stdout__, flush=True)\n"
              "    time.sleep(60)\n"
              "A.export_alerts = slow\n"
              f"sys.argv = {argv!r}\n"
              f"runpy.run_path({str(BRIDGE)!r}, run_name='__main__')\n")
    p = subprocess.Popen([sys.executable, "-c", driver], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        assert p.stdout.readline().strip() == "STAGED", p.stderr.read()
        p.terminate()
        assert p.wait(timeout=30) == 143
    finally:
        p.kill()
        p.communicate()
    assert snapshot(out) == {} and not [f for f in out.iterdir()]


def test_without_joblib_a_compressed_model_says_what_to_install(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    monkeypatch.setitem(sys.modules, "joblib", None)
    model = tmp_path / "compressed.joblib"
    model.write_bytes(b"x\x9c\x01compressed-not-a-pickle")
    with pytest.raises(bridge_module.DatasetError) as info:
        bridge_module.load_model(model)
    assert info.value.code == "MODEL_NEEDS_JOBLIB"


def test_a_load_error_from_joblib_is_reported_as_itself(tmp_path, monkeypatch):
    """It must not fall through to pickle, which would hide the real cause behind an unpickling error."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import types
    import qwen_export_v3_alerts as bridge_module

    def load(path):
        raise ModuleNotFoundError("No module named 'gone'")

    monkeypatch.setitem(sys.modules, "joblib", types.SimpleNamespace(load=load))
    model = tmp_path / "m.joblib"
    model.write_bytes(b"irrelevant")
    with pytest.raises(ModuleNotFoundError, match="gone"):
        bridge_module.load_model(model)


def test_wrong_published_hash_stops_the_export(demo, tmp_path):
    r = from_files(demo, tmp_path / "o", "--expected-validation-sha256", "0" * 64)
    assert r.returncode == 2 and "sha256" in r.stderr and snapshot(tmp_path / "o") == {}


@pytest.mark.parametrize("version", ["v3", "V3 ", " "])
def test_near_spellings_of_v3_are_refused(demo, tmp_path, version):
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions",
            demo / "rf_predictions.txt", "--dataset", "SYNTHETIC-NOT-CICIDS2017", "--dataset-version", version, "--out-dir", tmp_path / "o")
    assert r.returncode == 2 and error_code(r) == "BAD_ARGUMENTS"


def test_a_bad_dataset_value_is_refused_before_any_model_is_loaded(demo, tmp_path):
    marker = tmp_path / "model_was_loaded"
    (tmp_path / "dt.pkl").write_bytes(pickle.dumps(RunsOnLoad(marker)))
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--dt-model", tmp_path / "dt.pkl", "--rf-predictions",
            demo / "rf_predictions.txt", "--trust-model-files", "--dataset", " spaced ", "--dataset-version", "SYNTHETIC-DEMO",
            "--out-dir", tmp_path / "o")
    assert r.returncode == 2 and error_code(r) == "CONTRACT_DATASET" and not marker.exists()


def test_reserved_source_is_refused_before_it_is_hashed(demo, tmp_path):
    """Reading it to compare the two sources would already break the rule, so an audit hook watches every open."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    target = tmp_path / "test_predictions.txt"
    target.write_text("must never be read")
    wanted, opened = str(target), []
    sys.addaudithook(lambda event, args: opened.append(args) if event == "open" and str(args[0]) == wanted else None)
    with pytest.raises(bridge_module.NovaQwenError) as info:
        bridge_module.main(["--validation", str(demo / "synthetic_validation.csv"), "--dt-predictions", str(target),
                            "--rf-predictions", str(demo / "rf_predictions.txt"), "--dataset", "SYNTHETIC-NOT-CICIDS2017",
                            "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", str(tmp_path / "o")])
    assert info.value.code == "RESERVED_SPLIT_REFUSED" and not opened


def test_a_failure_while_writing_exports_leaves_nothing_behind(demo, tmp_path, monkeypatch, capsys):
    """Past the checks, the second export can still fail. Nothing appears, and an earlier run stays as it was."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    real = bridge_module.check_alignment

    def fails_on_the_second(export, *a, **k):
        report = real(export, *a, **k)
        if export.detector == "random_forest":
            report = {**report, "passed": False, "problems": ["forced for this test"]}
        return report

    monkeypatch.setattr(bridge_module, "check_alignment", fails_on_the_second)
    args = ["--validation", str(demo / "synthetic_validation.csv"), "--dt-predictions", str(demo / "dt_predictions.txt"),
            "--rf-predictions", str(demo / "rf_predictions.txt"), "--dataset", "SYNTHETIC-NOT-CICIDS2017",
            "--dataset-version", "SYNTHETIC-DEMO", "--out-dir"]
    out = tmp_path / "o"
    with pytest.raises(bridge_module.DatasetError):
        bridge_module.main([*args, str(out)])
    assert snapshot(out) == {}
    monkeypatch.setattr(bridge_module, "check_alignment", real)
    assert bridge_module.main([*args, str(out)]) == 0
    before = snapshot(out)
    monkeypatch.setattr(bridge_module, "check_alignment", fails_on_the_second)
    with pytest.raises(bridge_module.DatasetError):
        bridge_module.main([*args, str(out), "--run-id", "another-run"])
    assert snapshot(out) == before


def test_dataset_field_with_control_characters_is_refused(demo, tmp_path):
    r = run(BRIDGE, "--validation", demo / "synthetic_validation.csv", "--dt-predictions", demo / "dt_predictions.txt", "--rf-predictions",
            demo / "rf_predictions.txt", "--dataset", "a\x1bb", "--dataset-version", "SYNTHETIC-DEMO", "--out-dir", tmp_path / "o")
    assert r.returncode == 2 and error_code(r) == "CONTRACT_DATASET"


def test_unreadable_predictions_give_a_clean_failure(demo, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00")
    r = bridge(demo, tmp_path / "o", "--dt-predictions", bad, "--rf-predictions", demo / "rf_predictions.txt")
    assert r.returncode == 1 and r.stdout.strip().splitlines()[-1] == "EXPORT NOT DONE"
    assert json.loads(r.stderr)["error"]["code"] == "UNEXPECTED_ERROR"


def test_a_reserved_model_name_is_refused_before_any_source_is_hashed(demo, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import qwen_export_v3_alerts as bridge_module
    reserved, other = tmp_path / "cicids2017_V3_test_RESERVED_15.pkl", tmp_path / "rf.pkl"
    reserved.write_text("must never be read")
    other.write_text("x")
    hashed = []
    monkeypatch.setattr(bridge_module, "sha256_file", lambda p: hashed.append(p) or "0" * 64)
    code = bridge_module.entry(["--validation", str(demo / "synthetic_validation.csv"), "--dt-model", str(reserved), "--rf-model", str(other),
                                "--trust-model-files", *VERSION, "--out-dir", str(tmp_path / "o")])
    assert code == 2 and "RESERVED_SPLIT_REFUSED" in capsys.readouterr().err and hashed == []
