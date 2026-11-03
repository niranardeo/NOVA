"""Build schema 1.1 alert exports from saved classifier predictions, using the team's unmodified adapter.

Bridges a finished classifier run to the Qwen consumer without retraining and without much memory. With saved
predictions only the three evidence columns and the label are loaded; with saved models every column is read one chunk
at a time to predict, and only those four are kept. The complete validation frame stays in its original row order, so
positional record_ids keep their meaning. It also writes export_manifest.json, which ties the exports to the dataset
hash and to the accuracy the predictions really give, so they can be compared with the reported experiment.

From saved predictions (one per validation row, in file order). Use text with one class per line, or a .npy string
array: predict() returns an object array, so save it with np.save("dt_pred.npy", y_pred_dt.astype(str)).

    python scripts/qwen_export_v3_alerts.py --validation /media/sf_NOVA_Data/cicids2017_V3_validation_15.csv \\
        --dt-predictions data/processed/dt_pred.npy --rf-predictions data/processed/rf_pred.npy \\
        --dataset CICIDS2017-V3 --out-dir outputs

From saved models (predicts in chunks; needs requirements-qwen-e2e.txt). Loading a model file runs code from it, so
only use models you made yourself; --trust-model-files records that decision:

    python scripts/qwen_export_v3_alerts.py --validation ... --dt-model dt.joblib --rf-model rf.joblib \\
        --trust-model-files --dataset CICIDS2017-V3 --out-dir outputs

Do not pass a sample or SMOKE subset: this refuses anything but one prediction per validation row. Labels are read
only for the manifest's metrics and are never written to an export. The reserved test split is refused by name.
A failure before the final move leaves --out-dir as it was. The final move is three renames, which cannot be atomic
together: if one fails, or the process is killed, the folder can hold a mixed set, so trust it only after EXPORT DONE.
A hard kill (kill -9, the out-of-memory killer) cannot clean up and can leave a hidden .export-staging-* folder there;
no tool reads it, and it is safe to delete once no export is running.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.alert_adapter import export_alerts  # noqa: E402  (team code, unmodified)
from src.nova_qwen.alerts import ALLOWED_EVIDENCE, _identifier, load_export  # noqa: E402
from src.nova_qwen.datasets import (ALL_CLASSES, LABEL_COLUMN, V3, V3_VALIDATION_ROWS, check_alignment,  # noqa: E402
                                    check_dataset, refuse_reserved, require_version_spelling, sha256_file)
from src.nova_qwen.errors import DatasetError, NovaQwenError  # noqa: E402
from src.nova_qwen.output import write_json_atomic  # noqa: E402

DETECTORS = ("decision_tree", "random_forest")
RECORDED_PARAMS = ("max_depth", "min_samples_leaf", "n_estimators", "random_state", "n_jobs")
OTHER_DETECTOR_CLASS = {"decision_tree": "RandomForest", "random_forest": "DecisionTree"}
OUTPUT_NAMES = ("decision_tree_alerts.json", "random_forest_alerts.json", "export_manifest.json")


def load_predictions(path: Path, rows: int) -> np.ndarray:
    refuse_reserved(path)
    if not path.is_file():
        raise DatasetError(f"predictions file not found: {path}", code="DATASET_MISSING")
    if path.suffix == ".npy":
        try:
            preds = np.load(path, allow_pickle=False)  # object arrays would need pickle, which can run code
        except ValueError as exc:
            raise DatasetError(f"{path}: cannot load without pickle. predict() returns an object array, so save it "
                               f"with np.save(path, predictions.astype(str)) ({exc})") from exc
    else:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines and lines[0] in ("predicted_class", "prediction"):
            lines = lines[1:]
        preds = np.asarray(lines, dtype=str)
    return check_predictions(preds, rows, str(path))


def check_predictions(preds: np.ndarray, rows: int, source: str) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim != 1 or len(preds) != rows:
        raise DatasetError(f"{source}: {preds.shape} predictions, but the validation file has {rows} rows. Exports "
                           f"must cover the complete validation frame so record_ids stay meaningful.",
                           code="PREDICTIONS_NOT_COMPLETE")
    preds = preds.astype(str)
    unknown = sorted(set(np.unique(preds).tolist()) - set(ALL_CLASSES))
    if unknown:
        raise DatasetError(f"{source}: unexpected class labels {unknown[:10]}; expected class names {list(ALL_CLASSES)}",
                           code="PREDICTIONS_UNKNOWN_CLASS")
    return preds


def load_model(path: Path):
    refuse_reserved(path)
    if not path.is_file():
        raise DatasetError(f"model file not found: {path}", code="DATASET_MISSING")
    try:
        import joblib
    except ImportError:
        joblib = None
    if joblib is not None:
        return joblib.load(path)  # a class module missing from the model file is reported as itself
    import pickle
    with path.open("rb") as fh:
        try:
            return pickle.load(fh)
        except pickle.UnpicklingError as exc:
            raise DatasetError(f"{path}: joblib is not installed and this is not a plain pickle, so it may be a "
                               f"compressed joblib file; install requirements-qwen-e2e.txt ({exc})",
                               code="MODEL_NEEDS_JOBLIB") from exc


def final_estimator(model):
    """The classifier inside a Pipeline or a fitted search object, so the swap check and the settings see the real one."""
    while True:
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
        elif hasattr(model, "steps"):
            model = model.steps[-1][1]
        else:
            return model


def recorded_param(value):
    """Model settings as plain JSON values: a numpy scalar becomes its Python value, anything else its type name."""
    if isinstance(value, np.generic):
        value = value.item()
    return value if value is None or isinstance(value, (str, int, float, bool)) else f"<{type(value).__name__}>"


def predict_in_chunks(model, validation: Path, rows: int, chunksize: int) -> np.ndarray:
    names = getattr(model, "feature_names_in_", None)
    expected = [] if names is None else list(names)
    parts = []
    for chunk in pd.read_csv(validation, chunksize=chunksize, low_memory=False):
        features = chunk.drop(columns=[LABEL_COLUMN])
        if expected and list(features.columns) != expected:
            raise DatasetError("the model was trained on different feature columns than this validation file",
                               code="MODEL_FEATURES_DIFFER")
        parts.append(np.asarray(model.predict(features)))
    return check_predictions(np.concatenate(parts) if parts else np.array([]), rows, "model predictions")


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Accuracy, macro F1 and per-class recall, to compare with the reported experiment. Classes come from labels and
    predictions together, as in sklearn's default."""
    per_class = {}
    for cls in sorted(set(labels) | set(preds)):
        tp = int(np.sum((labels == cls) & (preds == cls)))
        fp = int(np.sum((labels != cls) & (preds == cls)))
        fn = int(np.sum((labels == cls) & (preds != cls)))
        per_class[cls] = {"recall": tp / (tp + fn) if tp + fn else 0.0, "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "support": tp + fn}
    attack, flagged = labels != ALL_CLASSES[0], preds != ALL_CLASSES[0]
    return {"accuracy": float(np.mean(labels == preds)),
            "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])), "per_class": per_class,
            "binary": {"TP": int(np.sum(attack & flagged)), "FP": int(np.sum(~attack & flagged)),
                       "FN": int(np.sum(attack & ~flagged)), "TN": int(np.sum(~attack & ~flagged))}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation", type=Path, required=True, help="the complete validation CSV")
    p.add_argument("--dt-predictions", type=Path)
    p.add_argument("--rf-predictions", type=Path)
    p.add_argument("--dt-model", type=Path)
    p.add_argument("--rf-model", type=Path)
    p.add_argument("--trust-model-files", action="store_true", help="required with --dt-model/--rf-model")
    p.add_argument("--dataset", required=True, help="value for the export's dataset field. CICIDS2017-V3 is "
                                                    "recommended, pending team approval")
    p.add_argument("--dataset-version", default=V3, help="V3 (default) must match Kevin's notebooks; anything else is "
                                                         "never reported as V3")
    p.add_argument("--expected-validation-sha256", help="published hash of the validation file; a mismatch stops the "
                                                        "export, and without it V3 is a profile match only")
    p.add_argument("--run-id", help="default: a new UTC timestamp id, as in the notebook")
    p.add_argument("--chunksize", type=int, default=50_000)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    return p


def identity_note(version: str, check: dict) -> str:
    if check["v3_validation_verified"]:
        return f"dataset version {version}, validation file verified: SHA-256 matches"
    if check["v3_profile_matched"]:
        return (f"dataset version {version}: rows, columns and class counts match, SHA-256 not compared, so file "
                f"content and row order are unproven")
    return f"dataset version {version}, NOT V3"


def prepare(det: str, pred_path, model_path, validation: Path, rows: int, chunksize: int) -> tuple:
    """(predictions, source record) for one classifier. Nothing is written here."""
    if model_path:
        model = load_model(model_path)
        inner = final_estimator(model)
        if OTHER_DETECTOR_CLASS[det] in type(inner).__name__:
            raise DatasetError(f"{det}: {model_path.name} holds a {type(inner).__name__}; are --dt-model and --rf-model "
                               f"swapped?", code="MODEL_DETECTOR_MISMATCH")
        preds = predict_in_chunks(model, validation, rows, chunksize)
        params = inner.get_params() if hasattr(inner, "get_params") else {}
        order_checked = getattr(model, "feature_names_in_", None) is not None
        if not order_checked:
            print(f"WARNING: {det}: the model has no saved feature names, so its column order was NOT checked. Compare "
                  f"the accuracy in the manifest with the reported result.", flush=True)
        return preds, {"kind": "model", "file_name": model_path.name, "sha256": sha256_file(model_path),
                       "class": type(model).__name__, "final_estimator_class": type(inner).__name__,
                       "feature_order_verified": order_checked,
                       "params": {k: recorded_param(params[k]) for k in RECORDED_PARAMS if k in params}}
    preds = load_predictions(pred_path, rows)
    return preds, {"kind": "predictions", "file_name": pred_path.name, "sha256": sha256_file(pred_path)}


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    sources = {"decision_tree": (a.dt_predictions, a.dt_model), "random_forest": (a.rf_predictions, a.rf_model)}
    for det, (preds, model) in sources.items():
        if bool(preds) == bool(model):
            raise NovaQwenError(f"{det}: give exactly one of predictions or model", code="BAD_ARGUMENTS")
        if model and not a.trust_model_files:
            raise NovaQwenError("loading a model file runs code from it; pass --trust-model-files only for models "
                                "you made yourself", code="BAD_ARGUMENTS")
    if a.chunksize < 1:
        raise NovaQwenError("--chunksize must be 1 or more", code="BAD_ARGUMENTS")
    require_version_spelling(a.dataset_version)
    if a.dataset_version != V3 and "v3" in a.dataset.lower():
        raise NovaQwenError(f"--dataset {a.dataset!r} names V3 but --dataset-version is {a.dataset_version!r}",
                            code="BAD_ARGUMENTS")
    _identifier(a.dataset, "dataset")
    if a.run_id is not None:
        _identifier(a.run_id, "run_id")
    for name in OUTPUT_NAMES:  # a real directory is the one target a rename cannot replace
        target = a.out_dir / name
        if target.is_dir() and not target.is_symlink():
            raise NovaQwenError(f"{target} is a directory, so the export cannot be written there", code="BAD_ARGUMENTS")
    chosen = [preds or model for preds, model in sources.values()]
    for path in chosen:  # by name first, so a reserved split is never hashed or loaded
        refuse_reserved(path)
        if not path.is_file():
            raise DatasetError(f"source file not found: {path}", code="DATASET_MISSING")
    both_models = bool(a.dt_model and a.rf_model)
    if chosen[0].resolve() == chosen[1].resolve() or (both_models and sha256_file(chosen[0]) == sha256_file(chosen[1])):
        raise NovaQwenError(f"the same file was given for decision_tree and random_forest ({chosen[0].name}, "
                            f"{chosen[1].name}); correlation would compare one classifier with itself",
                            code="SAME_CLASSIFIER_SOURCE")
    check = check_dataset(validation=a.validation, version=a.dataset_version,
                          expected_validation_sha256=a.expected_validation_sha256)
    if not check["passed"]:
        raise DatasetError(f"dataset check failed for version {a.dataset_version}: {check['problems']}",
                           details={"problems": check["problems"]})
    rows = check["validation"]["rows"]
    prepared = {det: prepare(det, pred_path, model_path, a.validation, rows, a.chunksize)
                for det, (pred_path, model_path) in sources.items()}
    identical = bool(np.array_equal(prepared["decision_tree"][0], prepared["random_forest"][0]))
    if identical:
        note = "the decision_tree and random_forest predictions are identical on every row"
        if a.dataset_version == V3:
            raise NovaQwenError(f"{note}, which the reported V3 experiment rules out; was one classifier's output "
                                f"given twice?", code="SAME_CLASSIFIER_SOURCE")
        print(f"WARNING: {note}, so correlation can show no disagreement. Check that one classifier's output was not "
              f"given twice.", flush=True)
    frame = pd.read_csv(a.validation, usecols=[*ALLOWED_EVIDENCE, LABEL_COLUMN])
    labels = frame.pop(LABEL_COLUMN).to_numpy(dtype=str)
    frame = frame[list(ALLOWED_EVIDENCE)]
    run_id = a.run_id or datetime.now(timezone.utc).strftime("validation-%Y%m%dT%H%M%S%fZ")
    manifest = {"report": "nova-qwen-export-manifest", "tool": "scripts/qwen_export_v3_alerts.py",
                "run_id": run_id, "dataset_field": a.dataset, "dataset_check": check,
                "v3_profile_matched": check["v3_profile_matched"], "v3_validation_verified": check["v3_validation_verified"],
                "predictions_identical": identical, "classifiers": {}}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = V3_VALIDATION_ROWS if a.dataset_version == V3 else None
    with tempfile.TemporaryDirectory(dir=a.out_dir, prefix=".export-staging-") as staging:
        staging = Path(staging)
        for det, (preds, source) in prepared.items():
            path = staging / f"{det}_alerts.json"
            with contextlib.redirect_stdout(io.StringIO()):  # the adapter prints this staging path, deleted below
                export_alerts(features=frame, predictions=preds, detector=det, output_path=path, run_id=run_id,
                              dataset=a.dataset)
            export = load_export(path, expected_detector=det)
            alignment = check_alignment(export, a.validation, expected_rows=expected_rows,
                                        require_alerts=a.dataset_version == V3)
            if not alignment["passed"]:
                raise DatasetError(f"{det}: the alignment check failed: {alignment['problems']}")
            classes: dict = {}
            for alert in export.alerts:
                classes[alert.predicted_class] = classes.get(alert.predicted_class, 0) + 1
            scores = metrics(labels, preds)
            manifest["classifiers"][det] = {"source": source, "export": export.source_record(),
                                            "predicted_attack_classes": classes, "alignment": alignment,
                                            "metrics_on_validation": scores}
            print(f"{det}: {len(export.alerts)} predicted attack alerts from {rows} validation rows; "
                  f"accuracy {scores['accuracy']:.4f}", flush=True)
        write_json_atomic(staging / "export_manifest.json", manifest)
        moved: list = []
        try:
            for name in OUTPUT_NAMES:
                os.replace(staging / name, a.out_dir / name)
                moved.append(name)
        except OSError as exc:
            raise NovaQwenError(f"could not replace {name} ({exc}). Already replaced by this run: {moved or 'nothing'}, so "
                                f"{a.out_dir} may hold a mixed set. Fix the cause and run the export again.",
                                code="EXPORT_MOVE_INCOMPLETE") from exc
    for det in DETECTORS:
        print(f"{det}: export written to {a.out_dir / f'{det}_alerts.json'}", flush=True)
    print(f"EXPORT DONE: {a.out_dir / 'export_manifest.json'} ({identity_note(a.dataset_version, check)})")
    return 0


def entry(argv=None) -> int:
    try:
        return main(argv)
    except NovaQwenError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        print("EXPORT NOT DONE")
        return 2
    except Exception as exc:  # unreadable files, bad model files, model.predict errors, an unwritable --out-dir
        print(json.dumps({"error": {"code": "UNEXPECTED_ERROR", "type": type(exc).__name__, "message": str(exc)}},
                         indent=2), file=sys.stderr)
        print("EXPORT NOT DONE")
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))  # so the staging folder is removed
    sys.exit(entry())
